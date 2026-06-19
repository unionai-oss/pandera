"""Generate synthetic data from a schema definition.

*new in 0.6.0*

This module is responsible for generating data based on the type and check
constraints specified in a ``pandera`` schema. It's built on top of the
`hypothesis <https://hypothesis.readthedocs.io/en/latest/index.html>`_ package
to compose strategies given multiple checks specified in a schema.

See the :ref:`user guide <data-synthesis-strategies>` for more details.
"""

import inspect
import operator
import re
import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from copy import deepcopy
from functools import partial, wraps
from typing import (
    Any,
    Optional,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandas as pd

from pandera.dtypes import (
    DataType,
    is_category,
    is_complex,
    is_datetime,
    is_float,
    is_timedelta,
)
from pandera.engines import numpy_engine, pandas_engine
from pandera.errors import BaseStrategyOnlyError, SchemaDefinitionError
from pandera.strategies.base_strategies import (
    CONSTRAINT_DISPATCHER,
    HAS_HYPOTHESIS,
    STRATEGY_DISPATCHER,
)
from pandera.strategies.constraints import (
    UNSET,
    ConstraintConflictError,
    FieldConstraints,
)

if HAS_HYPOTHESIS:
    import hypothesis
    import hypothesis.extra.numpy as npst
    import hypothesis.extra.pandas as pdst
    import hypothesis.strategies as st
    from hypothesis.internal.filtering import max_len, min_len
    from hypothesis.strategies import SearchStrategy, composite
else:
    from pandera.strategies.base_strategies import SearchStrategy, composite


StrategyFn = Callable[..., SearchStrategy]
# Fix this when modules have been re-organized to avoid circular imports
IndexComponent = Any
F = TypeVar("F", bound=Callable)


def _mask(
    val: Union[pd.Series, pd.Index], null_mask: list[bool]
) -> Union[pd.Series, pd.Index]:
    if pd.api.types.is_timedelta64_dtype(val):  # type: ignore [arg-type]
        return val.mask(null_mask, pd.NaT)  # type: ignore [union-attr,arg-type]
    elif val.dtype == pd.StringDtype():  # type: ignore [call-arg]
        return val.mask(null_mask, pd.NA)  # type: ignore [union-attr,arg-type]
    return val.mask(null_mask)  # type: ignore [union-attr]


@composite
def null_field_masks(draw, strategy: SearchStrategy | None):
    """Strategy for masking a column/index with null values.

    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    """
    val = draw(strategy)
    size = val.shape[0]
    null_mask = draw(st.lists(st.booleans(), min_size=size, max_size=size))
    if isinstance(val, pd.Index):
        val = val.to_series()
        val = _mask(val, null_mask)
        return pd.Index(val)
    return _mask(val, null_mask)


@composite
def null_dataframe_masks(
    draw,
    strategy: SearchStrategy | None,
    nullable_columns: dict[str, bool],
):
    """Strategy for masking a values in a pandas DataFrame.

    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param nullable_columns: dictionary where keys are column names and
        values indicate whether that column is nullable.
    """
    val = draw(strategy)
    size = val.shape[0]
    columns_strat = []
    for name, nullable in nullable_columns.items():
        element_st = st.booleans() if nullable else st.just(False)
        columns_strat.append(
            pdst.column(
                name=name,
                elements=element_st,
                dtype=bool,
                fill=st.just(False),
            )
        )
    mask_st = pdst.data_frames(
        columns=columns_strat,
        index=pdst.range_indexes(min_size=size, max_size=size),
    )
    null_mask = draw(mask_st)
    for column in val:
        val[column] = _mask(val[column], null_mask[column])
    return val


@composite
def set_pandas_index(
    draw,
    df_or_series_strat: SearchStrategy,
    index: IndexComponent,
):
    """Sets Index or MultiIndex object to pandas Series or DataFrame."""
    df_or_series = draw(df_or_series_strat)
    df_or_series.index = draw(index.strategy(size=df_or_series.shape[0]))
    return df_or_series


def verify_dtype(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    schema_type: str,
    name: str | None,
):
    """Verify that pandera_dtype argument is not None."""
    if pandera_dtype is None:
        raise SchemaDefinitionError(
            f"'{schema_type}' schema with name '{name}' has no specified "
            "dtype. You need to specify one in order to synthesize "
            "data from a strategy."
        )


def strategy_import_error(fn: F) -> F:
    """Decorator to generate input error if dependency is missing."""

    @wraps(fn)
    def _wrapper(*args, **kwargs):
        if not HAS_HYPOTHESIS:  # pragma: no cover
            raise ImportError(
                'Strategies for generating data requires "hypothesis" to be \n'
                "installed. You can install pandera together with the strategies \n"
                "dependencies with:\n"
                "pip install pandera[strategies]"
            )
        return fn(*args, **kwargs)

    return cast(F, _wrapper)


def register_check_strategy(strategy_fn: StrategyFn):
    """Decorate a Check method with a strategy.

    This should be applied to a built-in :class:`~pandera.api.checks.Check` method.

    :param strategy_fn: add strategy to a check, using check statistics to
        generate a ``hypothesis`` strategy.
    """

    def register_check_strategy_decorator(class_method):
        """Decorator that wraps Check class method."""

        @wraps(class_method)
        def _wrapper(cls, *args, **kwargs):
            check = class_method(cls, *args, **kwargs)
            if check.statistics is None:
                raise AttributeError(
                    "check object doesn't have a defined statistics property. "
                    "Use the checks.register_check_statistics decorator to "
                    f"specify the statistics for the {class_method.__name__} "
                    "method."
                )
            check.strategy = strategy_fn
            return check

        return _wrapper

    return register_check_strategy_decorator


def register_check_constraint(constraint_fn: Callable):
    """Decorate a Check method with a constraint adapter.

    Constraint adapters take the check's ``statistics`` as kwargs and
    return a
    :class:`~pandera.strategies.constraints.FieldConstraints` value
    describing the bounds/membership/regex constraints the check
    encodes. When a check has a constraint adapter, the strategy
    builder prefers it over ``check.strategy`` and merges its output
    with sibling constraints, emitting a single hypothesis strategy
    (no per-check ``.filter`` chaining). See
    ``specs/optimized-strategies.md``.

    :param constraint_fn: callable with signature
        ``(**statistics) -> FieldConstraints``.
    """

    def register_check_constraint_decorator(class_method):
        """Decorator that wraps Check class method."""

        @wraps(class_method)
        def _wrapper(cls, *args, **kwargs):
            check = class_method(cls, *args, **kwargs)
            if check.statistics is None:
                raise AttributeError(
                    "check object doesn't have a defined statistics "
                    "property. Use the checks.register_check_statistics "
                    "decorator to specify the statistics for the "
                    f"{class_method.__name__} method."
                )
            check.constraint = constraint_fn
            return check

        return _wrapper

    return register_check_constraint_decorator


# Values taken from
# https://hypothesis.readthedocs.io/en/latest/_modules/hypothesis/extra/numpy.html#from_dtype  # noqa
# NOTE: We're reducing the range here by an order of magnitude to avoid overflows
# when synthesizing timezone-aware timestamps.
MIN_DT_VALUE = -(2**62) + 1
MAX_DT_VALUE = 2**62 - 1


def _is_datetime_tz(pandera_dtype: DataType) -> bool:
    native_type = getattr(pandera_dtype, "type", None)
    return isinstance(native_type, pd.DatetimeTZDtype)


def _datetime_strategy(
    dtype: Union[np.dtype, pd.DatetimeTZDtype], strategy
) -> SearchStrategy:
    if isinstance(dtype, pd.DatetimeTZDtype):

        def _to_datetime(value) -> pd.DatetimeTZDtype:
            if isinstance(value, pd.Timestamp):
                return value.tz_convert(tz=dtype.tz)  # type: ignore[union-attr,return-value]
            return pd.Timestamp(value, unit=dtype.unit, tz=dtype.tz)  # type: ignore[union-attr,return-value]

        return st.builds(_to_datetime, strategy)
    else:
        res = (
            st.just(dtype.str.split("[")[-1][:-1])
            if "[" in dtype.str
            else st.sampled_from(npst.TIME_RESOLUTIONS)
        )
        return st.builds(dtype.type, strategy, res)


def convert_dtype(array: Union[pd.Series, pd.Index], col_dtype: Any):
    """Convert datatypes of an array (series or index)."""
    if str(col_dtype).startswith("datetime64"):
        try:
            return array.astype(col_dtype)
        except TypeError:
            tz = getattr(col_dtype, "tz", None)
            if tz is None:
                tz_match = re.match(r"datetime64\[ns, (.+)\]", str(col_dtype))
                tz = None if not tz_match else tz_match.group(1)

            if isinstance(array, pd.Index):
                return array.tz_localize(tz)  # type: ignore [attr-defined]
            return array.dt.tz_localize(tz)  # type: ignore [union-attr]
    return array.astype(col_dtype)


def convert_dtypes(df: pd.DataFrame, col_dtypes: dict[Union[int, str], Any]):
    """Convert datatypes of a dataframe."""
    for col_name, col_dtype in col_dtypes.items():
        array: pd.Series = df[col_name]  # type: ignore[assignment]
        df[col_name] = convert_dtype(array, col_dtype)

    return df


def numpy_time_dtypes(
    dtype: Union[np.dtype, pd.DatetimeTZDtype], min_value=None, max_value=None
):
    """Create numpy strategy for datetime and timedelta data types.

    :param dtype: numpy datetime or timedelta datatype
    :param min_value: minimum value of the datatype to create
    :param max_value: maximum value of the datatype to create
    :returns: ``hypothesis`` strategy
    """

    def _to_unix(value: Any) -> int:
        if dtype.type is np.timedelta64:
            return pd.Timedelta(value).value  # type: ignore[attr-defined]
        return pd.Timestamp(value).value  # type: ignore[attr-defined]

    min_value = MIN_DT_VALUE if min_value is None else _to_unix(min_value)
    max_value = MAX_DT_VALUE if max_value is None else _to_unix(max_value)
    return _datetime_strategy(dtype, st.integers(min_value, max_value))


def numpy_complex_dtypes(
    dtype,
    min_value: complex = complex(0, 0),
    max_value: complex | None = None,
    allow_infinity: bool | None = None,
    allow_nan: bool | None = None,
):
    """Create numpy strategy for complex numbers.

    :param dtype: numpy complex number datatype
    :param min_value: minimum value, must be complex number
    :param max_value: maximum value, must be complex number
    :returns: ``hypothesis`` strategy
    """
    max_real: float | None
    max_imag: float | None
    if max_value:
        max_real = max_value.real
        max_imag = max_value.imag
    else:
        max_real = max_imag = None
    if dtype.itemsize == 8:
        width = 32
    else:
        width = 64

    # switch min and max values for imaginary if min value > max value
    if max_imag is not None and min_value.imag > max_imag:
        min_imag = max_imag
        max_imag = min_value.imag
    else:
        min_imag = min_value.imag

    strategy = st.builds(
        complex,
        st.floats(
            min_value=min_value.real,
            max_value=max_real,
            width=width,
            allow_infinity=allow_infinity,
            allow_nan=allow_nan,
        ),
        st.floats(
            min_value=min_imag,
            max_value=max_imag,
            width=width,
            allow_infinity=allow_infinity,
            allow_nan=allow_nan,
        ),
    ).map(dtype.type)

    @st.composite
    def build_complex(draw):
        value = draw(strategy)
        hypothesis.assume(min_value <= value)
        if max_value is not None:
            hypothesis.assume(max_value >= value)
        return value

    return build_complex()


def to_numpy_dtype(pandera_dtype: DataType):
    """Convert a :class:`~pandera.dtypes.DataType` to numpy dtype compatible
    with hypothesis."""
    try:
        np_dtype = pandas_engine.Engine.numpy_dtype(pandera_dtype)
    except TypeError as err:
        if is_datetime(pandera_dtype):
            return np.dtype("datetime64[ns]")

        raise TypeError(
            f"Data generation for the '{pandera_dtype}' data type is "
            "currently unsupported."
        ) from err

    if np_dtype == np.dtype("object") or str(pandera_dtype) == "str":
        np_dtype = np.dtype(str)
    return np_dtype


def _is_string_dtype(pandera_dtype: DataType) -> bool:
    """Check if a dtype is a string type that may use pyarrow backend.

    In pandas 3.0+, the default string dtype uses pyarrow backend,
    which cannot handle Unicode surrogate characters. We need to filter
    surrogates for all string types to be safe.
    """
    dtype_str = str(pandera_dtype).lower()
    # Check for any string-like type
    if dtype_str in {"str", "string", "object"}:
        return True
    if dtype_str.startswith("string"):
        return True
    # Check if it's a STRING type (any storage)
    if isinstance(pandera_dtype, pandas_engine.STRING):
        return True
    # Check if it's NpString
    if isinstance(pandera_dtype, pandas_engine.NpString):
        return True
    return False


def _remove_surrogates(s):
    """Remove Unicode surrogate characters from a string.

    Surrogates (U+D800 to U+DFFF) are not valid Unicode code points
    and cannot be encoded by pyarrow.

    Returns the same type as input (preserves numpy string types).
    """
    s_str = str(s)
    cleaned = "".join(c for c in s_str if not ("\ud800" <= c <= "\udfff"))
    # Preserve numpy string type if input was numpy
    if hasattr(s, "dtype"):
        return np.str_(cleaned)
    return cleaned


def _str_no_surrogates(obj) -> str:
    """Convert object to string and remove surrogates.

    This is used when generating string data for pyarrow-backed dtypes
    which cannot handle Unicode surrogates.
    """
    s_str = str(obj)
    return "".join(c for c in s_str if not ("\ud800" <= c <= "\udfff"))


def pandas_dtype_strategy(
    pandera_dtype: DataType,
    strategy: SearchStrategy | None = None,
    **kwargs,
) -> SearchStrategy:
    """Strategy to generate data from a :class:`pandera.dtypes.DataType`.

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :kwargs: key-word arguments passed into
        `hypothesis.extra.numpy.from_dtype <https://hypothesis.readthedocs.io/en/latest/numpy.html#hypothesis.extra.numpy.from_dtype>`_ .
        For datetime, timedelta, and complex number datatypes, these arguments
        are passed into :func:`~pandera.strategies.numpy_time_dtypes` and
        :func:`~pandera.strategies.numpy_complex_dtypes`.
    :returns: ``hypothesis`` strategy
    """

    def compat_kwargs(*args):
        return {k: v for k, v in kwargs.items() if k in args}

    # hypothesis doesn't support categoricals or objects, so we'll will need to
    # build a pandera-specific solution.
    if is_category(pandera_dtype):
        raise TypeError(
            "data generation for the Category dtype is currently "
            "unsupported. Consider using a string or int dtype and "
            "Check.isin(values) to ensure a finite set of values."
        )

    np_dtype = to_numpy_dtype(pandera_dtype)
    is_string_type = _is_string_dtype(pandera_dtype)

    if strategy is not None:
        if _is_datetime_tz(pandera_dtype):
            return _datetime_strategy(pandera_dtype.type, strategy)  # type: ignore
        result = strategy.map(np_dtype.type)
        # Filter surrogates for string types (pandas 3.0+ uses pyarrow by default)
        if is_string_type:
            result = result.map(_remove_surrogates)
        return result
    elif is_datetime(pandera_dtype) or is_timedelta(pandera_dtype):
        return numpy_time_dtypes(
            pandera_dtype.type if _is_datetime_tz(pandera_dtype) else np_dtype,  # type: ignore
            **compat_kwargs("min_value", "max_value"),
        )
    elif is_complex(pandera_dtype):
        return numpy_complex_dtypes(
            np_dtype,
            **compat_kwargs(
                "min_value", "max_value", "allow_infinity", "allow_nan"
            ),
        )

    result = npst.from_dtype(
        np_dtype,
        **{  # type: ignore
            "allow_nan": False,
            "allow_infinity": False,
            **kwargs,
        },
    )
    # Filter surrogates for string types (pandas 3.0+ uses pyarrow by default)
    if is_string_type:
        result = result.map(_remove_surrogates)
    return result


def _is_temporal_dtype(pandera_dtype: DataType) -> bool:
    return is_datetime(pandera_dtype) or is_timedelta(pandera_dtype)


def _close_bound(
    pandera_dtype: DataType,
    value: Any,
    exclude: bool,
    side: str,
) -> tuple[Any, bool]:
    """Lower an exclusive bound to a closed bound when the dtype demands it.

    Hypothesis only supports ``exclude_min`` / ``exclude_max`` for
    floats and complex numbers. For integer / temporal dtypes we
    translate ``(value, exclude=True)`` into a closed-bound
    representation so we can pass kwargs that the underlying
    ``hypothesis.extra.numpy.from_dtype`` (or our
    ``numpy_time_dtypes`` helper) accepts.

    :param pandera_dtype: the field's dtype.
    :param value: the original bound value.
    :param exclude: whether the bound is exclusive in the original
        check.
    :param side: ``"min"`` or ``"max"``.
    :returns: ``(adjusted_value, exclude_flag)`` where ``exclude_flag``
        is the value to forward as ``exclude_min`` /
        ``exclude_max`` (only meaningful for float/complex dtypes).
    """
    if is_float(pandera_dtype) or is_complex(pandera_dtype):
        return value, exclude

    if not exclude:
        return value, False

    if _is_temporal_dtype(pandera_dtype):
        delta = pd.Timedelta(1, unit="ns")
        if side == "min":
            return pd.Timestamp(value) + delta if is_datetime(
                pandera_dtype
            ) else pd.Timedelta(value) + delta, False
        return pd.Timestamp(value) - delta if is_datetime(
            pandera_dtype
        ) else pd.Timedelta(value) - delta, False

    # Integer / boolean / other discrete dtypes.
    try:
        if side == "min":
            return value + 1, False
        return value - 1, False
    except TypeError:
        # Fall back: leave bound unchanged. The aggregator's residual
        # filter catches over-generation in this edge case.
        return value, False


_PREFIX_PREAMBLE = r"\A(?:"
_PREFIX_SUFFIX = r").*\Z"
_SUFFIX_PREAMBLE = r"\A.*(?:"
_SUFFIX_SUFFIX = r")\Z"


def _split_prefix(pattern: str) -> str | None:
    """Return ``X`` if ``pattern`` is ``\\A(?:X).*\\Z`` else ``None``."""
    if (
        pattern.startswith(_PREFIX_PREAMBLE)
        and pattern.endswith(_PREFIX_SUFFIX)
        and len(pattern) > len(_PREFIX_PREAMBLE) + len(_PREFIX_SUFFIX)
    ):
        return pattern[len(_PREFIX_PREAMBLE) : -len(_PREFIX_SUFFIX)]
    return None


def _split_suffix(pattern: str) -> str | None:
    """Return ``X`` if ``pattern`` is ``\\A.*(?:X)\\Z`` else ``None``."""
    if (
        pattern.startswith(_SUFFIX_PREAMBLE)
        and pattern.endswith(_SUFFIX_SUFFIX)
        and len(pattern) > len(_SUFFIX_PREAMBLE) + len(_SUFFIX_SUFFIX)
    ):
        return pattern[len(_SUFFIX_PREAMBLE) : -len(_SUFFIX_SUFFIX)]
    return None


def _try_structural_merge(
    fullmatch_patterns: tuple[str, ...],
    search_patterns: tuple[str, ...],
) -> tuple[str, bool] | None:
    """Try to combine patterns into a single concatenated regex.

    Recognises ``str_startswith`` (``\\A(?:X).*\\Z``) and
    ``str_endswith`` (``\\A.*(?:X)\\Z``) shapes plus generic search
    patterns and concatenates them in the order
    ``\\A<prefix>.*<contains_1>.*<contains_2>...<suffix>\\Z`` so
    hypothesis's regex generator drives directly toward valid strings
    (no lookahead-induced rejection sampling).

    Returns ``None`` when the patterns can't be safely concatenated
    (e.g. multiple distinct ``startswith`` / ``endswith`` constraints,
    or arbitrary ``str_matches`` patterns that aren't simple prefix /
    suffix shapes).
    """
    prefixes: list[str] = []
    suffixes: list[str] = []
    other_fullmatch: list[str] = []
    for p in fullmatch_patterns:
        if (px := _split_prefix(p)) is not None:
            prefixes.append(px)
            continue
        if (sx := _split_suffix(p)) is not None:
            suffixes.append(sx)
            continue
        other_fullmatch.append(p)

    if other_fullmatch:
        return None
    if len(prefixes) > 1 or len(suffixes) > 1:
        # Two distinct startswith / endswith are unsatisfiable in
        # general; defer to the lookahead path so the conflict surfaces
        # as a generation failure rather than a malformed regex.
        return None

    parts: list[str] = [r"\A"]
    if prefixes:
        parts.append(rf"(?:{prefixes[0]})")
    parts.append(".*")
    for p in search_patterns:
        parts.append(rf"(?:{p}).*")
    if suffixes:
        parts.append(rf"(?:{suffixes[0]})")
    parts.append(r"\Z")
    return "".join(parts), True


def _combine_patterns(
    fullmatch_patterns: tuple[str, ...],
    search_patterns: tuple[str, ...],
) -> tuple[str, bool] | None:
    """Combine multiple regex constraints into a single anchored regex.

    Strategy (in priority order):

    1. Single-pattern pass-through (most efficient for hypothesis).
    2. Structural concatenation of ``str_startswith`` /
       ``str_endswith`` / ``str_contains`` shapes.
    3. Zero-width lookahead conjunction (correct but generates
       inefficiently — included as a fallback).

    Returns ``None`` when any input pattern fails ``re.compile`` so
    callers can fall back to ``.filter``.

    :returns: ``(combined_pattern, fullmatch_flag)`` or ``None``.
    """
    fullmatch_patterns = tuple(fullmatch_patterns)
    search_patterns = tuple(search_patterns)

    if len(fullmatch_patterns) + len(search_patterns) == 0:
        return None

    for p in (*fullmatch_patterns, *search_patterns):
        try:
            re.compile(p)
        except re.error:
            return None

    # Single-pattern fast path: avoid any wrapping.
    if len(fullmatch_patterns) == 1 and not search_patterns:
        return fullmatch_patterns[0], True
    if not fullmatch_patterns and len(search_patterns) == 1:
        return search_patterns[0], False

    structural = _try_structural_merge(fullmatch_patterns, search_patterns)
    if structural is not None:
        return structural

    parts: list[str] = []
    for p in fullmatch_patterns:
        parts.append(rf"(?=\A(?:{p})\Z)")
    for p in search_patterns:
        parts.append(rf"(?=.*(?:{p}))")

    return "".join(parts) + r".*", True


def _compile_string_strategy(
    pandera_dtype: DataType, constraints: FieldConstraints
) -> SearchStrategy:
    """Compile string-shaped ``FieldConstraints`` to a single strategy.

    Combines all regex constraints into one anchored ``st.from_regex``
    via zero-width lookahead-AND when every pattern compiles cleanly.
    Length constraints are passed as ``min_size`` / ``max_size`` to
    ``st.text(...)`` when there are no regex patterns; when both regex
    and length constraints are present, the length range is embedded
    into the combined regex as a lookahead so hypothesis's regex
    generator drives toward valid strings (rather than us filtering
    after the fact). Surrogate-stripping for pyarrow-backed strings is
    preserved.
    """
    np_type = to_numpy_dtype(pandera_dtype).type

    str_min = constraints.str_min_len
    str_max = constraints.str_max_len
    if constraints.str_exact_len is not None:
        str_min = constraints.str_exact_len
        str_max = constraints.str_exact_len

    has_regex = bool(constraints.regex_fullmatch or constraints.regex_search)

    length_filter = None
    if str_min is not None or str_max is not None:
        lo = str_min if str_min is not None else 0
        hi = str_max if str_max is not None else 1 << 30
        length_filter = lambda s, lo=lo, hi=hi: lo <= len(s) <= hi  # noqa: E731

    if has_regex:
        combined = _combine_patterns(
            constraints.regex_fullmatch,
            constraints.regex_search,
        )
        if combined is not None:
            pattern, fullmatch = combined
            strat = st.from_regex(pattern, fullmatch=fullmatch).map(np_type)
        else:
            # Fall back: pick first uncompilable pattern and chain the
            # rest as filters. Mirrors today's per-check behaviour for
            # pathological inputs only.
            patterns = [
                *constraints.regex_fullmatch,
                *constraints.regex_search,
            ]
            strat = st.from_regex(patterns[0], fullmatch=True).map(np_type)
            for p in patterns[1:]:
                strat = strat.filter(re.compile(p).search)
    elif str_min is not None or str_max is not None:
        strat = st.text(min_size=str_min or 0, max_size=str_max).map(np_type)
    else:
        strat = pandas_dtype_strategy(pandera_dtype)

    if _is_string_dtype(pandera_dtype):
        strat = strat.map(_remove_surrogates)

    # Apply the length filter *after* surrogate removal so the
    # post-mapping string length is what gets validated.
    if length_filter is not None and (
        has_regex or _is_string_dtype(pandera_dtype)
    ):
        strat = strat.filter(length_filter)

    if constraints.notin:
        forbidden = constraints.notin
        strat = strat.filter(lambda v, f=forbidden: v not in f)

    for _name, predicate in constraints.residual_filters:
        strat = strat.filter(predicate)

    return strat


def compile_field_strategy(
    pandera_dtype: DataType, constraints: FieldConstraints
) -> SearchStrategy:
    """Compile a merged ``FieldConstraints`` into a single hypothesis strategy.

    All dtype-specific bridging (datetime tz, complex, time
    resolutions, surrogate handling) is delegated to
    :func:`pandas_dtype_strategy`; this helper only translates the
    aggregated constraints into the appropriate kwargs and parent
    strategy.

    :param pandera_dtype: the field's dtype.
    :param constraints: merged ``FieldConstraints`` produced by
        bucketing every check on the field.
    :returns: a ``hypothesis`` strategy with at most one trailing
        ``.filter`` per residual predicate (built-in checks contribute
        zero residuals).
    :raises ConstraintConflictError: when the merged constraint set
        is jointly unsatisfiable (e.g. ``isin`` set pruned to empty by
        bounds + ``notin``).
    """
    constraints = constraints.apply_post_merge_hooks()

    # 1. Equality short-circuits everything.
    if constraints.eq is not UNSET:
        if constraints.eq in constraints.notin:
            raise ConstraintConflictError(
                f"eq={constraints.eq!r} conflicts with notin"
            )
        return pandas_dtype_strategy(pandera_dtype, st.just(constraints.eq))

    # 2. Membership: prune the allowed set against bounds/notin so the
    # underlying ``st.sampled_from`` already only sees valid values.
    if constraints.isin is not None:
        allowed = set(constraints.isin) - set(constraints.notin)
        if constraints.min_value is not UNSET:
            cmp = operator.lt if constraints.exclude_min else operator.le
            allowed = {v for v in allowed if cmp(constraints.min_value, v)}
        if constraints.max_value is not UNSET:
            cmp = operator.gt if constraints.exclude_max else operator.ge
            allowed = {v for v in allowed if cmp(constraints.max_value, v)}
        if not allowed:
            raise ConstraintConflictError(
                "isin/notin/bounds intersection is empty"
            )
        sampled = st.sampled_from(sorted(allowed))
        strat = pandas_dtype_strategy(pandera_dtype, sampled)
        for _name, predicate in constraints.residual_filters:
            strat = strat.filter(predicate)
        return strat

    # 3. Strings.
    if _is_string_dtype(pandera_dtype):
        return _compile_string_strategy(pandera_dtype, constraints)

    # 4. Numeric / temporal: build kwargs for pandas_dtype_strategy.
    kwargs: dict[str, Any] = {}
    if constraints.min_value is not UNSET:
        min_v, excl_min = _close_bound(
            pandera_dtype,
            constraints.min_value,
            constraints.exclude_min,
            side="min",
        )
        kwargs["min_value"] = min_v
        if is_float(pandera_dtype) or is_complex(pandera_dtype):
            kwargs["exclude_min"] = excl_min
    if constraints.max_value is not UNSET:
        max_v, excl_max = _close_bound(
            pandera_dtype,
            constraints.max_value,
            constraints.exclude_max,
            side="max",
        )
        kwargs["max_value"] = max_v
        if is_float(pandera_dtype) or is_complex(pandera_dtype):
            kwargs["exclude_max"] = excl_max
    if is_float(pandera_dtype) or is_complex(pandera_dtype):
        kwargs["allow_nan"] = constraints.allow_nan
        kwargs["allow_infinity"] = constraints.allow_infinity

    strat = pandas_dtype_strategy(pandera_dtype, **kwargs)

    # 5. Single trailing ``notin`` filter (only when the membership
    # path above wasn't taken).
    if constraints.notin:
        forbidden = constraints.notin
        strat = strat.filter(lambda v, f=forbidden: v not in f)

    # 6. Residual opaque predicates from custom checks.
    for _name, predicate in constraints.residual_filters:
        strat = strat.filter(predicate)

    return strat


def eq_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: SearchStrategy | None = None,
    *,
    value: Any,
) -> SearchStrategy:
    """Strategy to generate a single value.

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param value: value to generate.
    :returns: ``hypothesis`` strategy
    """
    # override strategy preceding this one and generate value of the same type

    return pandas_dtype_strategy(pandera_dtype, st.just(value))


def ne_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: SearchStrategy | None = None,
    *,
    value: Any,
) -> SearchStrategy:
    """Strategy to generate anything except for a particular value.

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param value: value to avoid.
    :returns: ``hypothesis`` strategy
    """
    if strategy is None:
        strategy = pandas_dtype_strategy(pandera_dtype)
    return strategy.filter(partial(operator.ne, value))


def gt_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: SearchStrategy | None = None,
    *,
    min_value: Union[int, float],
) -> SearchStrategy:
    """Strategy to generate values greater than a minimum value.

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param min_value: generate values larger than this.
    :returns: ``hypothesis`` strategy
    """
    if strategy is None:
        strategy = pandas_dtype_strategy(
            pandera_dtype,
            min_value=min_value,
            exclude_min=True if is_float(pandera_dtype) else None,
        )
    return strategy.filter(partial(operator.lt, min_value))


def ge_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: SearchStrategy | None = None,
    *,
    min_value: Union[int, float],
) -> SearchStrategy:
    """Strategy to generate values greater than or equal to a minimum value.

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param min_value: generate values greater than or equal to this.
    :returns: ``hypothesis`` strategy
    """
    if strategy is None:
        return pandas_dtype_strategy(
            pandera_dtype,
            min_value=min_value,
            exclude_min=False if is_float(pandera_dtype) else None,
        )
    return strategy.filter(partial(operator.le, min_value))


def lt_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: SearchStrategy | None = None,
    *,
    max_value: Union[int, float],
) -> SearchStrategy:
    """Strategy to generate values less than a maximum value.

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param max_value: generate values less than this.
    :returns: ``hypothesis`` strategy
    """
    if strategy is None:
        strategy = pandas_dtype_strategy(
            pandera_dtype,
            max_value=max_value,
            exclude_max=True if is_float(pandera_dtype) else None,
        )
    return strategy.filter(partial(operator.gt, max_value))


def le_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: SearchStrategy | None = None,
    *,
    max_value: Union[int, float],
) -> SearchStrategy:
    """Strategy to generate values less than or equal to a maximum value.

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param max_value: generate values less than or equal to this.
    :returns: ``hypothesis`` strategy
    """
    if strategy is None:
        return pandas_dtype_strategy(
            pandera_dtype,
            max_value=max_value,
            exclude_max=False if is_float(pandera_dtype) else None,
        )
    return strategy.filter(partial(operator.ge, max_value))


def in_range_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: SearchStrategy | None = None,
    *,
    min_value: Union[int, float],
    max_value: Union[int, float],
    include_min: bool = True,
    include_max: bool = True,
) -> SearchStrategy:
    """Strategy to generate values within a particular range.

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param min_value: generate values greater than this.
    :param max_value: generate values less than this.
    :param include_min: include min_value in generated data.
    :param include_max: include max_value in generated data.
    :returns: ``hypothesis`` strategy
    """
    if strategy is None:
        return pandas_dtype_strategy(
            pandera_dtype,
            min_value=min_value,
            max_value=max_value,
            exclude_min=not include_min,
            exclude_max=not include_max,
        )
    min_op = operator.le if include_min else operator.lt
    max_op = operator.ge if include_max else operator.gt
    return strategy.filter(partial(min_op, min_value)).filter(
        partial(max_op, max_value)
    )


def isin_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: SearchStrategy | None = None,
    *,
    allowed_values: Sequence[Any],
) -> SearchStrategy:
    """Strategy to generate values within a finite set.

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param allowed_values: set of allowable values.
    :returns: ``hypothesis`` strategy
    """
    if strategy is None:
        return pandas_dtype_strategy(
            pandera_dtype, st.sampled_from(allowed_values)
        )
    return strategy.filter(lambda x: x in allowed_values)


def notin_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: SearchStrategy | None = None,
    *,
    forbidden_values: Sequence[Any],
) -> SearchStrategy:
    """Strategy to generate values excluding a set of forbidden values

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param forbidden_values: set of forbidden values.
    :returns: ``hypothesis`` strategy
    """
    if strategy is None:
        strategy = pandas_dtype_strategy(pandera_dtype)
    return strategy.filter(lambda x: x not in forbidden_values)


def str_matches_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: SearchStrategy | None = None,
    *,
    pattern: str,
) -> SearchStrategy:
    """Strategy to generate strings that patch a regex pattern.

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param pattern: regex pattern.
    :returns: ``hypothesis`` strategy
    """
    if strategy is None:
        return st.from_regex(pattern, fullmatch=True).map(
            to_numpy_dtype(pandera_dtype).type
        )
    return strategy.filter(re.compile(pattern).fullmatch)


def str_contains_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: SearchStrategy | None = None,
    *,
    pattern: str,
) -> SearchStrategy:
    """Strategy to generate strings that contain a particular pattern.

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param pattern: regex pattern.
    :returns: ``hypothesis`` strategy
    """
    if strategy is None:
        return st.from_regex(pattern, fullmatch=False).map(
            to_numpy_dtype(pandera_dtype).type
        )
    return strategy.filter(re.compile(pattern).search)


def str_startswith_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: SearchStrategy | None = None,
    *,
    string: str,
) -> SearchStrategy:
    """Strategy to generate strings that start with a specific string pattern.

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param string: string pattern.
    :returns: ``hypothesis`` strategy
    """
    pattern = rf"\A(?:{string})"
    if strategy is None:
        return st.from_regex(pattern, fullmatch=False).map(
            to_numpy_dtype(pandera_dtype).type
        )
    return strategy.filter(re.compile(pattern).search)


def str_endswith_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: SearchStrategy | None = None,
    *,
    string: str,
) -> SearchStrategy:
    """Strategy to generate strings that end with a specific string pattern.

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param string: string pattern.
    :returns: ``hypothesis`` strategy
    """
    pattern = rf"(?:{string})\Z"
    if strategy is None:
        return st.from_regex(pattern, fullmatch=False).map(
            to_numpy_dtype(pandera_dtype).type
        )
    return strategy.filter(re.compile(pattern).search)


def str_length_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: SearchStrategy | None = None,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
    exact_value: int | None = None,
) -> SearchStrategy:
    """Strategy to generate strings of a particular length

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param min_value: minimum string length.
    :param max_value: maximum string length.
    :param exact_value: exact string length.
    :returns: ``hypothesis`` strategy
    """
    if exact_value is not None:
        min_value = exact_value
        max_value = exact_value

    if min_value is None and max_value is None:
        raise ValueError(
            "At least one of min_value/max_value or exact_value must be set"
        )

    if strategy is None:
        return st.text(min_size=min_value, max_size=max_value).map(
            to_numpy_dtype(pandera_dtype).type
        )

    if min_value is not None:
        strategy = strategy.filter(partial(min_len, min_value))
    if max_value is not None:
        strategy = strategy.filter(partial(max_len, max_value))
    return strategy


def _timestamp_to_datetime64_strategy(
    strategy: SearchStrategy,
) -> SearchStrategy:
    """Convert timestamp to numpy.datetime64
    Hypothesis only supports pure numpy dtypes but numpy.datetime64() truncates
    nanoseconds if given a pandas.Timestamp. We need to pass the unix epoch via
    the pandas.Timestamp.value attribute.
    """
    return st.builds(lambda x: np.datetime64(x.value, "ns"), strategy)


def field_element_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: SearchStrategy | None = None,
    *,
    checks: Sequence | None = None,
) -> SearchStrategy:
    """Strategy to generate elements of a column or index.

    Buckets ``checks`` into:

    1. constraint-providing checks (built-in or via ``Check.constraint``
       / ``register_check_method(constraint=...)``): aggregated into a
       single ``FieldConstraints`` and compiled to one hypothesis
       strategy via :func:`compile_field_strategy`. No ``.filter``
       chaining.
    2. legacy ``check.strategy`` callables: applied as a chained
       strategy on top of the merged base, preserving today's
       behaviour for users who haven't migrated.
    3. element-wise checks with no strategy / constraint: lowered to a
       single residual filter (warn the user that this is slow).

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. Reserved; passing
        a non-``None`` value raises :class:`BaseStrategyOnlyError`.
    :param checks: sequence of :class:`~pandera.api.checks.Check` s to
        constrain the values of the data in the column/index.
    :returns: ``hypothesis`` strategy.
    """
    if strategy is not None:
        raise BaseStrategyOnlyError(
            "The series strategy is a base strategy. You cannot specify the "
            "strategy argument to chain it to a parent strategy."
        )
    check_list = list(checks or [])

    constraint_acc = FieldConstraints()
    legacy_strategies: list[tuple[Any, Callable]] = []
    residuals: list[tuple[str, Callable]] = []

    for check in check_list:
        constraint_fn = getattr(check, "constraint", None) or (
            CONSTRAINT_DISPATCHER.get((check.name, pd.Series))
        )
        if constraint_fn is not None:
            try:
                constraint_acc = constraint_acc.merge(
                    constraint_fn(**check.statistics)
                )
            except ConstraintConflictError as exc:
                _raise_unsatisfiable(check_list, exc)
            continue

        legacy_strategy = check.strategy or STRATEGY_DISPATCHER.get(
            (check.name, pd.Series)
        )
        if legacy_strategy is not None:
            legacy_strategies.append((check, legacy_strategy))
            continue

        if check.element_wise:
            warnings.warn(
                "Element-wise check doesn't have a defined strategy."
                "Falling back to filtering drawn values based on the "
                "check definition. This can considerably slow down "
                "data-generation."
            )
            residuals.append((check.name, check._check_fn))
        # NOTE: vectorized checks with undefined strategies should be
        # handled by the series/dataframe strategy.

    if residuals:
        from dataclasses import replace as _rep

        constraint_acc = _rep(
            constraint_acc,
            residual_filters=(
                constraint_acc.residual_filters + tuple(residuals)
            ),
        )

    has_aggregated = not constraint_acc.is_empty()

    if has_aggregated:
        try:
            elements = compile_field_strategy(pandera_dtype, constraint_acc)
        except ConstraintConflictError as exc:
            _raise_unsatisfiable(check_list, exc)
    else:
        # No constraint adapters fired and no residuals: preserve the
        # legacy fold semantics so the first legacy strategy can take
        # its "no parent strategy" fast path. Constraint adapters are
        # the path that aggregates work into a single strategy call;
        # without them, today's per-check fold is the fastest available
        # path and we don't want to regress it before the built-ins are
        # migrated in stage 5.
        elements = None

    for check, legacy_strategy in legacy_strategies:
        if has_aggregated and _strategy_supports_base_mode(legacy_strategy):
            _warn_legacy_strategy_chained_once(check, legacy_strategy)
        elements = legacy_strategy(pandera_dtype, elements, **check.statistics)

    if elements is None:
        elements = pandas_dtype_strategy(pandera_dtype)

    return elements


def _raise_unsatisfiable(checks: Sequence, exc: ConstraintConflictError):
    """Translate a constraint conflict into a SchemaDefinitionError."""
    names = [str(getattr(c, "name", "?")) for c in checks]
    raise SchemaDefinitionError(
        f"Cannot construct a data-generation strategy for checks "
        f"{names}: constraints are jointly unsatisfiable ({exc})."
    ) from exc


# Module-level cache so the §9.3 ``DeprecationWarning`` fires at most
# once per ``(check.name, fn id)`` pair per process. Hypothesis runs
# each strategy many times during a draw loop and we don't want to
# spam the user.
_LEGACY_CHAINED_WARNED: set[tuple[str, int]] = set()


def _strategy_supports_base_mode(fn: Callable | None) -> bool:
    """Return ``True`` when ``fn`` advertises base-mode support.

    A legacy strategy callable is considered "base-mode-supporting" if
    its ``strategy`` parameter has ``None`` as the default value (the
    convention used by every built-in legacy strategy in this module).
    Pure chained-mode strategies (where ``strategy`` has no default)
    are unaffected by Stage 7's deprecation warning.
    """
    if fn is None:
        return False
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    param = sig.parameters.get("strategy")
    return param is not None and param.default is None


def _warn_legacy_strategy_chained_once(check, fn: Callable) -> None:
    """Emit the §9.3 deprecation warning at most once per ``(name, fn)``.

    See ``specs/optimized-strategies.md`` §9.3 for the full migration
    rationale.
    """
    name = str(getattr(check, "name", "?"))
    key = (name, id(fn))
    if key in _LEGACY_CHAINED_WARNED:
        return
    _LEGACY_CHAINED_WARNED.add(key)
    warnings.warn(
        (
            f"The 'strategy' kwarg on Check(check_fn=..., strategy={fn!r}) "
            "is being invoked as a chained strategy because built-in "
            "checks are also present on this column and now produce the "
            "merged base strategy in a single hypothesis call. If "
            f"{getattr(fn, '__name__', fn)!r} relied on running as the "
            "base strategy in this context, migrate it to a constraint "
            "adapter via "
            "`pandera.strategies.pandas_strategies."
            "register_check_constraint(...)` or pass `constraint=...` "
            "to `register_check_method(...)`. This warning will become "
            "an error in pandera 1.0."
        ),
        DeprecationWarning,
        stacklevel=2,
    )


def series_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: SearchStrategy | None = None,
    *,
    checks: Sequence | None = None,
    nullable: bool = False,
    unique: bool = False,
    name: str | None = None,
    size: int | None = None,
) -> SearchStrategy:
    """Strategy to generate a pandas Series.

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param checks: sequence of :class:`~pandera.api.checks.Check` s to constrain
        the values of the data in the column/index.
    :param nullable: whether or not generated Series contains null values.
    :param unique: whether or not generated Series contains unique values.
    :param name: name of the Series.
    :param size: number of elements in the Series.
    :returns: ``hypothesis`` strategy.
    """
    elements = field_element_strategy(pandera_dtype, strategy, checks=checks)

    dtype = (
        None
        # let hypothesis use the elements strategy to build datatime-aware
        # series
        if _is_datetime_tz(pandera_dtype)
        else to_numpy_dtype(pandera_dtype)
    )

    strategy = (
        pdst.series(
            elements=elements,
            dtype=dtype,
            index=pdst.range_indexes(
                min_size=0 if size is None else size, max_size=size
            ),
            unique=bool(unique),
        )
        .filter(lambda x: x.shape[0] > 0)
        .map(lambda x: x.rename(name))
        .map(partial(convert_dtype, col_dtype=pandera_dtype.type))
    )
    if nullable:
        strategy = null_field_masks(strategy)

    def undefined_check_strategy(strategy, check):
        """Strategy for checks with undefined strategies."""
        warnings.warn(
            "Vectorized check doesn't have a defined strategy. "
            "Falling back to filtering drawn values based on the check "
            "definition. This can considerably slow down data-generation."
        )

        def _check_fn(series):
            return check(series).check_passed

        return strategy.filter(_check_fn)

    for check in checks if checks is not None else []:
        # for checks with undefined built-in or custom strategies that are
        # vectorized, apply check function to the entire series.
        if (
            check.strategy is None
            and not STRATEGY_DISPATCHER.get((check.name, pd.Series))
            and not check.element_wise
        ):
            strategy = undefined_check_strategy(strategy, check)

    return strategy


def column_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: SearchStrategy | None = None,
    *,
    checks: Sequence | None = None,
    unique: bool = False,
    name: str | None = None,
):
    """Create a data object describing a column in a DataFrame.

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param checks: sequence of :class:`~pandera.api.checks.Check` s to constrain
        the values of the data in the column/index.
    :param unique: whether or not generated Series contains unique values.
    :param name: name of the Series.
    :returns: a `column <https://hypothesis.readthedocs.io/en/latest/numpy.html#hypothesis.extra.pandas.column>`_ object.
    """
    verify_dtype(pandera_dtype, schema_type="column", name=name)
    elements = field_element_strategy(pandera_dtype, strategy, checks=checks)
    return pdst.column(
        name=name,
        elements=elements,
        dtype=to_numpy_dtype(pandera_dtype),
        unique=unique,
    )


def index_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: SearchStrategy | None = None,
    *,
    checks: Sequence | None = None,
    nullable: bool = False,
    unique: bool = False,
    name: str | None = None,
    size: int | None = None,
):
    """Strategy to generate a pandas Index.

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param checks: sequence of :class:`~pandera.api.checks.Check` s to constrain
        the values of the data in the column/index.
    :param nullable: whether or not generated Series contains null values.
    :param unique: whether or not generated Series contains unique values.
    :param name: name of the Series.
    :param size: number of elements in the Series.
    :returns: ``hypothesis`` strategy.
    """
    verify_dtype(pandera_dtype, schema_type="index", name=name)
    elements = field_element_strategy(pandera_dtype, strategy, checks=checks)

    strategy = pdst.indexes(
        elements=elements,
        dtype=to_numpy_dtype(pandera_dtype),
        min_size=0 if size is None else size,
        max_size=size,
        unique=bool(unique),
    ).map(partial(convert_dtype, col_dtype=pandera_dtype.type))

    # this is a hack to convert np.str_ data values into native python str.
    col_dtype = str(pandera_dtype)
    if col_dtype in {"object", "str"} or col_dtype.startswith("string"):
        strategy = strategy.map(lambda index: index.map(_str_no_surrogates))

    if name is not None:
        strategy = strategy.map(lambda index: index.rename(name))
    if nullable:
        strategy = null_field_masks(strategy)
    return strategy


def dataframe_strategy(
    pandera_dtype: DataType | None = None,
    strategy: SearchStrategy | None = None,
    *,
    columns: dict | None = None,
    checks: Sequence | None = None,
    unique: list[str] | None = None,
    index: IndexComponent | None = None,
    size: int | None = None,
    n_regex_columns: int = 1,
):
    """Strategy to generate a pandas DataFrame.

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: if specified, this will raise a BaseStrategyOnlyError,
        since it cannot be chained to a prior strategy.
    :param columns: a dictionary where keys are column names and values
        are :class:`~pandera.api.pandas.components.Column` objects.
    :param checks: sequence of :class:`~pandera.api.checks.Check` s to constrain
        the values of the data at the dataframe level.
    :param unique: a list of column names that should be jointly unique.
    :param index: Index or MultiIndex schema component.
    :param size: number of elements in the Series.
    :param n_regex_columns: number of regex columns to generate.
    :returns: ``hypothesis`` strategy.
    """

    if n_regex_columns < 1:
        raise ValueError(
            "`n_regex_columns` must be a positive integer, found: "
            f"{n_regex_columns}"
        )
    if strategy is not None:
        raise BaseStrategyOnlyError(
            "The dataframe strategy is a base strategy. You cannot specify "
            "the strategy argument to chain it to a parent strategy."
        )

    columns = {} if columns is None else columns
    checks = [] if checks is None else checks

    def undefined_check_strategy(strategy, check, column=None):
        """Strategy for checks with undefined strategies."""

        def _element_wise_check_fn(element):
            return check._check_fn(element)

        def _column_check_fn(dataframe):
            return check(dataframe[column]).check_passed

        def _dataframe_check_fn(dataframe):
            return check(dataframe).check_passed

        if check.element_wise:
            check_fn = _element_wise_check_fn
            warning_type = "Element-wise"
        elif column is None:
            check_fn = _dataframe_check_fn
            warning_type = "Dataframe"
        else:
            check_fn = _column_check_fn
            warning_type = "Column"

        warnings.warn(
            f"{warning_type} check doesn't have a defined strategy. "
            "Falling back to filtering drawn values based on the check "
            "definition. This can considerably slow down data-generation."
        )

        return strategy.filter(check_fn)

    def make_row_strategy(col, checks):
        """Apply ``checks`` to ``col``'s per-element strategy.

        Mirrors :func:`field_element_strategy`'s bucketing logic so
        dataframe-level checks compose without ``.filter`` chaining
        wherever a constraint adapter is available:

        1. constraint-providing checks (built-in via
           ``(check.name, pd.DataFrame)`` in ``CONSTRAINT_DISPATCHER``,
           or ``check.constraint`` set explicitly): aggregated into a
           single ``FieldConstraints`` and lowered to one strategy.
        2. legacy ``check.strategy`` callables (or
           ``STRATEGY_DISPATCHER[(name, pd.DataFrame)]``): chained as
           before, on top of the merged base.
        3. element-wise checks with no strategy / constraint: degraded
           to a single residual ``.filter`` with the pre-existing
           ``undefined_check_strategy`` warning.

        For the v1 refactor every built-in is column-scoped, so no
        check populates the ``(name, pd.DataFrame)`` constraint slot
        and the practical generation path is unchanged. The bucketing
        is in place as the extension point for future dataframe-level
        constraint adapters.
        """
        constraint_acc = FieldConstraints()
        legacy_strategies: list[tuple[Any, Callable]] = []
        undefined_checks: list = []

        for check in checks:
            constraint_fn = getattr(check, "constraint", None) or (
                CONSTRAINT_DISPATCHER.get((check.name, pd.DataFrame))
            )
            if constraint_fn is not None:
                try:
                    constraint_acc = constraint_acc.merge(
                        constraint_fn(**check.statistics)
                    )
                except ConstraintConflictError as exc:
                    _raise_unsatisfiable(checks, exc)
                continue

            legacy_strategy = check.strategy or STRATEGY_DISPATCHER.get(
                (check.name, pd.DataFrame)
            )
            if legacy_strategy is not None:
                legacy_strategies.append((check, legacy_strategy))
                continue

            undefined_checks.append(check)

        has_aggregated = not constraint_acc.is_empty()

        if has_aggregated:
            try:
                strategy = compile_field_strategy(col.dtype, constraint_acc)
            except ConstraintConflictError as exc:
                _raise_unsatisfiable(checks, exc)
        else:
            strategy = None

        for check, legacy_strategy in legacy_strategies:
            if has_aggregated and _strategy_supports_base_mode(
                legacy_strategy
            ):
                _warn_legacy_strategy_chained_once(check, legacy_strategy)
            strategy = legacy_strategy(col.dtype, strategy, **check.statistics)

        for check in undefined_checks:
            strategy = undefined_check_strategy(
                strategy=(
                    pandas_dtype_strategy(col.dtype)
                    if strategy is None
                    else strategy
                ),
                check=check,
            )

        if strategy is None:
            strategy = pandas_dtype_strategy(col.dtype)
        return strategy

    @composite
    def _dataframe_strategy(draw):
        row_strategy_checks = []
        undefined_strat_df_checks = []
        for check in checks:
            if (
                check.strategy
                or STRATEGY_DISPATCHER.get((check.name, pd.DataFrame), None)
                or getattr(check, "constraint", None) is not None
                or CONSTRAINT_DISPATCHER.get((check.name, pd.DataFrame))
                or check.element_wise
            ):
                # we can apply element-wise checks defined at the dataframe
                # level to the row strategy
                row_strategy_checks.append(check)
            else:
                undefined_strat_df_checks.append(check)

        # expand column set to generate column names for columns where
        # regex=True.
        expanded_columns = {}
        for col_name, column in columns.items():
            if unique and col_name in unique:
                # if the column is in the set of columns specified in `unique`,
                # make the column strategy independently unique. This is
                # technically stricter than it should be, since the list of
                # columns in `unique` are required to be jointly unique, but
                # this is a simple solution that produces synthetic data that
                # fulfills the uniqueness constraints of the dataframe.
                column = deepcopy(column)
                column.unique = True
            if not column.regex:
                expanded_columns[col_name] = column
            else:
                regex_columns = draw(
                    st.lists(
                        st.from_regex(column.name, fullmatch=True),
                        min_size=n_regex_columns,
                        max_size=n_regex_columns,
                        unique=True,
                    )
                )
                for regex_col in regex_columns:
                    expanded_columns[regex_col] = deepcopy(column).set_name(
                        regex_col
                    )

        # collect all non-element-wise column checks with undefined strategies
        # Constraint adapters fully describe the element distribution, so
        # they don't need a redundant ``.filter`` pass on the dataframe.
        undefined_strat_column_checks: dict[str, list] = defaultdict(list)
        for col_name, column in expanded_columns.items():
            undefined_strat_column_checks[col_name].extend(
                check
                for check in column.checks
                if STRATEGY_DISPATCHER.get((check.name, pd.DataFrame)) is None
                and getattr(check, "constraint", None) is None
                and CONSTRAINT_DISPATCHER.get((check.name, pd.Series)) is None
                and not check.element_wise
            )

        # override the column datatype with dataframe-level datatype if
        # specified
        col_dtypes = {
            col_name: (
                str(col.dtype) if pandera_dtype is None else str(pandera_dtype)
            )
            for col_name, col in expanded_columns.items()
        }
        nullable_columns = {
            col_name: col.nullable
            for col_name, col in expanded_columns.items()
        }

        row_strategy = None
        if row_strategy_checks:
            row_strategy = st.fixed_dictionaries(
                {
                    col_name: make_row_strategy(col, row_strategy_checks)
                    for col_name, col in expanded_columns.items()
                }
            )

        strategy = pdst.data_frames(
            columns=[
                column.strategy_component()
                for column in expanded_columns.values()
            ],
            rows=row_strategy,
            index=pdst.range_indexes(
                min_size=0 if size is None else size, max_size=size
            ),
        )

        # this is a hack to convert np.str_ data values into native python str.
        string_columns = []
        for col_name, col_dtype in col_dtypes.items():
            if col_dtype in {"object", "str"} or col_dtype.startswith(
                "string"
            ):
                string_columns.append(col_name)

        if string_columns:
            strategy = strategy.map(
                lambda df: df.assign(
                    **{
                        col_name: df[col_name].map(_str_no_surrogates)
                        for col_name in string_columns
                    }
                )
            )

        strategy = strategy.map(partial(convert_dtypes, col_dtypes=col_dtypes))

        if size is not None and size > 0 and any(nullable_columns.values()):
            strategy = null_dataframe_masks(strategy, nullable_columns)

        if index is not None:
            strategy = set_pandas_index(strategy, index)

        for check in undefined_strat_df_checks:
            strategy = undefined_check_strategy(strategy, check)

        for col_name, column_checks in undefined_strat_column_checks.items():
            for check in column_checks:  # type: ignore
                strategy = undefined_check_strategy(
                    strategy, check, column=col_name
                )

        return draw(strategy)

    return _dataframe_strategy()


def multiindex_strategy(
    pandera_dtype: DataType | None = None,
    strategy: SearchStrategy | None = None,
    *,
    indexes: list | None = None,
    size: int | None = None,
):
    """Strategy to generate a pandas MultiIndex object.

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param indexes: a list of :class:`~pandera.api.pandas.components.Index`
        objects.
    :param size: number of elements in the Series.
    :returns: ``hypothesis`` strategy.
    """

    if strategy is not None:
        raise BaseStrategyOnlyError(
            "The dataframe strategy is a base strategy. You cannot specify "
            "the strategy argument to chain it to a parent strategy."
        )
    indexes = [] if indexes is None else indexes
    index_dtypes = {
        index.name if index.name is not None else i: str(index.dtype)
        for i, index in enumerate(indexes)
    }
    nullable_index = {
        index.name if index.name is not None else i: index.nullable
        for i, index in enumerate(indexes)
    }
    strategy = pdst.data_frames(
        [index.strategy_component() for index in indexes],
        index=pdst.range_indexes(
            min_size=0 if size is None else size, max_size=size
        ),
    ).map(partial(convert_dtypes, col_dtypes=index_dtypes))

    # this is a hack to convert np.str_ data values into native python str.
    for name, dtype in index_dtypes.items():
        if dtype in {"object", "str"} or dtype.startswith("string"):
            strategy = strategy.map(
                lambda df, name=name: df.assign(
                    **{name: df[name].map(_str_no_surrogates)}
                )
            )

    if any(nullable_index.values()):
        strategy = null_dataframe_masks(strategy, nullable_index)
    return strategy.map(pd.MultiIndex.from_frame)
