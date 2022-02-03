# pylint: disable=no-value-for-parameter,too-many-lines
"""Generate synthetic data from a schema definition.

*new in 0.6.0*

This module is responsible for generating data based on the type and check
constraints specified in a ``pandera`` schema. It's built on top of the
`hypothesis <https://hypothesis.readthedocs.io/en/latest/index.html>`_ package
to compose strategies given multiple checks specified in a schema.

See the :ref:`user guide<data synthesis strategies>` for more details.
"""
import operator
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandas as pd

from .dtypes import (
    DataType,
    is_category,
    is_complex,
    is_datetime,
    is_float,
    is_timedelta,
)
from .engines import numpy_engine, pandas_engine
from .errors import BaseStrategyOnlyError, SchemaDefinitionError

try:
    import hypothesis
    import hypothesis.extra.numpy as npst
    import hypothesis.extra.pandas as pdst
    import hypothesis.strategies as st
    from hypothesis.strategies import SearchStrategy, composite
except ImportError:  # pragma: no cover

    # pylint: disable=too-few-public-methods
    class SearchStrategy:  # type: ignore
        """placeholder type."""

    def composite(fn):
        """placeholder composite strategy."""
        return fn

    HAS_HYPOTHESIS = False
else:
    HAS_HYPOTHESIS = True


StrategyFn = Callable[..., SearchStrategy]
# Fix this when modules have been re-organized to avoid circular imports
IndexComponent = Any
F = TypeVar("F", bound=Callable)


def _mask(
    val: Union[pd.Series, pd.Index], null_mask: List[bool]
) -> Union[pd.Series, pd.Index]:
    if pd.api.types.is_timedelta64_dtype(val):
        return val.mask(null_mask, pd.NaT)
    elif val.dtype == pd.StringDtype():
        return val.mask(null_mask, pd.NA)
    return val.mask(null_mask)


@composite
def null_field_masks(draw, strategy: Optional[SearchStrategy]):
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
    strategy: Optional[SearchStrategy],
    nullable_columns: Dict[str, bool],
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
    name: Optional[str],
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
                "installed. You can install pandera together with the IO \n"
                "dependencies with:\n"
                "pip install pandera[strategies]"
            )
        return fn(*args, **kwargs)

    return cast(F, _wrapper)


def register_check_strategy(strategy_fn: StrategyFn):
    """Decorate a Check method with a strategy.

    This should be applied to a built-in :class:`~pandera.checks.Check` method.

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
            strategy_kwargs = {
                arg: stat
                for arg, stat in check.statistics.items()
                if stat is not None
            }
            check.strategy = partial(strategy_fn, **strategy_kwargs)
            return check

        return _wrapper

    return register_check_strategy_decorator


# pylint: disable=line-too-long
# Values taken from
# https://hypothesis.readthedocs.io/en/latest/_modules/hypothesis/extra/numpy.html#from_dtype  # noqa
MIN_DT_VALUE = -(2**63)
MAX_DT_VALUE = 2**63 - 1


def _is_datetime_tz(pandera_dtype: DataType) -> bool:
    native_type = getattr(pandera_dtype, "type", None)
    return isinstance(native_type, pd.DatetimeTZDtype)


def _datetime_strategy(
    dtype: Union[np.dtype, pd.DatetimeTZDtype], strategy
) -> SearchStrategy:

    if isinstance(dtype, pd.DatetimeTZDtype):

        def _to_datetime(value) -> pd.DatetimeTZDtype:
            if isinstance(value, pd.Timestamp):
                return value.tz_convert(tz=dtype.tz)  # type: ignore[union-attr]
            return pd.Timestamp(value, unit=dtype.unit, tz=dtype.tz)  # type: ignore[union-attr]

        return st.builds(_to_datetime, strategy)
    else:
        res = (
            st.just(dtype.str.split("[")[-1][:-1])
            if "[" in dtype.str
            else st.sampled_from(npst.TIME_RESOLUTIONS)
        )
        return st.builds(dtype.type, strategy, res)


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
            return pd.Timedelta(value).value
        return pd.Timestamp(value).value

    min_value = MIN_DT_VALUE if min_value is None else _to_unix(min_value)
    max_value = MAX_DT_VALUE if max_value is None else _to_unix(max_value)
    return _datetime_strategy(dtype, st.integers(min_value, max_value))


def numpy_complex_dtypes(
    dtype,
    min_value: complex = complex(0, 0),
    max_value: Optional[complex] = None,
    allow_infinity: bool = None,
    allow_nan: bool = None,
):
    """Create numpy strategy for complex numbers.

    :param dtype: numpy complex number datatype
    :param min_value: minimum value, must be complex number
    :param max_value: maximum value, must be complex number
    :returns: ``hypothesis`` strategy
    """
    max_real: Optional[float]
    max_imag: Optional[float]
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


def pandas_dtype_strategy(
    pandera_dtype: DataType,
    strategy: Optional[SearchStrategy] = None,
    **kwargs,
) -> SearchStrategy:
    # pylint: disable=line-too-long,no-else-raise
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
    if strategy:
        if _is_datetime_tz(pandera_dtype):
            return _datetime_strategy(pandera_dtype.type, strategy)  # type: ignore
        return strategy.map(np_dtype.type)
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
    return npst.from_dtype(
        np_dtype,
        **{  # type: ignore
            "allow_nan": False,
            "allow_infinity": False,
            **kwargs,
        },
    )


def eq_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: Optional[SearchStrategy] = None,
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
    # pylint: disable=unused-argument
    return pandas_dtype_strategy(pandera_dtype, st.just(value))


def ne_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: Optional[SearchStrategy] = None,
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
    return strategy.filter(lambda x: x != value)


def gt_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: Optional[SearchStrategy] = None,
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
    return strategy.filter(lambda x: x > min_value)


def ge_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: Optional[SearchStrategy] = None,
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
    return strategy.filter(lambda x: x >= min_value)


def lt_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: Optional[SearchStrategy] = None,
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
    return strategy.filter(lambda x: x < max_value)


def le_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: Optional[SearchStrategy] = None,
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
    return strategy.filter(lambda x: x <= max_value)


def in_range_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: Optional[SearchStrategy] = None,
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
    min_op = operator.ge if include_min else operator.gt
    max_op = operator.le if include_max else operator.lt
    return strategy.filter(
        lambda x: min_op(x, min_value) and max_op(x, max_value)
    )


def isin_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: Optional[SearchStrategy] = None,
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
    strategy: Optional[SearchStrategy] = None,
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
    strategy: Optional[SearchStrategy] = None,
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

    def matches(x):
        return re.match(pattern, x)

    return strategy.filter(matches)


def str_contains_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: Optional[SearchStrategy] = None,
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

    def contains(x):
        return re.search(pattern, x)

    return strategy.filter(contains)


def str_startswith_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: Optional[SearchStrategy] = None,
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
    if strategy is None:
        return st.from_regex(f"\\A{string}", fullmatch=False).map(
            to_numpy_dtype(pandera_dtype).type
        )

    return strategy.filter(lambda x: x.startswith(string))


def str_endswith_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: Optional[SearchStrategy] = None,
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
    if strategy is None:
        return st.from_regex(f"{string}\\Z", fullmatch=False).map(
            to_numpy_dtype(pandera_dtype).type
        )

    return strategy.filter(lambda x: x.endswith(string))


def str_length_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: Optional[SearchStrategy] = None,
    *,
    min_value: int,
    max_value: int,
) -> SearchStrategy:
    """Strategy to generate strings of a particular length

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param min_value: minimum string length.
    :param max_value: maximum string length.
    :returns: ``hypothesis`` strategy
    """
    if strategy is None:
        return st.text(min_size=min_value, max_size=max_value).map(
            to_numpy_dtype(pandera_dtype).type
        )

    return strategy.filter(lambda x: min_value <= len(x) <= max_value)


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
    strategy: Optional[SearchStrategy] = None,
    *,
    checks: Optional[Sequence] = None,
) -> SearchStrategy:
    """Strategy to generate elements of a column or index.

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param checks: sequence of :class:`~pandera.checks.Check` s to constrain
        the values of the data in the column/index.
    :returns: ``hypothesis`` strategy
    """
    if strategy:
        raise BaseStrategyOnlyError(
            "The series strategy is a base strategy. You cannot specify the "
            "strategy argument to chain it to a parent strategy."
        )
    checks = [] if checks is None else checks
    elements = None

    def undefined_check_strategy(elements, check):
        """Strategy for checks with undefined strategies."""
        warnings.warn(
            "Element-wise check doesn't have a defined strategy."
            "Falling back to filtering drawn values based on the check "
            "definition. This can considerably slow down data-generation."
        )
        return (
            pandas_dtype_strategy(pandera_dtype)
            if elements is None
            else elements
        ).filter(check._check_fn)

    for check in checks:
        if hasattr(check, "strategy"):
            elements = check.strategy(pandera_dtype, elements)
        elif check.element_wise:
            elements = undefined_check_strategy(elements, check)
        # NOTE: vectorized checks with undefined strategies should be handled
        # by the series/dataframe strategy.
    if elements is None:
        elements = pandas_dtype_strategy(pandera_dtype)

    # Hypothesis only supports pure numpy datetime64 (i.e. timezone naive).
    # We cast to datetime64 after applying the check strategy so that checks
    # can see timezone-aware values.
    if _is_datetime_tz(pandera_dtype):
        elements = _timestamp_to_datetime64_strategy(elements)

    return elements


def series_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: Optional[SearchStrategy] = None,
    *,
    checks: Optional[Sequence] = None,
    nullable: bool = False,
    unique: bool = False,
    name: Optional[str] = None,
    size: Optional[int] = None,
):
    """Strategy to generate a pandas Series.

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param checks: sequence of :class:`~pandera.checks.Check` s to constrain
        the values of the data in the column/index.
    :param nullable: whether or not generated Series contains null values.
    :param unique: whether or not generated Series contains unique values.
    :param name: name of the Series.
    :param size: number of elements in the Series.
    :returns: ``hypothesis`` strategy.
    """
    elements = field_element_strategy(pandera_dtype, strategy, checks=checks)
    strategy = (
        pdst.series(
            elements=elements,
            dtype=to_numpy_dtype(pandera_dtype),
            index=pdst.range_indexes(
                min_size=0 if size is None else size, max_size=size
            ),
            unique=unique,
        )
        .filter(lambda x: x.shape[0] > 0)
        .map(lambda x: x.rename(name))
        .map(lambda x: x.astype(pandera_dtype.type))
    )
    if nullable:
        strategy = null_field_masks(strategy)

    def undefined_check_strategy(strategy, check):
        """Strategy for checks with undefined strategies."""
        warnings.warn(
            "Vectorized check doesn't have a defined strategy."
            "Falling back to filtering drawn values based on the check "
            "definition. This can considerably slow down data-generation."
        )

        def _check_fn(series):
            return check(series).check_passed

        return strategy.filter(_check_fn)

    for check in checks if checks is not None else []:
        if not hasattr(check, "strategy") and not check.element_wise:
            strategy = undefined_check_strategy(strategy, check)

    return strategy


def column_strategy(
    pandera_dtype: Union[numpy_engine.DataType, pandas_engine.DataType],
    strategy: Optional[SearchStrategy] = None,
    *,
    checks: Optional[Sequence] = None,
    unique: bool = False,
    name: Optional[str] = None,
):
    # pylint: disable=line-too-long
    """Create a data object describing a column in a DataFrame.

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param checks: sequence of :class:`~pandera.checks.Check` s to constrain
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
    strategy: Optional[SearchStrategy] = None,
    *,
    checks: Optional[Sequence] = None,
    nullable: bool = False,
    unique: bool = False,
    name: Optional[str] = None,
    size: Optional[int] = None,
):
    """Strategy to generate a pandas Index.

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param checks: sequence of :class:`~pandera.checks.Check` s to constrain
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
        unique=unique,
    ).map(lambda x: x.astype(pandera_dtype.type))

    # this is a hack to convert np.str_ data values into native python str.
    col_dtype = str(pandera_dtype)
    if col_dtype in {"object", "str"} or col_dtype.startswith("string"):
        # pylint: disable=cell-var-from-loop,undefined-loop-variable
        strategy = strategy.map(lambda index: index.map(str))

    if name is not None:
        strategy = strategy.map(lambda index: index.rename(name))
    if nullable:
        strategy = null_field_masks(strategy)
    return strategy


def dataframe_strategy(
    pandera_dtype: Optional[DataType] = None,
    strategy: Optional[SearchStrategy] = None,
    *,
    columns: Optional[Dict] = None,
    checks: Optional[Sequence] = None,
    unique: Optional[List[str]] = None,
    index: Optional[IndexComponent] = None,
    size: Optional[int] = None,
    n_regex_columns: int = 1,
):
    """Strategy to generate a pandas DataFrame.

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: if specified, this will raise a BaseStrategyOnlyError,
        since it cannot be chained to a prior strategy.
    :param columns: a dictionary where keys are column names and values
        are :class:`~pandera.schema_components.Column` objects.
    :param checks: sequence of :class:`~pandera.checks.Check` s to constrain
        the values of the data at the dataframe level.
    :param unique: a list of column names that should be jointly unique.
    :param index: Index or MultiIndex schema component.
    :param size: number of elements in the Series.
    :param n_regex_columns: number of regex columns to generate.
    :returns: ``hypothesis`` strategy.
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    if n_regex_columns < 1:
        raise ValueError(
            "`n_regex_columns` must be a positive integer, found: "
            f"{n_regex_columns}"
        )
    if strategy:
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
        strategy = None
        for check in checks:
            if hasattr(check, "strategy"):
                strategy = check.strategy(col.dtype, strategy)
            else:
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
            if hasattr(check, "strategy") or check.element_wise:
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
        undefined_strat_column_checks: Dict[str, list] = defaultdict(list)
        for col_name, column in expanded_columns.items():
            undefined_strat_column_checks[col_name].extend(
                check
                for check in column.checks
                if not hasattr(check, "strategy") and not check.element_wise
            )

        # override the column datatype with dataframe-level datatype if
        # specified
        col_dtypes = {
            col_name: str(col.dtype)
            if pandera_dtype is None
            else str(pandera_dtype)
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
        for col_name, col_dtype in col_dtypes.items():
            if col_dtype in {"object", "str"} or col_dtype.startswith(
                "string"
            ):
                # pylint: disable=cell-var-from-loop,undefined-loop-variable
                strategy = strategy.map(
                    lambda df: df.assign(**{col_name: df[col_name].map(str)})
                )

        strategy = strategy.map(
            lambda df: df if df.empty else df.astype(col_dtypes)
        )

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


# pylint: disable=unused-argument
def multiindex_strategy(
    pandera_dtype: Optional[DataType] = None,
    strategy: Optional[SearchStrategy] = None,
    *,
    indexes: Optional[List] = None,
    size: Optional[int] = None,
):
    """Strategy to generate a pandas MultiIndex object.

    :param pandera_dtype: :class:`pandera.dtypes.DataType` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param indexes: a list of :class:`~pandera.schema_components.Index`
        objects.
    :param size: number of elements in the Series.
    :returns: ``hypothesis`` strategy.
    """
    # pylint: disable=unnecessary-lambda
    if strategy:
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
    ).map(lambda x: x.astype(index_dtypes))

    # this is a hack to convert np.str_ data values into native python str.
    for name, dtype in index_dtypes.items():
        if dtype in {"object", "str"} or dtype.startswith("string"):
            # pylint: disable=cell-var-from-loop,undefined-loop-variable
            strategy = strategy.map(
                lambda df: df.assign(**{name: df[name].map(str)})
            )

    if any(nullable_index.values()):
        strategy = null_dataframe_masks(strategy, nullable_index)
    return strategy.map(pd.MultiIndex.from_frame)
