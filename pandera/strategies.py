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
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .dtypes import PandasDtype
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


@composite
def null_field_masks(draw, strategy: Optional[SearchStrategy]):
    """Strategy for masking a column/index with null values.

    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    """
    val = draw(strategy)
    size = val.shape[0]
    null_mask = draw(st.lists(st.booleans(), min_size=size, max_size=size))
    # assume that there is at least one masked value
    hypothesis.assume(any(null_mask))
    if isinstance(val, pd.Index):
        val = val.to_series()
        val = val.mask(null_mask)
        return pd.Index(val)
    return val.mask(null_mask)


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
    # assume that there is at least one masked value
    hypothesis.assume(null_mask.any(axis=None))
    return val.mask(null_mask)


@composite
def set_pandas_index(
    draw,
    df_or_series_strat: SearchStrategy,
    index_strat: SearchStrategy,
):
    """Sets Index or MultiIndex object to pandas Series or DataFrame."""
    df_or_series = draw(df_or_series_strat)
    index = draw(index_strat)
    df_or_series.index = index
    return df_or_series


def verify_pandas_dtype(pandas_dtype, schema_type: str, name: Optional[str]):
    """Verify that pandas_dtype argument is not None."""
    if pandas_dtype is None:
        raise SchemaDefinitionError(
            f"'{schema_type}' schema with name '{name}' has no specified "
            "pandas_dtype. You need to specify one in order to synthesize "
            "data from a strategy."
        )


def strategy_import_error(fn):
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

    return _wrapper


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
MIN_DT_VALUE = -(2 ** 63)
MAX_DT_VALUE = 2 ** 63 - 1


def numpy_time_dtypes(dtype, min_value=None, max_value=None):
    """Create numpy strategy for datetime and timedelta data types.

    :param dtype: numpy datetime or timedelta datatype
    :param min_value: minimum value of the datatype to create
    :param max_value: maximum value of the datatype to create
    :returns: ``hypothesis`` strategy
    """
    res = (
        st.just(dtype.str.split("[")[-1][:-1])
        if "[" in dtype.str
        else st.sampled_from(npst.TIME_RESOLUTIONS)
    )
    return st.builds(
        dtype.type,
        st.integers(
            MIN_DT_VALUE if min_value is None else min_value.astype(np.int64),
            MAX_DT_VALUE if max_value is None else max_value.astype(np.int64),
        ),
        res,
    )


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


def pandas_dtype_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    **kwargs,
) -> SearchStrategy:
    # pylint: disable=line-too-long,no-else-raise
    """Strategy to generate data from a :class:`pandera.dtypes.PandasDtype`.

    :param pandas_dtype: :class:`pandera.dtypes.PandasDtype` instance.
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
    if pandas_dtype is PandasDtype.Category:
        raise TypeError(
            "data generation for the Category dtype is currently "
            "unsupported. Consider using a string or int dtype and "
            "Check.isin(values) to ensure a finite set of values."
        )

    # The object type falls back onto generating strings.
    if pandas_dtype is PandasDtype.Object:
        dtype = np.dtype("str")
    else:
        dtype = pandas_dtype.numpy_dtype

    if strategy:
        return strategy.map(dtype.type)
    elif pandas_dtype.is_datetime or pandas_dtype.is_timedelta:
        return numpy_time_dtypes(
            dtype,
            **compat_kwargs("min_value", "max_value"),
        )
    elif pandas_dtype.is_complex:
        return numpy_complex_dtypes(
            dtype,
            **compat_kwargs(
                "min_value", "max_value", "allow_infinity", "allow_nan"
            ),
        )
    return npst.from_dtype(
        dtype,
        **{  # type: ignore
            "allow_nan": False,
            "allow_infinity": False,
            **kwargs,
        },
    )


def eq_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    value: Any,
) -> SearchStrategy:
    """Strategy to generate a single value.

    :param pandas_dtype: :class:`pandera.dtypes.PandasDtype` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param value: value to generate.
    :returns: ``hypothesis`` strategy
    """
    # override strategy preceding this one and generate value of the same type
    if strategy is None:
        strategy = pandas_dtype_strategy(pandas_dtype)
    return st.just(value).map(pandas_dtype.numpy_dtype.type)


def ne_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    value: Any,
) -> SearchStrategy:
    """Strategy to generate anything except for a particular value.

    :param pandas_dtype: :class:`pandera.dtypes.PandasDtype` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param value: value to avoid.
    :returns: ``hypothesis`` strategy
    """
    if strategy is None:
        strategy = pandas_dtype_strategy(pandas_dtype)
    return strategy.filter(lambda x: x != value)


def gt_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    min_value: Union[int, float],
) -> SearchStrategy:
    """Strategy to generate values greater than a minimum value.

    :param pandas_dtype: :class:`pandera.dtypes.PandasDtype` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param min_value: generate values larger than this.
    :returns: ``hypothesis`` strategy
    """
    if strategy is None:
        strategy = pandas_dtype_strategy(
            pandas_dtype,
            min_value=min_value,
            exclude_min=True if pandas_dtype.is_float else None,
        )
    return strategy.filter(lambda x: x > min_value)


def ge_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    min_value: Union[int, float],
) -> SearchStrategy:
    """Strategy to generate values greater than or equal to a minimum value.

    :param pandas_dtype: :class:`pandera.dtypes.PandasDtype` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param min_value: generate values greater than or equal to this.
    :returns: ``hypothesis`` strategy
    """
    if strategy is None:
        return pandas_dtype_strategy(
            pandas_dtype,
            min_value=min_value,
            exclude_min=False if pandas_dtype.is_float else None,
        )
    return strategy.filter(lambda x: x >= min_value)


def lt_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    max_value: Union[int, float],
) -> SearchStrategy:
    """Strategy to generate values less than a maximum value.

    :param pandas_dtype: :class:`pandera.dtypes.PandasDtype` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param max_value: generate values less than this.
    :returns: ``hypothesis`` strategy
    """
    if strategy is None:
        strategy = pandas_dtype_strategy(
            pandas_dtype,
            max_value=max_value,
            exclude_max=True if pandas_dtype.is_float else None,
        )
    return strategy.filter(lambda x: x < max_value)


def le_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    max_value: Union[int, float],
) -> SearchStrategy:
    """Strategy to generate values less than or equal to a maximum value.

    :param pandas_dtype: :class:`pandera.dtypes.PandasDtype` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param max_value: generate values less than or equal to this.
    :returns: ``hypothesis`` strategy
    """
    if strategy is None:
        return pandas_dtype_strategy(
            pandas_dtype,
            max_value=max_value,
            exclude_max=False if pandas_dtype.is_float else None,
        )
    return strategy.filter(lambda x: x <= max_value)


def in_range_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    min_value: Union[int, float],
    max_value: Union[int, float],
    include_min: bool = True,
    include_max: bool = True,
) -> SearchStrategy:
    """Strategy to generate values within a particular range.

    :param pandas_dtype: :class:`pandera.dtypes.PandasDtype` instance.
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
            pandas_dtype,
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
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    allowed_values: Sequence[Any],
) -> SearchStrategy:
    """Strategy to generate values within a finite set.

    :param pandas_dtype: :class:`pandera.dtypes.PandasDtype` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param allowed_values: set of allowable values.
    :returns: ``hypothesis`` strategy
    """
    if strategy is None:
        return st.sampled_from(allowed_values).map(
            pandas_dtype.numpy_dtype.type
        )
    return strategy.filter(lambda x: x in allowed_values)


def notin_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    forbidden_values: Sequence[Any],
) -> SearchStrategy:
    """Strategy to generate values excluding a set of forbidden values

    :param pandas_dtype: :class:`pandera.dtypes.PandasDtype` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param forbidden_values: set of forbidden values.
    :returns: ``hypothesis`` strategy
    """
    if strategy is None:
        strategy = pandas_dtype_strategy(pandas_dtype)
    return strategy.filter(lambda x: x not in forbidden_values)


def str_matches_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    pattern: str,
) -> SearchStrategy:
    """Strategy to generate strings that patch a regex pattern.

    :param pandas_dtype: :class:`pandera.dtypes.PandasDtype` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param pattern: regex pattern.
    :returns: ``hypothesis`` strategy
    """
    if strategy is None:
        return st.from_regex(pattern, fullmatch=True).map(
            pandas_dtype.numpy_dtype.type
        )

    def matches(x):
        return re.match(pattern, x)

    return strategy.filter(matches)


def str_contains_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    pattern: str,
) -> SearchStrategy:
    """Strategy to generate strings that contain a particular pattern.

    :param pandas_dtype: :class:`pandera.dtypes.PandasDtype` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param pattern: regex pattern.
    :returns: ``hypothesis`` strategy
    """
    if strategy is None:
        return st.from_regex(pattern, fullmatch=False).map(
            pandas_dtype.numpy_dtype.type
        )

    def contains(x):
        return re.search(pattern, x)

    return strategy.filter(contains)


def str_startswith_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    string: str,
) -> SearchStrategy:
    """Strategy to generate strings that start with a specific string pattern.

    :param pandas_dtype: :class:`pandera.dtypes.PandasDtype` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param string: string pattern.
    :returns: ``hypothesis`` strategy
    """
    if strategy is None:
        return st.from_regex(f"\\A{string}", fullmatch=False).map(
            pandas_dtype.numpy_dtype.type
        )

    return strategy.filter(lambda x: x.startswith(string))


def str_endswith_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    string: str,
) -> SearchStrategy:
    """Strategy to generate strings that end with a specific string pattern.

    :param pandas_dtype: :class:`pandera.dtypes.PandasDtype` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param string: string pattern.
    :returns: ``hypothesis`` strategy
    """
    if strategy is None:
        return st.from_regex(f"{string}\\Z", fullmatch=False).map(
            pandas_dtype.numpy_dtype.type
        )

    return strategy.filter(lambda x: x.endswith(string))


def str_length_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    min_value: int,
    max_value: int,
) -> SearchStrategy:
    """Strategy to generate strings of a particular length

    :param pandas_dtype: :class:`pandera.dtypes.PandasDtype` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param min_value: minimum string length.
    :param max_value: maximum string length.
    :returns: ``hypothesis`` strategy
    """
    if strategy is None:
        return st.text(min_size=min_value, max_size=max_value).map(
            pandas_dtype.numpy_dtype.type
        )

    return strategy.filter(lambda x: min_value <= len(x) <= max_value)


def field_element_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    checks: Optional[Sequence] = None,
) -> SearchStrategy:
    """Strategy to generate elements of a column or index.

    :param pandas_dtype: :class:`pandera.dtypes.PandasDtype` instance.
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
            pandas_dtype_strategy(pandas_dtype)
            if elements is None
            else elements
        ).filter(check._check_fn)

    for check in checks:
        if hasattr(check, "strategy"):
            elements = check.strategy(pandas_dtype, elements)
        elif check.element_wise:
            elements = undefined_check_strategy(elements, check)
        # NOTE: vectorized checks with undefined strategies should be handled
        # by the series/dataframe strategy.
    if elements is None:
        elements = pandas_dtype_strategy(pandas_dtype)
    return elements


def series_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    checks: Optional[Sequence] = None,
    nullable: Optional[bool] = False,
    allow_duplicates: Optional[bool] = True,
    name: Optional[str] = None,
    size: Optional[int] = None,
):
    """Strategy to generate a pandas Series.

    :param pandas_dtype: :class:`pandera.dtypes.PandasDtype` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param checks: sequence of :class:`~pandera.checks.Check` s to constrain
        the values of the data in the column/index.
    :param nullable: whether or not generated Series contains null values.
    :param allow_duplicates: whether or not generated Series contains
        duplicates.
    :param name: name of the Series.
    :param size: number of elements in the Series.
    :returns: ``hypothesis`` strategy.
    """
    elements = field_element_strategy(pandas_dtype, strategy, checks=checks)
    strategy = (
        pdst.series(
            elements=elements,
            dtype=pandas_dtype.numpy_dtype,
            index=pdst.range_indexes(
                min_size=0 if size is None else size, max_size=size
            ),
            unique=not allow_duplicates,
        )
        .filter(lambda x: x.shape[0] > 0)
        .map(lambda x: x.rename(name))
        .map(lambda x: x.astype(pandas_dtype.str_alias))
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
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    checks: Optional[Sequence] = None,
    allow_duplicates: Optional[bool] = True,
    name: Optional[str] = None,
):
    # pylint: disable=line-too-long
    """Create a data object describing a column in a DataFrame.

    :param pandas_dtype: :class:`pandera.dtypes.PandasDtype` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param checks: sequence of :class:`~pandera.checks.Check` s to constrain
        the values of the data in the column/index.
    :param allow_duplicates: whether or not generated Series contains
        duplicates.
    :param name: name of the Series.
    :returns: a `column <https://hypothesis.readthedocs.io/en/latest/numpy.html#hypothesis.extra.pandas.column>`_ object.
    """
    verify_pandas_dtype(pandas_dtype, schema_type="column", name=name)
    elements = field_element_strategy(pandas_dtype, strategy, checks=checks)
    return pdst.column(
        name=name,
        elements=elements,
        dtype=pandas_dtype.numpy_dtype,
        unique=not allow_duplicates,
    )


def index_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    checks: Optional[Sequence] = None,
    nullable: Optional[bool] = False,
    allow_duplicates: Optional[bool] = True,
    name: Optional[str] = None,
    size: Optional[int] = None,
):
    """Strategy to generate a pandas Index.

    :param pandas_dtype: :class:`pandera.dtypes.PandasDtype` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param checks: sequence of :class:`~pandera.checks.Check` s to constrain
        the values of the data in the column/index.
    :param nullable: whether or not generated Series contains null values.
    :param allow_duplicates: whether or not generated Series contains
        duplicates.
    :param name: name of the Series.
    :param size: number of elements in the Series.
    :returns: ``hypothesis`` strategy.
    """
    verify_pandas_dtype(pandas_dtype, schema_type="index", name=name)
    elements = field_element_strategy(pandas_dtype, strategy, checks=checks)
    strategy = pdst.indexes(
        elements=elements,
        dtype=pandas_dtype.numpy_dtype,
        min_size=0 if size is None else size,
        max_size=size,
        unique=not allow_duplicates,
    ).map(lambda x: x.astype(pandas_dtype.str_alias))
    if name is not None:
        strategy = strategy.map(lambda index: index.rename(name))
    if nullable:
        strategy = null_field_masks(strategy)
    return strategy


def dataframe_strategy(
    pandas_dtype: Optional[PandasDtype] = None,
    strategy: Optional[SearchStrategy] = None,
    *,
    columns: Optional[Dict] = None,
    checks: Optional[Sequence] = None,
    index: Optional[IndexComponent] = None,
    size: Optional[int] = None,
):
    """Strategy to generate a pandas DataFrame.

    :param pandas_dtype: :class:`pandera.dtypes.PandasDtype` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param columns: a dictionary where keys are column names and values
        are :class:`~pandera.schema_components.Column` objects.
    :param checks: sequence of :class:`~pandera.checks.Check` s to constrain
        the values of the data at the dataframe level.
    :param index: Index or MultiIndex schema component.
    :param size: number of elements in the Series.
    :returns: ``hypothesis`` strategy.
    """
    # pylint: disable=too-many-locals
    if strategy:
        raise BaseStrategyOnlyError(
            "The dataframe strategy is a base strategy. You cannot specify "
            "the strategy argument to chain it to a parent strategy."
        )
    columns = {} if columns is None else columns
    checks = [] if checks is None else checks

    col_dtypes = {
        col_name: col.dtype if pandas_dtype is None else pandas_dtype.str_alias
        for col_name, col in columns.items()
    }

    nullable_columns = {
        col_name: col.nullable for col_name, col in columns.items()
    }

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
                strategy = check.strategy(col.pdtype, strategy)
            else:
                strategy = undefined_check_strategy(
                    strategy=(
                        pandas_dtype_strategy(col.pdtype)
                        if strategy is None
                        else strategy
                    ),
                    check=check,
                )
        if strategy is None:
            strategy = pandas_dtype_strategy(col.pdtype)
        return strategy

    row_strategy_checks = []
    undefined_strat_df_checks = []
    for check in checks:
        if hasattr(check, "strategy") or check.element_wise:
            # we can apply element-wise checks defined at the dataframe level
            # to the row strategy
            row_strategy_checks.append(check)
        else:
            undefined_strat_df_checks.append(check)

    # collect all non-element-wise column checks with undefined strategies
    undefined_strat_column_checks: Dict[str, list] = defaultdict(list)
    for col_name, column in columns.items():
        undefined_strat_column_checks[col_name].extend(
            check
            for check in column.checks
            if not hasattr(check, "strategy") and not check.element_wise
        )

    row_strategy = None
    if checks:
        row_strategy = st.fixed_dictionaries(
            {
                col_name: make_row_strategy(col, row_strategy_checks)
                for col_name, col in columns.items()
            }
        )

    strategy = pdst.data_frames(
        columns=[column.strategy_component() for column in columns.values()],
        rows=row_strategy,
        index=pdst.range_indexes(
            min_size=0 if size is None else size, max_size=size
        ),
    ).map(lambda df: df if df.empty else df.astype(col_dtypes))

    if any(nullable_columns.values()):
        strategy = null_dataframe_masks(strategy, nullable_columns)

    if index is not None:
        strategy = set_pandas_index(strategy, index.strategy(size=size))

    strategy = strategy.map(  # type: ignore
        lambda x: x.astype(pandas_dtype.str_alias)
        if pandas_dtype is not None
        else x
    )

    for check in undefined_strat_df_checks:
        strategy = undefined_check_strategy(strategy, check)

    for col_name, column_checks in undefined_strat_column_checks.items():
        for check in column_checks:  # type: ignore
            strategy = undefined_check_strategy(
                strategy, check, column=col_name
            )

    return strategy


# pylint: disable=unused-argument
def multiindex_strategy(
    pandas_dtype: Optional[PandasDtype] = None,
    strategy: Optional[SearchStrategy] = None,
    *,
    indexes: Optional[List] = None,
    size: Optional[int] = None,
):
    """Strategy to generate a pandas MultiIndex object.

    :param pandas_dtype: :class:`pandera.dtypes.PandasDtype` instance.
    :param strategy: an optional hypothesis strategy. If specified, the
        pandas dtype strategy will be chained onto this strategy.
    :param indexes: a list of :class:`~pandera.schema_components.Inded`
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
        index.name if index.name is not None else i: index.dtype
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
    if any(nullable_index.values()):
        strategy = null_dataframe_masks(strategy, nullable_index)
    return strategy.map(pd.MultiIndex.from_frame)
