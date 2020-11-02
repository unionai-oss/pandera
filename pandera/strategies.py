"""Generate synthetic data from a schema definition.

This module is responsible for generating data based on the type and check
constraints specified in a ``pandera`` schema. It's built on top of the
`hypothesis <https://hypothesis.readthedocs.io/en/latest/index.html>`_ package
to compose strategies given multiple checks specified in a schema.
"""

import inspect
import operator
import re
from functools import partial, wraps
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .dtypes import PandasDtype

try:
    import hypothesis.extra.numpy as npst
    import hypothesis.extra.pandas as pdst
    import hypothesis.strategies as st
    from hypothesis import assume
    from hypothesis.strategies import SearchStrategy
except ImportError:
    HAS_HYPOTHESIS = False
else:
    HAS_HYPOTHESIS = True


StrategyFn = Callable[..., SearchStrategy]


class BaseStrategyOnlyError(Exception):
    """Custom error for reporting strategies that must be base strategies."""

    pass


def register_check_strategy(strategy_fn: StrategyFn):
    """Decorate a Check method with a strategy.

    This should be applied to a built-in Check method.
    """

    def register_check_strategy_decorator(class_method):

        if not HAS_HYPOTHESIS:
            return class_method

        @wraps(class_method)
        def _wrapper(cls, *args, **kwargs):
            check = class_method(cls, *args, **kwargs)
            if not hasattr(check, "statistics"):
                raise AttributeError(
                    "check object doesn't have a statistics property"
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


def pandas_dtype_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    **kwargs,
) -> SearchStrategy:

    if pandas_dtype is PandasDtype.Category:
        # hypothesis doesn't support categoricals, so will need to build a
        # solution for pandera.
        raise TypeError(
            "Categorical dtype is currently unsupported. Consider using "
            "a string or int dtype and Check.isin(values) to ensure a finite "
            "set of values."
        )

    dtype = pandas_dtype.numpy_dtype
    if pandas_dtype is PandasDtype.Object:
        # default to generating strings for generating objects
        dtype = np.dtype("str")

    if strategy:
        return strategy.map(dtype)
    return npst.from_dtype(dtype, **kwargs)


def eq_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    value: Any,
) -> SearchStrategy:
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
    if strategy is None:
        strategy = pandas_dtype_strategy(pandas_dtype)
    return strategy.filter(lambda x: x != value)


def gt_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    min_value: Union[int, float],
) -> SearchStrategy:
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
    if strategy is None:
        return pandas_dtype_strategy(
            pandas_dtype,
            min_value=min_value,
            max_value=max_value,
            exclude_min=not include_min,
            exclude_max=not include_max,
        )
    max_op = operator.lt if include_max else operator.le
    min_op = operator.gt if include_max else operator.gt
    return strategy.filter(
        lambda x: min_op(x, min_value) and max_op(x, max_value)
    )


def isin_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    allowed_values: Sequence[Any],
):
    if strategy is None:
        return st.sampled_from(allowed_values).map(pandas_dtype.numpy_dtype)
    return strategy.filter(lambda x: x in allowed_values)


def notin_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    forbidden_values: Sequence[Any],
):
    if strategy is None:
        strategy = pandas_dtype_strategy(pandas_dtype)
    return strategy.filter(lambda x: x not in forbidden_values)


def str_matches_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    pattern: str,
):
    if strategy is None:
        return st.from_regex(pattern, fullmatch=True).map(
            pandas_dtype.numpy_dtype
        )

    def matches(x):
        return re.match(x)

    return strategy.filter(matches)


def str_contains_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    pattern: str,
):
    if strategy is None:
        return st.from_regex(pattern, fullmatch=False).map(
            pandas_dtype.numpy_dtype
        )

    def contains(x):
        return re.search(x)

    return strategy.filter(contains)


def str_startswith_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    string: str,
):
    if strategy is None:
        return st.from_regex(f"^{string}", fullmatch=False).map(
            pandas_dtype.numpy_dtype
        )

    return strategy.filter(lambda x: x.startswith(string))


def str_endswith_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    string: str,
):
    if strategy is None:
        return st.from_regex(f"{string}$", fullmatch=False).map(
            pandas_dtype.numpy_dtype
        )

    return strategy.filter(lambda x: x.endswith(string))


def str_length_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    min_value: int,
    max_value: int,
):
    if strategy is None:
        return st.text(min_size=min_value, max_size=max_value).map(
            pandas_dtype.numpy_dtype
        )

    return strategy.filter(lambda x: min_value <= len(x) <= max_value)


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
    if strategy:
        raise BaseStrategyOnlyError(
            "The series strategy is a base strategy. You cannot specify the "
            "strategy argument to chain it to a parent strategy."
        )
    checks = [] if checks is None else checks
    elements = None
    for check in checks:
        elements = check.strategy(pandas_dtype, elements)
    if elements is None:
        elements = pandas_dtype_strategy(pandas_dtype)
    if not nullable:
        elements = elements.filter(lambda x: pd.notna(x))
    return (
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
    )


def column_strategy(
    pandas_dtype: PandasDtype,
    strategy: Optional[SearchStrategy] = None,
    *,
    checks: Optional[Sequence] = None,
    nullable: Optional[bool] = False,
    allow_duplicates: Optional[bool] = True,
    name: Optional[str] = None,
):
    if strategy:
        raise BaseStrategyOnlyError(
            "The column strategy is a base strategy. You cannot specify the "
            "strategy argument to chain it to a parent strategy."
        )
    checks = [] if checks is None else checks
    elements = None
    for check in checks:
        elements = check.strategy(pandas_dtype, elements)
    if elements is None:
        elements = pandas_dtype_strategy(pandas_dtype)
    if not nullable:
        elements = elements.filter(lambda x: pd.notna(x))

    return pdst.column(
        name=name,
        elements=elements,
        dtype=pandas_dtype.numpy_dtype,
        unique=not allow_duplicates,
    )


def dataframe_strategy(
    pandas_dtype: Optional[PandasDtype] = None,
    strategy: Optional[SearchStrategy] = None,
    *,
    columns: Optional[Dict] = None,
    size: Optional[int] = None,
):
    if strategy:
        raise BaseStrategyOnlyError(
            "The dataframe strategy is a base strategy. You cannot specify "
            "the strategy argument to chain it to a parent strategy."
        )
    # TODO: handle pandas_dtype being specified at the dataframe level
    # TODO: handle checks being defined at the dataframe level
    columns = {} if columns is None else columns
    col_dtypes = {col_name: col.dtype for col_name, col in columns.items()}
    return pdst.data_frames(
        [column.strategy() for column in columns.values()],
        index=pdst.range_indexes(
            min_size=0 if size is None else size, max_size=size
        ),
    ).map(lambda x: x.astype(col_dtypes))


def index_strategy():
    pass


def multiindex_strategy():
    pass
