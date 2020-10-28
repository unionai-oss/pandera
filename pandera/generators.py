"""Generate synthetic data from a schema definition.

This module is responsible for generating data based on the type and check
constraints specified in a ``pandera`` schema. It's built on top of the
`hypothesis <https://hypothesis.readthedocs.io/en/latest/index.html>`_ package
to compose strategies given multiple checks specified in a schema.
"""

import operator
from typing import Any, Union

import hypothesis.extra.numpy as numpy_st
import hypothesis.extra.pandas as pandas_st
import hypothesis.strategies as st
import numpy as np
from hypothesis import assume
from hypothesis.strategies import SearchStrategy

from .dtypes import PandasDtype


def pandas_dtype_strategy(
    pandas_dtype: PandasDtype, strategy: SearchStrategy = None, **kwargs
):

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
    return numpy_st.from_dtype(dtype, **kwargs)


def resolve_strategies():
    """Wrapper function"""
    pass


def eq_strategy(
    pandas_dtype: PandasDtype, strategy: SearchStrategy = None, *, value: Any,
):
    # override strategy preceding this one and generate value of the same type
    if strategy is None:
        strategy = pandas_dtype_strategy(pandas_dtype)
    return st.just(value).map(pandas_dtype.numpy_dtype.type)


def ne_strategy(
    pandas_dtype: PandasDtype, strategy: SearchStrategy = None, *, value: Any,
):
    if strategy is None:
        strategy = pandas_dtype_strategy(pandas_dtype)
    return strategy.filter(lambda x: x != value)


def gt_strategy(
    pandas_dtype: PandasDtype,
    strategy: SearchStrategy = None,
    *,
    min_value: Union[int, float],
):
    if strategy is None:
        strategy = pandas_dtype_strategy(
            pandas_dtype,
            min_value=min_value,
            exclude_min=True if pandas_dtype.is_float else None,
        )
    return strategy.filter(lambda x: x > min_value)


def ge_strategy(
    pandas_dtype: PandasDtype,
    strategy: SearchStrategy = None,
    *,
    min_value: Union[int, float],
):
    if strategy is None:
        return pandas_dtype_strategy(
            pandas_dtype,
            min_value=min_value,
            exclude_min=False if pandas_dtype.is_float else None,
        )
    return strategy.filter(lambda x: x >= min_value)


def lt_strategy(
    pandas_dtype: PandasDtype,
    strategy: SearchStrategy = None,
    *,
    max_value: Union[int, float],
):
    if strategy is None:
        strategy = pandas_dtype_strategy(
            pandas_dtype,
            max_value=max_value,
            exclude_max=True if pandas_dtype.is_float else None,
        )
    return strategy.filter(lambda x: x < max_value)


def le_strategy(
    pandas_dtype: PandasDtype,
    strategy: SearchStrategy = None,
    *,
    max_value: Union[int, float],
):
    if strategy is None:
        return pandas_dtype_strategy(
            pandas_dtype,
            max_value=max_value,
            exclude_max=False if pandas_dtype.is_float else None,
        )
    return strategy.filter(lambda x: x <= max_value)


def in_range_strategy(
    pandas_dtype: PandasDtype,
    strategy: SearchStrategy = None,
    *,
    min_value: Union[int, float],
    max_value: Union[int, float],
    include_min: bool = True,
    include_max: bool = True,
):
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


def isin_strategy():
    pass


def notin_strategy():
    pass


def str_matches_strategy():
    pass


def str_contains_strategy():
    pass


def str_startswith_strategy():
    pass


def str_length_strategy():
    pass


def strategy_from_column():
    pass


def strategy_from_series_schema():
    pass


def strategy_from_dataframe_schema():
    pass
