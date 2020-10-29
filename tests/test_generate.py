from typing import Any

import hypothesis
import hypothesis.extra.numpy as numpy_st
import hypothesis.extra.pandas as pandas_st
import hypothesis.strategies as st
import pandas as pd
import pytest

import pandera as pa
import pandera.generators as generators
from pandera.checks import _CheckBase, register_check_statistics


@pytest.mark.parametrize(
    "pdtype", [pdtype for pdtype in pa.PandasDtype],
)
@hypothesis.given(st.data())
def test_pandas_dtype_strategy(pdtype, data):
    """Test that series can be constructed from pandas dtype."""
    if pdtype is pa.PandasDtype.Category:
        with pytest.raises(
            TypeError, match="Categorical dtype is currently unsupported"
        ):
            generators.pandas_dtype_strategy(pdtype)
        return

    strategy = generators.pandas_dtype_strategy(pdtype)
    pd.Series([data.draw(strategy)], dtype=pdtype.str_alias)


@hypothesis.given(st.data())
def test_check_strategy(data):
    pdtype = pa.PandasDtype.Int
    value = data.draw(numpy_st.from_dtype(pdtype.numpy_dtype))
    min_value, max_value = value - 10, value + 10

    assert data.draw(generators.ne_strategy(pdtype, value=value)) != value
    assert data.draw(generators.eq_strategy(pdtype, value=value)) == value
    assert data.draw(generators.gt_strategy(pdtype, min_value=value)) > value
    assert data.draw(generators.ge_strategy(pdtype, min_value=value)) >= value
    assert data.draw(generators.lt_strategy(pdtype, max_value=value)) < value
    assert data.draw(generators.le_strategy(pdtype, max_value=value)) <= value
    assert (
        min_value
        <= data.draw(
            generators.in_range_strategy(
                pdtype, min_value=min_value, max_value=max_value
            )
        )
        <= max_value
    )


@hypothesis.settings(
    suppress_health_check=[
        hypothesis.HealthCheck.filter_too_much,
        hypothesis.HealthCheck.too_slow,
    ]
)
@hypothesis.given(st.data())
def test_check_in_range_strategy_chained(data):
    # TODO: test when in_range_strategy is a second check in a series
    pass


@hypothesis.given(st.data())
def test_register_check_strategy(data):
    def custom_eq_strategy(
        pandas_dtype: pa.PandasDtype,
        strategy: st.SearchStrategy = None,
        *,
        value: Any,
    ):
        return st.just(value).map(pandas_dtype.numpy_dtype.type)

    class CustomCheck(_CheckBase):
        @classmethod
        @generators.register_check_strategy(custom_eq_strategy)
        @register_check_statistics(["value"])
        def custom_equals(cls, value, **kwargs) -> "CustomCheck":
            """Define a built-in check."""

            def _custom_equals(series: pd.Series) -> pd.Series:
                """Comparison function for check"""
                return series == value

            return cls(
                _custom_equals,
                name=cls.custom_equals.__name__,
                error="equal_to(%s)" % value,
                **kwargs,
            )

    check = CustomCheck.custom_equals(100)
    result = data.draw(check.strategy(pa.Int))
    assert result == 100


def test_column_generate():
    pass


def test_schema_generate():
    pass
