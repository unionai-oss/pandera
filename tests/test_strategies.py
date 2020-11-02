from typing import Any

import hypothesis
import hypothesis.extra.numpy as npst
import hypothesis.extra.pandas as pdst
import hypothesis.strategies as st
import pandas as pd
import pytest

import pandera as pa
import pandera.strategies as strategies
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
            strategies.pandas_dtype_strategy(pdtype)
        return

    strategy = strategies.pandas_dtype_strategy(pdtype)
    pd.Series([data.draw(strategy)], dtype=pdtype.str_alias)


@hypothesis.given(st.data())
def test_check_strategy(data):
    pdtype = pa.PandasDtype.Int
    value = data.draw(npst.from_dtype(pdtype.numpy_dtype))
    min_value, max_value = value - 10, value + 10

    assert data.draw(strategies.ne_strategy(pdtype, value=value)) != value
    assert data.draw(strategies.eq_strategy(pdtype, value=value)) == value
    assert data.draw(strategies.gt_strategy(pdtype, min_value=value)) > value
    assert data.draw(strategies.ge_strategy(pdtype, min_value=value)) >= value
    assert data.draw(strategies.lt_strategy(pdtype, max_value=value)) < value
    assert data.draw(strategies.le_strategy(pdtype, max_value=value)) <= value
    assert (
        min_value
        <= data.draw(
            strategies.in_range_strategy(
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
        @strategies.register_check_strategy(custom_eq_strategy)
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


@hypothesis.given(st.data())
def test_builtin_check_strategies(data):
    pdtype = pa.Int8
    value = data.draw(npst.from_dtype(pdtype.numpy_dtype))
    check = pa.Check.equal_to(value)
    strategy = check.strategy(pdtype)
    assert data.draw(strategy) == value


# TODO: test the rest of the built-in check strategies


@hypothesis.given(st.data())
def test_series_strategy(data):
    series_schema = pa.SeriesSchema(pa.Int, pa.Check.gt(0))
    series_schema(data.draw(series_schema.strategy()))


@hypothesis.given(st.data())
def test_column_strategy(data):
    column_schema = pa.Column(pa.Int, pa.Check.gt(0), name="column")
    column_schema(data.draw(column_schema.strategy(as_component=False)))


@pytest.mark.parametrize(
    "pdtype",
    [
        pdtype
        for pdtype in pa.PandasDtype
        if not pdtype.is_category and not pdtype.is_complex
    ],
)
@hypothesis.given(st.data())
def test_dataframe_strategy(pdtype, data):
    dataframe_schema = pa.DataFrameSchema(
        {f"{pdtype.value}_col": pa.Column(pdtype)}
    )
    dataframe_schema(data.draw(dataframe_schema.strategy(size=5)))


@pytest.mark.parametrize(
    "pdtype", [pdtype for pdtype in pa.PandasDtype if pdtype.is_complex],
)
@pytest.mark.filterwarnings("ignore:overflow encountered in absolute")
@hypothesis.given(st.data())
def test_dataframe_strategy_complex_numbers(pdtype, data):
    dataframe_schema = pa.DataFrameSchema(
        {f"{pdtype.value}_col": pa.Column(pdtype)}
    )
    dataframe_schema(data.draw(dataframe_schema.strategy(size=5)))


def test_index_strategy():
    pass


def test_multiindex_strategy():
    pass
