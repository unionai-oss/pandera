import hypothesis
import hypothesis.extra.numpy as numpy_st
import hypothesis.extra.pandas as pandas_st
import hypothesis.strategies as st
import pandas as pd
import pytest

import pandera as pa
import pandera.generators as generators


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


def test_column_generate():
    pass


def test_schema_generate():
    pass
