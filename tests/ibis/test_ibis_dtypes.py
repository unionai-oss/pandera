"""Ibis dtype tests."""

import ibis
import ibis.expr.datatypes as dt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from polars.testing.parametric import dataframes

from pandera.engines import ibis_engine as ie

NUMERIC_TYPES = [
    ie.Int32,
    ie.Int64,
    ie.Float64,
]

TEMPORAL_TYPES = []  # type: ignore # TODO(deepyaman): Delete annotation once populated.

OTHER_TYPES = [
    ie.String,
]

ALL_TYPES = NUMERIC_TYPES + TEMPORAL_TYPES + OTHER_TYPES


@st.composite
def memtables(draw, **kwargs):
    return ibis.memtable(draw(dataframes(**kwargs)))


def get_table_strategy(dtype: dt.DataType) -> st.SearchStrategy:
    """Get a strategy for an Ibis table of a given dtype."""
    return memtables(
        cols=2,
        allowed_dtypes=dtype.to_polars(),
        allow_null=False,
        min_size=10,
        max_size=10,
    )


@pytest.mark.parametrize("dtype", ALL_TYPES)
@given(st.data())
@settings(max_examples=10)
def test_coerce_no_cast(dtype, data):
    """Test that dtypes can be coerced without casting."""
    pandera_dtype = dtype()
    t = data.draw(get_table_strategy(dtype=pandera_dtype.type))
    coerced = pandera_dtype.coerce(t)
    assert_frame_equal(t.to_polars(), coerced.to_polars())
