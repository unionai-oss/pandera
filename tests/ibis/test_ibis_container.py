"""Unit tests for Ibis container."""

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
import pandas as pd
import pytest

import pandera as pa
from pandera.ibis import Column, DataFrameSchema


@pytest.fixture
def t_basic():
    """Basic Ibis table fixture."""
    df = pd.DataFrame(
        {
            "string_col": ["0", "1", "2"],
            "int_col": [0, 1, 2],
        }
    )
    return ibis.memtable(df, name="t")


@pytest.fixture
def t_schema_basic():
    """Basic Ibis table schema fixture."""
    return DataFrameSchema(
        {
            # "string_col": Column(str),
            "int_col": Column(int),
        }
    )


def test_basic_ibis_table(t_basic, t_schema_basic):
    """Test basic Ibis table."""
    query = t_schema_basic.validate(t_basic)
    assert isinstance(query, ir.Table)


def test_basic_ibis_table_dtype_error(t_basic, t_schema_basic):
    """Test basic Ibis table."""
    t = t_basic.mutate(int_col=t_basic.int_col.cast("int32"))
    with pytest.raises(pa.errors.SchemaError):
        # type check errors occur even before collection
        t_schema_basic.validate(t)


def test_required_columns():
    """Test required columns."""
    schema = DataFrameSchema(
        {
            "a": Column(dt.Int64, required=True),
            "b": Column(dt.Int64, required=False),
        }
    )
    t = ibis.memtable(pd.DataFrame({"a": [1, 2, 3]}))
    assert schema.validate(t).execute().equals(t.execute())
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(t.rename({"c": "a"})).execute()
