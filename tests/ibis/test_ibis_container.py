"""Unit tests for Ibis container."""

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
import pandas as pd
import pytest
from ibis import _, selectors as s

import pandera as pa
from pandera.api.ibis.types import IbisData
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
            "string_col": Column(dt.String),
            "int_col": Column(dt.Int64),
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
            "b": Column(dt.String, required=False),
        }
    )
    t = ibis.memtable(pd.DataFrame({"a": [1, 2, 3]}))
    assert schema.validate(t).execute().equals(t.execute())
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(t.rename({"c": "a"})).execute()


def test_dataframe_level_checks():
    def custom_check(data: IbisData):
        return data.table.select(s.across(s.all(), _ == 0))

    schema = DataFrameSchema(
        columns={"a": Column(dt.Int64), "b": Column(dt.Int64)},
        checks=[
            pa.Check(custom_check),
            pa.Check(lambda d: d.table.select(s.across(s.all(), _ == 0))),
        ],
    )
    t = ibis.memtable({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]})
    with pytest.raises(pa.errors.SchemaError):
        t.pipe(schema.validate)

    try:
        t.pipe(schema.validate, lazy=True)
    except pa.errors.SchemaErrors as err:
        assert err.failure_cases.shape[0] == 12
