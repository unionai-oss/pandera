"""Tests for Ibis schema inference."""

import ibis
import pandas as pd
import pytest

import pandera.ibis as pa
from pandera.schema_inference.ibis import infer_dataframe_schema, infer_schema


def test_infer_dataframe_schema_basic() -> None:
    """Infer a schema from an Ibis in-memory table."""
    t = ibis.memtable(
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
    )
    schema = infer_dataframe_schema(t)
    assert isinstance(schema, pa.DataFrameSchema)
    assert schema.coerce is True
    assert set(schema.columns) == {"a", "b"}


def test_infer_schema_alias() -> None:
    """infer_schema matches infer_dataframe_schema for valid tables."""
    t = ibis.memtable(pd.DataFrame({"a": [1]}))
    assert (
        infer_schema(t).columns.keys()
        == infer_dataframe_schema(t).columns.keys()
    )


def test_infer_schema_wrong_type() -> None:
    """infer_schema rejects non-Ibis tables."""
    with pytest.raises(TypeError, match="Expected ibis.Table"):
        infer_schema(pd.DataFrame())  # type: ignore[arg-type]
