"""Tests for Polars schema inference."""

from typing import Any, cast

import polars as pl
import pytest

import pandera.polars as pa
from pandera.schema_inference.polars import (
    infer_dataframe_schema,
    infer_schema,
)


def test_infer_dataframe_schema_basic() -> None:
    """Infer a schema from a Polars DataFrame."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    schema = infer_dataframe_schema(df)
    assert isinstance(schema, pa.DataFrameSchema)
    assert schema.coerce is True
    assert set(schema.columns) == {"a", "b"}


def test_infer_schema_lazyframe() -> None:
    """LazyFrames are collected before inference."""
    lf = pl.LazyFrame({"a": [1, 2]})
    schema = infer_schema(lf)
    assert isinstance(schema, pa.DataFrameSchema)
    assert "a" in schema.columns


def test_infer_schema_type_error() -> None:
    """infer_schema rejects non-Polars objects."""
    with pytest.raises(TypeError, match="Expected polars"):
        infer_schema(cast(Any, 1))


def test_empty_dataframe() -> None:
    """Empty frame yields empty schema."""
    df = pl.DataFrame()
    schema = infer_dataframe_schema(df)
    assert isinstance(schema, pa.DataFrameSchema)
    assert schema.columns == {}
