"""Unit tests for polars validation based on configuration settings."""

import pytest

import polars as pl

import pandera.polars as pa
from pandera.config import CONFIG


def test_lazyframe_validation_default():
    """
    Test that with default configuration setting for validation depth (None),
    schema validation with LazyFrames is performed only on the schema.
    """
    CONFIG.validation_depth = None

    schema = pa.DataFrameSchema({"a": pa.Column(pl.Int64, pa.Check.gt(0))})

    valid = pl.LazyFrame({"a": [1, 2, 3]})
    invalid = pl.LazyFrame({"a": [1, 2, -3]})

    assert valid.pipe(schema.validate).collect().equals(valid.collect())
    assert invalid.pipe(schema.validate).collect().equals(invalid.collect())

    assert valid.collect().pipe(schema.validate).equals(valid.collect())
    with pytest.raises(pa.errors.SchemaError):
        invalid.collect().pipe(schema.validate)
