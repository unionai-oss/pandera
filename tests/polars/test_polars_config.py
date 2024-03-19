"""Unit tests for polars validation based on configuration settings."""

import pytest

import polars as pl

import pandera.polars as pa
from pandera.config import CONFIG


@pytest.fixture(scope="function", autouse=True)
def set_validation_depth():
    """
    These tests ensure that the validation depth is set to 'None'
    for unit tests.
    """
    _validation_depth = CONFIG.validation_depth
    CONFIG.validation_depth = None
    yield
    CONFIG.validation_depth = _validation_depth


def test_lazyframe_validation_default():
    """
    Test that with default configuration setting for validation depth (None),
    schema validation with LazyFrames is performed only on the schema.
    """
    schema = pa.DataFrameSchema({"a": pa.Column(pl.Int64, pa.Check.gt(0))})

    valid = pl.LazyFrame({"a": [1, 2, 3]})
    invalid = pl.LazyFrame({"a": [1, 2, -3]})

    assert valid.pipe(schema.validate).collect().equals(valid.collect())
    assert invalid.pipe(schema.validate).collect().equals(invalid.collect())

    assert valid.collect().pipe(schema.validate).equals(valid.collect())
    with pytest.raises(pa.errors.SchemaError):
        invalid.collect().pipe(schema.validate)
