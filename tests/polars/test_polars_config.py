# pylint: disable=unused-argument
"""Unit tests for polars validation based on configuration settings."""

import pytest

import polars as pl

import pandera.polars as pa
from pandera.config import (
    CONFIG,
    ValidationDepth,
    config_context,
    get_config_global,
    get_config_context,
    reset_config_context,
)


@pytest.fixture(scope="function")
def validation_depth_none():
    """
    These tests ensure that the validation depth is set to 'None'
    for unit tests.
    """
    _validation_depth = CONFIG.validation_depth
    CONFIG.validation_depth = None
    try:
        yield
    finally:
        CONFIG.validation_depth = _validation_depth
        reset_config_context()


@pytest.fixture(scope="function")
def validation_depth_schema_and_data():
    """
    These tests ensure that the validation depth is set to 'None'
    for unit tests.
    """
    _validation_depth = CONFIG.validation_depth
    CONFIG.validation_depth = ValidationDepth.SCHEMA_AND_DATA
    try:
        yield
    finally:
        CONFIG.validation_depth = _validation_depth
        reset_config_context()


def test_lazyframe_validation_depth_none(validation_depth_none):
    """
    Test that with default configuration setting for validation depth (None),
    schema validation with LazyFrames is performed only on the schema.
    """
    schema = pa.DataFrameSchema({"a": pa.Column(pl.Int64, pa.Check.gt(0))})
    valid = pl.LazyFrame({"a": [1, 2, 3]})
    invalid = pl.LazyFrame({"a": [1, 2, -3]})

    assert valid.pipe(schema.validate).collect().equals(valid.collect())
    assert invalid.pipe(schema.validate).collect().equals(invalid.collect())

    with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
        assert valid.collect().pipe(schema.validate).equals(valid.collect())
        with pytest.raises(pa.errors.SchemaError):
            invalid.collect().pipe(schema.validate)


def test_lazyframe_validation_depth_schema_and_data(
    validation_depth_schema_and_data,
):
    """
    Test that with default configuration setting for validation depth (None),
    schema validation with LazyFrames is performed only on the schema.
    """
    schema = pa.DataFrameSchema({"a": pa.Column(pl.Int64, pa.Check.gt(0))})
    valid = pl.LazyFrame({"a": [1, 2, 3]})
    invalid = pl.LazyFrame({"a": [1, 2, -3]})

    assert valid.collect().pipe(schema.validate).equals(valid.collect())
    with pytest.raises(pa.errors.SchemaError):
        invalid.collect().pipe(schema.validate)


def test_coerce_validation_depth_none(validation_depth_none):
    assert get_config_global().validation_depth is None
    schema = pa.DataFrameSchema({"a": pa.Column(int)}, coerce=True)
    data = pl.LazyFrame({"a": ["1", "2", "foo"]})

    # simply calling validation shouldn't raise a coercion error, since we're
    # casting the types lazily
    validated_data = schema.validate(data)
    assert validated_data.schema["a"] == pl.Int64

    with pytest.raises(pl.ComputeError):
        validated_data.collect()

    # when validation explicitly with PANDERA_VALIDATION_DEPTH=SCHEMA_AND_DATA
    with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
        assert (
            get_config_context().validation_depth
            == ValidationDepth.SCHEMA_AND_DATA
        )
        try:
            schema.validate(data)
        except pa.errors.SchemaError as exc:
            assert exc.failure_cases.rows(named=True) == [{"a": "foo"}]
