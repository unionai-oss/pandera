# pylint: disable=unused-argument
"""Unit tests for polars validation based on configuration settings."""

import polars as pl
import pytest

import pandera.polars as pa
from pandera.engines.polars_engine import polars_version
from pandera.api.polars.utils import get_lazyframe_schema
from pandera.api.base.error_handler import ErrorCategory
from pandera.config import (
    CONFIG,
    ValidationDepth,
    config_context,
    get_config_context,
    get_config_global,
    reset_config_context,
)


@pytest.fixture(scope="function")
def validation_disabled():
    """Disable validation in the context of a fixture."""
    _validation_enabled = CONFIG.validation_enabled
    CONFIG.validation_enabled = False
    reset_config_context()
    try:
        yield
    finally:
        CONFIG.validation_enabled = _validation_enabled
        reset_config_context()


@pytest.fixture(scope="function")
def validation_depth_none():
    """Ensure that the validation depth is set to None for unit tests.

    This fixture is meant to simulate setting validation depth via the
    PANDERA_VALIDATION_DEPTH environment variable.
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
    """Ensure that the validation depth is set to SCHEMA_AND_DATA.

    This fixture is meant to simulate setting validation depth via the
    PANDERA_VALIDATION_DEPTH environment variable.
    """
    _validation_depth = CONFIG.validation_depth
    CONFIG.validation_depth = ValidationDepth.SCHEMA_AND_DATA
    try:
        yield
    finally:
        CONFIG.validation_depth = _validation_depth
        reset_config_context()


@pytest.fixture
def schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "a": pa.Column(pl.Int64, pa.Check.gt(0)),
            "b": pa.Column(pl.Utf8),
        }
    )


def test_validation_disabled(validation_disabled, schema):
    """Test that disabling validation doesn't raise errors for invalid data."""
    invalid = pl.DataFrame({"a": [-1, 2, 3], "b": [*"abc"]})
    assert schema.validate(invalid).equals(invalid)


def test_lazyframe_validation_depth_none(validation_depth_none, schema):
    """
    Test that with default configuration setting for validation depth (None),
    schema validation with LazyFrames is performed only on the schema.
    """
    valid = pl.LazyFrame({"a": [1, 2, 3], "b": [*"abc"]})
    invalid_data_level = pl.LazyFrame({"a": [1, 2, -3], "b": [*"abc"]})
    invalid_schema_level = pl.LazyFrame({"a": [1, 2, 3], "b": [1, 2, 3]})

    # validating LazyFrames should only validate schema-level properties, even
    # invalid dataframe should not raise an error.
    assert schema.validate(valid).collect().equals(valid.collect())
    assert (
        schema.validate(invalid_data_level)
        .collect()
        .equals(invalid_data_level.collect())
    )

    # invalid schema-level data should only have SCHEMA errors
    try:
        invalid_schema_level.pipe(schema.validate, lazy=True)
    except pa.errors.SchemaErrors as exc:
        assert ErrorCategory.SCHEMA.name in exc.message
        assert ErrorCategory.DATA.name not in exc.message

    # invalid data-level data should only have DATA errors
    try:
        invalid_data_level.pipe(schema.validate, lazy=True)
    except pa.errors.SchemaErrors as exc:
        assert ErrorCategory.SCHEMA.name not in exc.message
        assert ErrorCategory.DATA.name in exc.message

    # test that using config context manager while environment-level validation
    # depth is None can activate schema-and-data validation for LazyFrames.
    with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
        assert valid.collect().pipe(schema.validate).equals(valid.collect())
        with pytest.raises(
            pa.errors.SchemaError,
            match="Column 'a' failed validator .+ <Check greater_than",
        ):
            invalid_data_level.pipe(schema.validate)


def test_dataframe_validation_depth_none(validation_depth_none, schema):
    valid = pl.DataFrame({"a": [1, 2, 3], "b": [*"abc"]})
    invalid_data_level = pl.DataFrame({"a": [1, 2, -3], "b": [*"abc"]})
    invalid_schema_level = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})

    assert schema.validate(valid).equals(valid)

    # pl.DataFrame validation should validate schema- and data-level errors
    with pytest.raises(
        pa.errors.SchemaError,
        match="Column 'a' failed validator .+ <Check greater_than",
    ):
        assert schema.validate(invalid_data_level)

    with pytest.raises(
        pa.errors.SchemaError,
        match="expected column 'b' to have type String, got Int64",
    ):
        assert schema.validate(invalid_schema_level)


def test_lazyframe_validation_depth_schema_and_data(
    validation_depth_schema_and_data,
    schema,
):
    """
    Test that setting environment-level config for validation depth to
    SCHEMA_AND_DATA will perform data-level validation on LazyFrames.
    """
    valid = pl.LazyFrame({"a": [1, 2, 3], "b": [*"abc"]})
    invalid = pl.LazyFrame({"a": [1, 2, -3], "b": [1, 2, 3]})

    assert valid.pipe(schema.validate).collect().equals(valid.collect())

    with pytest.raises(
        pa.errors.SchemaError,
        match="Column 'a' failed validator .+ <Check greater_than",
    ):
        invalid.pipe(schema.validate)

    try:
        invalid.pipe(schema.validate, lazy=True)
    except pa.errors.SchemaErrors as exc:
        assert ErrorCategory.SCHEMA.name in exc.message
        assert ErrorCategory.DATA.name in exc.message


def test_coerce_validation_depth_none(validation_depth_none, schema):
    assert get_config_global().validation_depth is None
    schema._coerce = True
    data = pl.LazyFrame({"a": ["1", "2", "foo"], "b": [*"abc"]})

    # simply calling validation shouldn't raise a coercion error, since we're
    # casting the types lazily
    validated_data = schema.validate(data)
    assert get_lazyframe_schema(validated_data)["a"] == pl.Int64

    ErrorCls = (
        pl.exceptions.InvalidOperationError
        if polars_version().release >= (1, 0, 0)
        else pl.exceptions.ComputeError
    )
    with pytest.raises(ErrorCls):
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
