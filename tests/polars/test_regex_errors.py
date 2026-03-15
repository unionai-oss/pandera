"""Tests for regex column matching errors."""

import pytest
import polars as pl

import pandera.polars as pa


def test_regex_no_match_error():
    """Test that a descriptive error is raised when regex pattern matches no columns."""
    ldf = pl.DataFrame(
        {
            "var_1": [0.4, 0.3, 0.9],
            "var_2": [0.5, 0.7, 0.8],
            "var_3": [1.2, 1.1, 0.2],
        }
    ).lazy()

    # Use regex pattern that won't match (var_x instead of var_)
    schema = pa.DataFrameSchema(
        {
            "var_x.*": pa.Column(
                float,
                regex=True,
            ),
        }
    )

    # Should raise SchemaError with descriptive message about regex not matching
    with pytest.raises(
        pa.errors.SchemaError,
        match=r"did not match any columns in the dataframe",
    ):
        schema.validate(ldf)

    # Also verify the list of available columns is included in the error
    with pytest.raises(pa.errors.SchemaError) as exc_info:
        schema.validate(ldf)

    error_message = str(exc_info.value)
    assert "var_1" in error_message
    assert "var_2" in error_message
    assert "var_3" in error_message


def test_regex_pattern_does_not_match_any_columns():
    """Test that a descriptive error is raised when regex pattern matches no columns."""
    ldf = pl.DataFrame(
        {
            "var_1": [0.4, 0.3, 0.9],
            "var_2": [0.5, 0.7, 0.8],
            "var_3": [1.2, 1.1, 0.2],
        }
    ).lazy()

    # Use a regex pattern that won't match ('var_x' instead of 'var_')
    schema = pa.DataFrameSchema(
        {
            "var_x.*": pa.Column(
                float,
                regex=True,
            ),
        }
    )

    # Should raise SchemaError with descriptive message
    with pytest.raises(
        pa.errors.SchemaError,
        match=r"Column regex name='var_x\.' did not match any columns",
    ):
        schema.validate(ldf)


def test_regex_pattern_does_not_match_any_columns():
    """Test error when regex pattern does not match any columns in dataframe."""
    ldf = pl.DataFrame(
        {
            "var_1": [0.4, 0.3, 0.9],
            "var_2": [0.5, 0.7, 0.8],
        }
    ).lazy()

    # Use a regex pattern that won't match ('var_x' instead of 'var_')
    schema = pa.DataFrameSchema(
        {
            "var_x.*": pa.Column(
                float,
                regex=True,
            ),
        }
    )

    # Should raise SchemaError with descriptive message about regex not matching
    with pytest.raises(
        pa.errors.SchemaError,
        match=r"did not match any columns in the dataframe",
    ):
        schema.validate(ldf)

    # Also verify the list of available columns is included in the error
    with pytest.raises(pa.errors.SchemaError) as exc_info:
        schema.validate(ldf)

    error_message = str(exc_info.value)
    assert "var_1" in error_message
    assert "var_2" in error_message
