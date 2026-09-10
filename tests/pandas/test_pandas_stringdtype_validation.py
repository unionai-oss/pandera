"""Tests for pandas DataFrameSchema validation with StringDtype columns."""

import pandas as pd
import pytest

import pandera.pandas as pa
from pandera.errors import SchemaError


class TestPandasStringDtypeValidation:
    """Tests for pandas DataFrameSchema validation with StringDtype columns."""

    def test_empty_dataframe_with_stringdtype_fails(self):
        """Empty DataFrame with object dtype should fail when schema expects StringDtype."""
        schema = pa.DataFrameSchema({"name": pa.Column(pd.StringDtype())})
        df = pd.DataFrame(columns=["name"])

        with pytest.raises(
            SchemaError, match="expected series 'name' to have type string"
        ):
            schema.validate(df)

    def test_empty_dataframe_with_object_dtype_passes(self):
        """Empty DataFrame with object dtype should pass when schema expects object."""
        schema = pa.DataFrameSchema({"name": pa.Column(object)})
        df = pd.DataFrame(columns=["name"])

        result = schema.validate(df)
        assert isinstance(result, pd.DataFrame)

    def test_dataframe_with_object_value_fails_stringdtype(self):
        """DataFrame with Object() value should fail when schema expects StringDtype."""

        class CustomObject:
            pass

        schema = pa.DataFrameSchema({"name": pa.Column(pd.StringDtype())})
        df = pd.DataFrame({"name": [CustomObject()]})

        with pytest.raises(
            SchemaError, match="expected series 'name' to have type string"
        ):
            schema.validate(df)

    def test_dataframe_with_string_values_passes(self):
        """DataFrame with valid strings should pass."""
        schema = pa.DataFrameSchema({"name": pa.Column(pd.StringDtype())})
        df = pd.DataFrame({"name": ["test", "another"]})

        result = schema.validate(df)
        assert isinstance(result, pd.DataFrame)

    def test_dataframe_with_none_nullable_true_passes(self):
        """DataFrame with None should pass when nullable=True."""
        schema = pa.DataFrameSchema(
            {"name": pa.Column(pd.StringDtype(), nullable=True)}
        )
        df = pd.DataFrame({"name": [None]})

        result = schema.validate(df)
        assert isinstance(result, pd.DataFrame)

    def test_dataframe_with_none_nullable_false_fails(self):
        """DataFrame with None should fail when nullable=False."""
        schema = pa.DataFrameSchema(
            {"name": pa.Column(pd.StringDtype(), nullable=False)}
        )
        df = pd.DataFrame({"name": [None]})

        with pytest.raises(SchemaError, match="non-nullable series"):
            schema.validate(df)

    def test_dataframe_with_mixed_values_passes(self):
        """DataFrame with mixed None and strings should pass when nullable=True."""
        schema = pa.DataFrameSchema(
            {"name": pa.Column(pd.StringDtype(), nullable=True)}
        )
        df = pd.DataFrame({"name": ["test", None, "another"]})

        result = schema.validate(df)
        assert isinstance(result, pd.DataFrame)

    def test_dataframe_with_mixed_values_and_object_fails(self):
        """DataFrame with mixed values including Object() should fail."""

        class CustomObject:
            pass

        schema = pa.DataFrameSchema(
            {"name": pa.Column(pd.StringDtype(), nullable=True)}
        )
        df = pd.DataFrame({"name": ["test", None, CustomObject()]})

        with pytest.raises(
            SchemaError, match="expected series 'name' to have type string"
        ):
            schema.validate(df)

    def test_empty_dataframe_multiple_columns(self):
        """Empty DataFrame with multiple columns should fail."""
        schema = pa.DataFrameSchema(
            {"name": pa.Column(pd.StringDtype()), "value": pa.Column(int)}
        )
        df = pd.DataFrame(columns=["name", "value"])

        with pytest.raises(SchemaError, match="expected series"):
            schema.validate(df)

    def test_empty_dataframe_with_stringdtype_and_object_column(self):
        """Empty DataFrame with mixed column types should fail."""
        schema = pa.DataFrameSchema(
            {"name": pa.Column(pd.StringDtype()), "data": pa.Column(object)}
        )
        df = pd.DataFrame(columns=["name", "data"])

        with pytest.raises(
            SchemaError, match="expected series 'name' to have type string"
        ):
            schema.validate(df)
