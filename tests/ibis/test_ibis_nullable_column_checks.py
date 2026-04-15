"""Tests for ibis validation with nullable columns and check functions."""

import ibis
import pytest

import pandera.ibis


class TestIbisNullableColumnChecks:
    """Tests for ibis validation when nullable columns contain only null values."""

    def test_ibis_column_with_all_nulls_passes_nullable_check(self):
        """Validation should pass when checking a nullable column with all nulls."""
        data = {"my_value": [None, None]}
        df = ibis.memtable(data).cast({"my_value": "float64"})

        schema = pandera.ibis.DataFrameSchema(
            {
                "my_value": pandera.ibis.Column(
                    float,
                    nullable=True,
                    checks=[
                        pandera.ibis.Check.greater_than_or_equal_to(0),
                    ],
                ),
            }
        )

        result = schema.validate(df)
        assert isinstance(result, ibis.Table)

    def test_ibis_column_with_mixed_nulls_passes_nullable_check(self):
        """Validation should pass with some nulls and some valid values."""
        data = {"my_value": [42, None]}
        df = ibis.memtable(data).cast({"my_value": "float64"})

        schema = pandera.ibis.DataFrameSchema(
            {
                "my_value": pandera.ibis.Column(
                    float,
                    nullable=True,
                    checks=[
                        pandera.ibis.Check.greater_than_or_equal_to(0),
                    ],
                ),
            }
        )

        result = schema.validate(df)
        assert isinstance(result, ibis.Table)

    def test_ibis_column_with_invalid_values_fails(self):
        """Validation should fail when non-null values don't pass the check."""
        data = {"my_value": [-5, None]}
        df = ibis.memtable(data).cast({"my_value": "float64"})

        schema = pandera.ibis.DataFrameSchema(
            {
                "my_value": pandera.ibis.Column(
                    float,
                    nullable=True,
                    checks=[
                        pandera.ibis.Check.greater_than_or_equal_to(0),
                    ],
                ),
            }
        )

        with pytest.raises(pandera.ibis.errors.SchemaError):
            schema.validate(df)

    def test_ibis_column_with_all_valid_values_passes(self):
        """Validation should pass when all non-null values pass the check."""
        data = {"my_value": [10, 20]}
        df = ibis.memtable(data).cast({"my_value": "float64"})

        schema = pandera.ibis.DataFrameSchema(
            {
                "my_value": pandera.ibis.Column(
                    float,
                    nullable=True,
                    checks=[
                        pandera.ibis.Check.greater_than_or_equal_to(0),
                    ],
                ),
            }
        )

        result = schema.validate(df)
        assert isinstance(result, ibis.Table)