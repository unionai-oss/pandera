"""Test for issue #2294: ibis validation fails checks if column all nulls."""

import pytest

import pandera.ibis
import ibis


class TestIssue2294:
    """Tests for the fix to issue #2294."""

    def test_ibis_column_with_all_nulls_passes_nullable_check(self):
        """Ibis validation should pass when checking a nullable column with all nulls."""
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

        # Should not raise - all nulls should be ignored
        result = schema.validate(df)
        assert isinstance(result, ibis.Table)

    def test_ibis_column_with_mixed_nulls_passes_nullable_check(self):
        """Ibis validation should pass with some nulls and some valid values."""
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
        """Ibis validation should fail when non-null values don't pass checks."""
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
        """Ibis validation should pass when all non-null values pass checks."""
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
