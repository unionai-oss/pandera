"""Test the error message produced from validating an invalid dataframe is correctly formatted`"""

import pytest
import pandas as pd

from pandera.api.checks import Check
from pandera.api.pandas.components import Column
from pandera.api.pandas.container import DataFrameSchema
from pandera.errors import SchemaErrors


def _mock_custom_wide_check(df):
    """Mock check function for us in the spec below"""
    return (df["column_1"] + df["column_2"] >= df["column_3"]).all()


@pytest.mark.parametrize(
    "schema, df, error_message",
    [
        (
            DataFrameSchema(
                {"color": Column(str), "size": Column(str)}, strict=True
            ),
            pd.DataFrame(
                {
                    "color": ["red", "blue", "green"],
                    "size": ["1A", "2", "3"],
                    "year": ["2022", "2021", "2024"],
                }
            ),
            """
Schema Nameless Schema: A total of 1 schema errors were found.
┌───────┬────────┬──────────────────┬──────────────┬─────────────────┬──────────────┐
│ index ┆ column ┆      check       ┆ failure_case ┆  schema_context ┆ check_number │
╞═══════╪════════╪══════════════════╪══════════════╪═════════════════╪══════════════╡
│ None  │ None   │ column_in_schema │ year         │ DataFrameSchema │ None         │
└───────┴────────┴──────────────────┴──────────────┴─────────────────┴──────────────┘
""",
        ),
        (
            DataFrameSchema(
                {
                    "flavour": Column(
                        str,
                        checks=[Check.isin(["coke", "7up", "mountain_dew"])],
                    )
                },
                name="MySchema",
                strict=True,
            ),
            pd.DataFrame({"flavour": ["pepsi", "coke", "fanta"]}),
            """
Schema MySchema: A total of 2 schema errors were found.
┌───────┬─────────┬──────────────────────┬──────────────┬────────────────┬──────────────┐
│ index ┆  column ┆        check         ┆ failure_case ┆ schema_context ┆ check_number │
╞═══════╪═════════╪══════════════════════╪══════════════╪════════════════╪══════════════╡
│ 0     │ flavour │ isin(['coke', '7up', │ pepsi        │ Column         │ 0            │
│       │         │ 'mountain_dew'])     │              │                │              │
│ 2     │ flavour │ isin(['coke', '7up', │ fanta        │ Column         │ 0            │
│       │         │ 'mountain_dew'])     │              │                │              │
└───────┴─────────┴──────────────────────┴──────────────┴────────────────┴──────────────┘
""",
        ),
        (
            DataFrameSchema(
                {
                    "column_1": Column(int),
                    "column_2": Column(int),
                    "column_3": Column(int),
                },
                name="Wide Schema",
                strict=True,
                checks=[Check(_mock_custom_wide_check)],
            ),
            pd.DataFrame(
                {
                    "column_1": [1, 2, 1],
                    "column_2": [1, 0, 1],
                    "column_3": [2, 2, 4],
                }
            ),
            """
Schema Wide Schema: A total of 1 schema errors were found.
┌───────┬─────────────┬────────────────────────┬──────────────┬─────────────────┬──────────────┐
│ index ┆    column   ┆          check         ┆ failure_case ┆  schema_context ┆ check_number │
╞═══════╪═════════════╪════════════════════════╪══════════════╪═════════════════╪══════════════╡
│ None  │ Wide Schema │  custom_wide_check     │ False        │ DataFrameSchema │ 0            │
└───────┴─────────────┴────────────────────────┴──────────────┴─────────────────┴──────────────┘
""",
        ),
    ],
)
def test_schema_error_messages(schema, df, error_message):
    """Test the SchemaErrors message produced by schema validation"""
    with pytest.raises(SchemaErrors) as e:
        schema.validate(df, lazy=True)

    assert error_message.strip() == str(e.value).strip()
