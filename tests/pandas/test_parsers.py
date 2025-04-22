"""Tests the way Columns are Parsed"""

import copy

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

import pandera.pandas as pa
from pandera.api.pandas.array import SeriesSchema
from pandera.api.pandas.container import DataFrameSchema
from pandera.api.parsers import Parser
from pandera.typing import Series


def test_dataframe_schema_parse() -> None:
    """Test that DataFrameSchema-level Parses work properly."""
    data = pd.DataFrame([[1, 4, 9, 16, 25] for _ in range(10)])

    schema_check_return_bool = DataFrameSchema(
        parsers=Parser(lambda df: df.transform("sqrt"))
    )
    assert schema_check_return_bool.validate(data).equals(data.apply(np.sqrt))


def test_dataframe_schema_parse_with_element_wise() -> None:
    """Test that DataFrameSchema-level Parses work properly."""
    data = pd.DataFrame([[1, 4, 9, 16, 25] for _ in range(10)])
    schema_check_return_bool = DataFrameSchema(
        parsers=Parser(np.sqrt, element_wise=True)
    )
    result = (
        data.map(np.sqrt) if hasattr(data, "map") else data.applymap(np.sqrt)
    )
    assert schema_check_return_bool.validate(data).equals(result)


def test_series_schema_parse_with_element_wise() -> None:
    data = pd.Series([1, 4, 9, 16, 25])
    schema_check_return_bool = SeriesSchema(
        parsers=Parser(np.sqrt, element_wise=True)
    )
    result = (
        data.map(np.sqrt) if hasattr(data, "map") else data.applymap(np.sqrt)
    )
    assert schema_check_return_bool.validate(data).equals(result)


def test_parser_equality_operators() -> None:
    """Test the usage of == between a Parser and an entirely different Parser,
    and a non-Parser."""
    parser = Parser(lambda g: g["foo"]["col1"].iat[0] == 1)

    not_equal_parser = Parser(lambda x: x.isna().sum() == 0)
    assert parser == copy.deepcopy(parser)
    assert parser != not_equal_parser
    assert parser != "not a parser"


def test_equality_operators_functional_equivalence() -> None:
    """Test the usage of == for Parsers where the Parser callable object has
    the same implementation."""
    main_parser = Parser(lambda g: g["foo"]["col1"].iat[0] == 1)
    same_parser = Parser(lambda h: h["foo"]["col1"].iat[0] == 1)

    assert main_parser == same_parser


def test_check_backend_not_found():
    """Test that parsers complain if a backend is not register for that type."""

    class CustomDataObject:
        """Custom data object."""

    dummy_check = Parser(lambda _: True)

    with pytest.raises(KeyError, match="Backend not found for class"):
        dummy_check(CustomDataObject())


def test_parser_non_existing() -> None:
    """Test a check on a non-existing column."""

    class Schema(pa.DataFrameModel):
        a: Series[int]

        @pa.check("nope")
        @classmethod
        def int_column_lt_100(cls, series: pd.Series):
            return series < 100

    err_msg = (
        "Check int_column_lt_100 is assigned to a non-existing field 'nope'"
    )
    with pytest.raises(pa.errors.SchemaInitError, match=err_msg):
        Schema.to_schema()


def test_parser_called_once():

    data = pd.DataFrame({"col": [2.0, 4.0, 9.0]})
    n_calls = 0

    class DFModel(pa.DataFrameModel):
        col: float

        @pa.parser("col")
        @classmethod
        def negate(cls, series):
            nonlocal n_calls
            n_calls += 1
            return series * -1

    DFModel.validate(data)
    assert n_calls == 1


def test_parser_with_coercion():
    """Make sure that parser is applied before coercion."""

    class SchemaParserWithIntCoercion(pa.DataFrameModel):
        column: pd.Int64Dtype = pa.Field(nullable=True, coerce=True)

        @pa.dataframe_parser
        @classmethod
        def replace_empty_with_na(cls, df: pd.DataFrame) -> pd.DataFrame:
            return df.replace(["", " ", "nan"], pd.NA)

    df = pd.DataFrame({"column": ["", " ", "nan", 100, ""]})
    result = SchemaParserWithIntCoercion.validate(df)
    assert result["column"].dtype == pd.Int64Dtype()
    assert result["column"].isna().sum() == 4

    class SchemaWithCategoryCoercion(pa.DataFrameModel):

        col1: pd.CategoricalDtype = pa.Field(
            dtype_kwargs={"categories": ["category1", "category2"]}
        )
        col2: pd.StringDtype

        @pa.dataframe_parser
        @classmethod
        def normalize_string_values(cls, df):
            return df.assign(
                col1=df["col1"].str.strip().str.lower(),
                col2=df["col2"].str.strip().str.lower(),
            )

        class Config:
            coerce = True

    test_df = pd.DataFrame(
        data={"col1": ["CATEGORY1", "CatEGory2 "], "col2": ["foo", "Bar"]},
        dtype="string",
    )
    validated_df = SchemaWithCategoryCoercion.validate(test_df)
    assert validated_df["col1"].dtype == pd.CategoricalDtype(
        categories=["category1", "category2"]
    )
    assert validated_df["col2"].dtype == pd.StringDtype()


def test_parser_with_add_missing_columns():

    class Schema(pa.DataFrameModel):
        """Schema."""

        a: str
        b: int
        c: int
        index: pa.typing.Index[int]

        class Config:
            """Schema config."""

            strict = False
            coerce = True
            add_missing_columns = True

        @pa.dataframe_parser
        @classmethod
        def preprocess(cls, df: pd.DataFrame) -> pd.DataFrame:
            """Preprocessing."""
            if "b" not in df.columns and "c" not in df.columns:
                raise pa.errors.SchemaError(
                    schema=cls,
                    data=df,
                    message=f"No `b` or `c` in {df.columns}",
                )

            if "b" not in df.columns:
                df["b"] = df["c"]
            if "c" not in df.columns:
                df["c"] = df["b"]
            return df

    validated_df = Schema.validate(pd.DataFrame({"a": ["xxx"], "b": 0}))
    assert_frame_equal(
        validated_df, pd.DataFrame({"a": ["xxx"], "b": 0, "c": 0})
    )

    with pytest.raises(pa.errors.SchemaError, match="No `b` or `c` in"):
        Schema.validate(pd.DataFrame({"a": ["xxx"]}))
