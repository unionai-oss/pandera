"""Tests the way Columns are Parsed"""

import copy

import numpy as np
import pandas as pd
import pytest

import pandera as pa
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
