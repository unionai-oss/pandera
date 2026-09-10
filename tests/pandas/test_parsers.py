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
from pandera.engines.pandas_engine import PANDAS_3_0_0_PLUS
from pandera.typing import Index, Series


def test_dataframe_schema_parse() -> None:
    """Test that DataFrameSchema-level Parses work properly."""
    data = pd.DataFrame([[1, 4, 9, 16, 25] for _ in range(10)])

    schema_check_return_bool = DataFrameSchema(
        parsers=Parser(lambda df: df.transform("sqrt"))
    )
    assert schema_check_return_bool.validate(data).equals(
        data.transform("sqrt")
    )


def test_dataframe_schema_parse_with_element_wise() -> None:
    """Test that DataFrameSchema-level Parses work properly."""
    data = pd.DataFrame([[1, 4, 9, 16, 25] for _ in range(10)])
    schema_check_return_bool = DataFrameSchema(
        parsers=Parser(np.sqrt, element_wise=True)
    )
    result = data.map(np.sqrt)
    assert schema_check_return_bool.validate(data).equals(result)


def test_series_schema_parse_with_element_wise() -> None:
    data = pd.Series([1, 4, 9, 16, 25])
    schema_check_return_bool = SeriesSchema(
        parsers=Parser(np.sqrt, element_wise=True)
    )
    # Series always has map, but for consistency with DataFrame test
    result = data.map(np.sqrt)
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


def test_dataframe_parser_parenthesized_form():
    """``@dataframe_parser()`` and ``@dataframe_parser(**kwargs)`` should work
    like the bare ``@dataframe_parser``, matching ``dataframe_check``."""
    data = pd.DataFrame({"col": [1.0, 4.0, 9.0]})
    expected = pd.DataFrame({"col": [2.0, 8.0, 18.0]})

    class DFModel(pa.DataFrameModel):
        col: float

        @pa.dataframe_parser()
        @classmethod
        def double(cls, df: pd.DataFrame) -> pd.DataFrame:
            return df * 2

    assert_frame_equal(DFModel.validate(data), expected)

    class DFModelWithKwargs(pa.DataFrameModel):
        col: float

        @pa.dataframe_parser(description="double the frame")
        @classmethod
        def double(cls, df: pd.DataFrame) -> pd.DataFrame:
            return df * 2

    assert_frame_equal(DFModelWithKwargs.validate(data), expected)


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
    if PANDAS_3_0_0_PLUS:
        import numpy as np

        assert validated_df["col2"].dtype == pd.StringDtype(na_value=np.nan)
    else:
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
    expected = pd.DataFrame({"a": ["xxx"], "b": 0, "c": 0})
    # With pandas < 3, str columns coerce to object (NpString); pandas >= 3 to str
    if PANDAS_3_0_0_PLUS:
        expected["a"] = expected["a"].astype("str")
    assert_frame_equal(validated_df, expected)

    with pytest.raises(pa.errors.SchemaError, match="No `b` or `c` in"):
        Schema.validate(pd.DataFrame({"a": ["xxx"]}))


def _clean_to_float(series: pd.Series) -> pd.Series:
    stripped = series.astype(str).str.strip()
    return pd.to_numeric(stripped, errors="coerce").astype(float)


@pytest.mark.parametrize(
    "schema_coerce,column_coerce",
    [(True, False), (False, True), (True, True)],
)
def test_column_parser_with_coercion(schema_coerce, column_coerce):
    """Column-level parsers should be applied before dtype coercion.

    Regression test for
    https://github.com/unionai-oss/pandera/issues/2047
    """
    df = pd.DataFrame({"col1": ["-0013123", "-0000012", "        "]})
    schema = DataFrameSchema(
        {
            "col1": pa.Column(
                float,
                parsers=Parser(_clean_to_float),
                nullable=True,
                coerce=column_coerce,
            )
        },
        coerce=schema_coerce,
    )
    validated = schema.validate(df)
    expected = pd.Series(
        [-13123.0, -12.0, np.nan], name="col1", dtype="float64"
    )
    pd.testing.assert_series_equal(validated["col1"], expected)


@pytest.mark.parametrize(
    "schema_coerce,column_coerce",
    [(True, False), (False, True), (True, True)],
)
def test_column_parser_output_needs_coercion(schema_coerce, column_coerce):
    """Coercion should still be applied to the output of column-level
    parsers whose output does not yet have the expected dtype."""
    df = pd.DataFrame({"col1": ["1,5", "2,5"]})
    schema = DataFrameSchema(
        {
            "col1": pa.Column(
                float,
                parsers=Parser(lambda s: s.str.replace(",", ".")),
                coerce=column_coerce,
            )
        },
        coerce=schema_coerce,
    )
    validated = schema.validate(df)
    expected = pd.Series([1.5, 2.5], name="col1", dtype="float64")
    pd.testing.assert_series_equal(validated["col1"], expected)


def test_column_parser_with_inferred_schema_coercion():
    """Updating an inferred schema with a parser column should validate the
    same way as a manually defined schema (issue #2047)."""
    df = pd.DataFrame({"col1": ["-0013123", "-0000012", "        "]})
    schema = pa.infer_schema(df).update_columns(
        {
            "col1": {
                "dtype": float,
                "parsers": Parser(_clean_to_float),
                "nullable": True,
            }
        }
    )
    validated = schema.validate(df)
    expected = pd.Series(
        [-13123.0, -12.0, np.nan], name="col1", dtype="float64"
    )
    pd.testing.assert_series_equal(validated["col1"], expected)


def test_parser_on_dataframe_model_index_field():
    """``@pa.parser`` should be applied to fields annotated as Index
    (issue #1684)."""

    class Model(pa.DataFrameModel):
        idx: Index[int]
        col: Series[int]

        class Config:
            coerce = True

        @pa.parser("idx")
        @classmethod
        def double(cls, series: pd.Series) -> pd.Series:
            return series * 2

        @pa.parser("col")
        @classmethod
        def triple(cls, series: pd.Series) -> pd.Series:
            return series * 3

    df = pd.DataFrame({"col": [1, 2, 3]}, index=pd.Index([1, 2, 3]))
    validated = Model.validate(df)
    assert validated.index.tolist() == [2, 4, 6]
    assert validated["col"].tolist() == [3, 6, 9]


def test_index_parser():
    """Parsers on an Index schema component should transform the index."""
    schema = DataFrameSchema(
        columns={"col": pa.Column(int)},
        index=pa.Index(int, parsers=Parser(lambda s: s * 2), name="idx"),
    )
    df = pd.DataFrame(
        {"col": [1, 2, 3]}, index=pd.Index([1, 2, 3], name="idx")
    )
    validated = schema.validate(df)
    assert validated.index.tolist() == [2, 4, 6]
    assert validated.index.name == "idx"


def test_index_parser_output_needs_coercion():
    """Index parsers should run before dtype coercion."""
    schema = DataFrameSchema(
        columns={"col": pa.Column(int)},
        index=pa.Index(
            float,
            parsers=Parser(lambda s: s.str.replace(",", ".")),
            coerce=True,
        ),
    )
    df = pd.DataFrame({"col": [1, 2]}, index=pd.Index(["1,5", "2,5"]))
    validated = schema.validate(df)
    assert validated.index.tolist() == [1.5, 2.5]
