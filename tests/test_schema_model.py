"""Tests schema creation and validation from type annotations."""

from typing import Any, Optional

import pandas as pd
import pytest

import pandera as pa
from pandera.errors import SchemaError, SchemaErrors, SchemaInitError
from pandera.schema_model import (
    DataFrame,
    Field,
    Index,
    SchemaModel,
    Series,
    validator,
    dataframe_validator,
    dataframe_transformer,
    BaseConfig,
)


def test_schemamodel_to_dataframeschema():
    """Tests that a SchemaModel.to_schema() can produce the correct schema."""

    class Schema(SchemaModel):
        a: Series["int"]
        b: Series["string"]
        idx: Index["string"]

    expected = pa.DataFrameSchema(
        columns={"a": pa.Column(pa.Int64), "b": pa.Column("string")},
        index=pa.Index("string"),
    )

    assert expected == Schema.to_schema()


def test_check_types():
    class A(SchemaModel):
        a: Series["int"]

    @pa.check_types
    def transform(df: DataFrame[A]) -> DataFrame[A]:
        return df

    df = pd.DataFrame({"a": [1]})
    try:
        transform(df)
    except Exception as e:
        pytest.fail(f"Unexpected Exception {e}")


def test_check_types_errors():
    class A(SchemaModel):
        a: Series["int"]
        idx: Index["string"]

    class B(SchemaModel):
        b: Series["int"]

    df = pd.DataFrame({"a": [1]}, index=["1"])

    @pa.check_types
    def transform_index(df) -> DataFrame[A]:
        return df.reset_index(drop=True)

    with pytest.raises(SchemaError):
        transform_index(df)

    @pa.check_types
    def to_b(df: DataFrame[A]) -> DataFrame[B]:
        return df

    with pytest.raises(SchemaError, match="column 'b' not in dataframe"):
        to_b(df)

    @pa.check_types
    def to_str(df: DataFrame[A]) -> DataFrame[A]:
        df["a"] = "1"
        return df

    with pytest.raises(SchemaError, match="expected series 'a' to have type int64"):
        to_str(df)


def test_optional_column():
    class Schema(SchemaModel):
        a: Optional[Series["string"]]
        b: Optional[Series["string"]] = Field(eq="b")

    schema = Schema.to_schema()
    assert not schema.columns["a"].required
    assert not schema.columns["b"].required


def test_optional_index():
    class Schema(SchemaModel):
        idx: Optional[Index["string"]]

    with pytest.raises(SchemaInitError, match="Index 'idx' cannot be Optional."):
        Schema.to_schema()


def test_schemamodel_with_fields():
    """Tests that a SchemaModel.to_schema() can produce the correct schema."""

    class Schema(SchemaModel):
        a: Series["int"] = Field(eq=9, neq=0)
        b: Series["string"]
        idx: Index["string"] = Field(str_length={"min_value": 1})

    actual = Schema.to_schema()
    expected = pa.DataFrameSchema(
        columns={
            "a": pa.Column(
                "int", checks=[pa.Check.equal_to(9), pa.Check.not_equal_to(0)]
            ),
            "b": pa.Column("string"),
        },
        index=pa.Index("string", pa.Check.str_length(1)),
    )

    assert actual == expected


def test_field_to_column():
    for flag in ["nullable", "allow_duplicates", "coerce", "regex"]:
        for value in [True, False]:
            col = Field(**{flag: value}).to_column(pa.DateTime, required=value)
            assert isinstance(col, pa.Column)
            assert col.dtype == pa.DateTime.value
            assert col.properties[flag] == value
            assert col.required == value


def test_field_to_index():
    for flag in ["nullable", "allow_duplicates"]:
        for value in [True, False]:
            index = Field(**{flag: value}).to_index(pa.DateTime)
            assert isinstance(index, pa.Index)
            assert index.dtype == pa.DateTime.value
            assert getattr(index, flag) == value


def test_field_no_checks():
    assert not Field().to_column("string").checks


@pytest.mark.parametrize(
    "arg,value,expected",
    [
        ("eq", 9, pa.Check.equal_to(9)),
        ("neq", 9, pa.Check.not_equal_to(9)),
        ("gt", 9, pa.Check.greater_than(9)),
        ("ge", 9, pa.Check.greater_than_or_equal_to(9)),
        ("lt", 9, pa.Check.less_than(9)),
        ("le", 9, pa.Check.less_than_or_equal_to(9)),
        ("in_range", {"min_value": 1, "max_value": 9}, pa.Check.in_range(1, 9)),
        ("isin", [9, "a"], pa.Check.isin([9, "a"])),
        ("notin", [9, "a"], pa.Check.notin([9, "a"])),
        ("str_contains", "a", pa.Check.str_contains("a")),
        ("str_endswith", "a", pa.Check.str_endswith("a")),
        ("str_matches", "a", pa.Check.str_matches("a")),
        ("str_length", {"min_value": 1, "max_value": 9}, pa.Check.str_length(1, 9)),
        ("str_startswith", "a", pa.Check.str_startswith("a")),
    ],
)
def test_field_checks(arg: str, value: Any, expected: pa.Check):
    checks = Field(**{arg: value}).to_column("string").checks
    assert len(checks) == 1
    assert checks[0] == expected


def test_multiindex():
    class Schema(SchemaModel):
        a: Index["int"]
        b: Index["string"]

    expected = pa.DataFrameSchema(
        index=pa.MultiIndex([pa.Index(pa.Int, name="a"), pa.Index(pa.String, name="b")])
    )
    assert expected == Schema.to_schema()


def test_validator_single_column():
    class Schema(SchemaModel):
        a: Series["int"]

        @validator("a")
        def int_column_lt_100(series: pd.Series) -> bool:
            return series < 100

    df = pd.DataFrame({"a": [101]})
    schema = Schema.to_schema()
    err_msg = r"Column\s*a\s*<Check int_column_lt_100>\s*\[101\]\s*1"
    with pytest.raises(SchemaErrors, match=err_msg):
        schema.validate(df, lazy=True)


def test_validator_single_index():
    class Schema(SchemaModel):
        a: Index["string"]

        @validator("a")
        def not_dog(idx: pd.Index) -> bool:
            return ~idx.str.contains("dog")

    df = pd.DataFrame(index=["cat", "dog"])
    schema = Schema.to_schema()
    err_msg = r"Index\s*<NA>\s*<Check not_dog>\s*\[dog\]\s*"
    with pytest.raises(SchemaErrors, match=err_msg):
        schema.validate(df, lazy=True)


def test_validator_and_check():
    class Schema(SchemaModel):
        a: Series["int"] = Field(eq=1)

        @validator("a")
        def int_column_lt_100(series: pd.Series) -> bool:
            return series < 100

    schema = Schema.to_schema()
    assert len(schema.columns["a"].checks) == 2


def test_validator_non_existing():
    class Schema(SchemaModel):
        a: Series["int"]

        @validator("nope")
        def int_column_lt_100(series: pd.Series) -> bool:
            return series < 100

    err_msg = "Validator int_column_lt_100 is assigned to a non-existing field 'nope'"
    with pytest.raises(SchemaInitError, match=err_msg):
        Schema.to_schema()


def test_multiple_validators():
    class Schema(SchemaModel):
        a: Series["int"]

        @validator("a")
        def int_column_lt_100(series: pd.Series) -> bool:
            return series < 100

        @validator("a")
        def int_column_gt_0(series: pd.Series) -> bool:
            return series > 0

    schema = Schema.to_schema()
    assert len(schema.columns["a"].checks) == 2

    df = pd.DataFrame({"a": [0]})
    err_msg = r"Column\s*a\s*<Check int_column_gt_0>\s*\[0\]\s*1"
    with pytest.raises(SchemaErrors, match=err_msg):
        schema.validate(df, lazy=True)

    df = pd.DataFrame({"a": [101]})
    err_msg = r"Column\s*a\s*<Check int_column_lt_100>\s*\[101\]\s*1"
    with pytest.raises(SchemaErrors, match=err_msg):
        schema.validate(df, lazy=True)


def test_validator_multiple_columns():
    class Schema(SchemaModel):
        a: Series["int"]
        b: Series["int"]

        @validator("a", "b")
        def int_column_lt_100(series: pd.Series) -> bool:
            return series < 100

    df = pd.DataFrame({"a": [101], "b": [200]})
    schema = Schema.to_schema()
    with pytest.raises(SchemaErrors, match="2 schema errors were found"):
        schema.validate(df, lazy=True)


def test_validator_regex():
    class Schema(SchemaModel):
        a: Series["int"]
        abc: Series["int"]
        cba: Series["int"]

        @validator("^a", regex=True)
        def int_column_lt_100(series: pd.Series) -> bool:
            return series < 100

    df = pd.DataFrame({"a": [101], "abc": [1], "cba": [200]})
    schema = Schema.to_schema()
    with pytest.raises(SchemaErrors, match="1 schema errors were found"):
        schema.validate(df, lazy=True)


def test_inherit_schemamodel_fields():
    """Tests that a SchemaModel.to_schema() can produce the correct schema."""

    class A(SchemaModel):
        a: Series["int"]
        idx: Index["string"]

    class B(A):
        b: Series["string"]
        idx: Index["string"]

    class C(A):
        b: Series["int"]

    expected = pa.DataFrameSchema(
        columns={"a": pa.Column(pa.Int64), "b": pa.Column("int")},
        index=pa.Index("string"),
    )

    assert expected == C.to_schema()


def test_inherit_schemamodel_fields_checks():
    """Tests that a SchemaModel.to_schema() can produce the correct schema."""

    class A(SchemaModel):
        a: Series["int"]

        @validator("^a", regex=True)
        def int_column_lt_100(series: pd.Series) -> bool:
            return series < 100

    class B(A):
        b: Series["string"]
        idx: Index["string"]

        @validator("a")
        def int_column_lt_5(series: pd.Series) -> bool:
            return series < 5

    class C(B):
        b: Series["int"]
        abc: Series["int"]

        @validator("idx")
        def not_dog(idx: pd.Index) -> bool:
            return ~idx.str.contains("dog")

    schema = C.to_schema()
    print(f"{schema.columns['a'].checks=}")
    assert len(schema.columns["a"].checks) == 2
    assert len(schema.columns["abc"].checks) == 1
    assert len(schema.index.checks) == 1


def test_dataframe_validator():
    class A(SchemaModel):
        a: Series["int"]
        b: Series["int"]

        @dataframe_validator
        def value_lt_100(df: pd.DataFrame) -> bool:
            return df < 100

    class B(A):
        @dataframe_validator()
        def value_gt_0(df: pd.DataFrame) -> bool:
            return df > 0

    df = pd.DataFrame({"a": [101, 1], "b": [1, 0]})
    schema = B.to_schema()
    with pytest.raises(SchemaErrors, match="2 schema errors were found"):
        schema.validate(df, lazy=True)


def test_dataframe_transformer():
    class Schema(SchemaModel):
        a: Series["int"]

        @dataframe_transformer
        def neg_a(df: pd.DataFrame) -> pd.DataFrame:
            df["a"] = -df["a"]
            return df

    df = pd.DataFrame({"a": [1]})
    actual = Schema.to_schema().validate(df)
    expected = pd.DataFrame({"a": [-1]})
    pd.testing.assert_frame_equal(actual, expected)


def test_multiple_dataframe_transformers():
    class A(SchemaModel):
        a: Series["int"]

        @dataframe_transformer
        def neg_a(df: pd.DataFrame) -> pd.DataFrame:
            df["a"] = -df["a"]
            return df

    class B(A):
        b: Series["int"]

        @dataframe_transformer
        def neg_b(df: pd.DataFrame) -> pd.DataFrame:
            df["b"] = -df["b"]
            return df

    df = pd.DataFrame({"a": [1]})
    with pytest.raises(
        SchemaInitError, match="can only have one 'dataframe_transformer'"
    ):
        B.to_schema().validate(df)


def test_config():
    class A(SchemaModel):
        a: Series["int"]
        idx_1: Index["string"]
        idx_2: Index["string"]

        class Config:
            name = "A schema"
            coerce = True
            multiindex_coerce = True
            multiindex_strict = True
            multiindex_name: Optional[str] = "mi"

    class B(A):
        b: Series["int"]

        class Config:
            name = "B schema"
            strict = True

    expected = pa.DataFrameSchema(
        columns={"a": pa.Column(pa.Int), "b": pa.Column(pa.Int)},
        index=pa.MultiIndex(
            [pa.Index(pa.String, name="idx_1"), pa.Index(pa.String, name="idx_2")],
            coerce=True,
            strict=True,
            name="mi",
        ),
        name="B schema",
        coerce=True,
        strict=True,
    )

    assert expected == B.to_schema()
