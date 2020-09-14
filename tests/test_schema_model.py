"""Tests schema creation and validation from type annotations."""
from typing import Any, Optional

import pandas as pd
import pytest

import pandera as pa
from pandera.typing import Index, Series


def test_simple_to_schema():
    """Tests that a SchemaModel.to_schema() can produce the correct schema."""

    class Schema(pa.SchemaModel):
        a: Series[int]
        b: Series[str]
        idx: Index[str]

    expected = pa.DataFrameSchema(
        columns={"a": pa.Column(int), "b": pa.Column(str)},
        index=pa.Index(str),
    )

    assert expected == Schema.to_schema()


def test_invalid_annotations():
    """Tests that a SchemaModel.to_schema() can produce the correct schema."""

    class IntSchema(pa.SchemaModel):
        a: int

    with pytest.raises(pa.errors.SchemaInitError, match="Invalid annotation"):
        IntSchema.to_schema()

    from decimal import Decimal

    class InvalidDtypeSchema(pa.SchemaModel):
        d: Series[Decimal]

    with pytest.raises(TypeError, match="python type '<class 'decimal.Decimal'>"):
        InvalidDtypeSchema.to_schema()


def test_optional_column():
    class Schema(pa.SchemaModel):
        a: Optional[Series[str]]
        b: Optional[Series[str]] = pa.Field(eq="b")

    schema = Schema.to_schema()
    assert not schema.columns["a"].required
    assert not schema.columns["b"].required


def test_optional_index():
    class Schema(pa.SchemaModel):
        idx: Optional[Index[str]]

    with pytest.raises(
        pa.errors.SchemaInitError, match="Index 'idx' cannot be Optional."
    ):
        Schema.to_schema()


def test_schemamodel_with_fields():
    """Tests that a SchemaModel.to_schema() can produce the correct schema."""

    class Schema(pa.SchemaModel):
        a: Series[int] = pa.Field(eq=9, neq=0)
        b: Series[str]
        idx: Index[str] = pa.Field(str_length={"min_value": 1})

    actual = Schema.to_schema()
    expected = pa.DataFrameSchema(
        columns={
            "a": pa.Column(
                int, checks=[pa.Check.equal_to(9), pa.Check.not_equal_to(0)]
            ),
            "b": pa.Column(str),
        },
        index=pa.Index(str, pa.Check.str_length(1)),
    )

    assert actual == expected


def test_field_to_column():
    for flag in ["nullable", "allow_duplicates", "coerce", "regex"]:
        for value in [True, False]:
            col = pa.Field(**{flag: value}).to_column(pa.DateTime, required=value)
            assert isinstance(col, pa.Column)
            assert col.dtype == pa.DateTime.value
            assert col.properties[flag] == value
            assert col.required == value


def test_field_to_index():
    for flag in ["nullable", "allow_duplicates"]:
        for value in [True, False]:
            index = pa.Field(**{flag: value}).to_index(pa.DateTime)
            assert isinstance(index, pa.Index)
            assert index.dtype == pa.DateTime.value
            assert getattr(index, flag) == value


def test_field_no_checks():
    assert not pa.Field().to_column(str).checks


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
    checks = pa.Field(**{arg: value}).to_column(str).checks
    assert len(checks) == 1
    assert checks[0] == expected


def test_multiindex():
    class Schema(pa.SchemaModel):
        a: Index[int]
        b: Index[str]

    expected = pa.DataFrameSchema(
        index=pa.MultiIndex([pa.Index(int, name="a"), pa.Index(str, name="b")])
    )
    assert expected == Schema.to_schema()


def test_validator_single_column():
    class Schema(pa.SchemaModel):
        a: Series[int]

        @pa.validator("a")
        def int_column_lt_100(series: pd.Series) -> bool:
            return series < 100

    df = pd.DataFrame({"a": [101]})
    schema = Schema.to_schema()
    err_msg = r"Column\s*a\s*<Check int_column_lt_100>\s*\[101\]\s*1"
    with pytest.raises(pa.errors.SchemaErrors, match=err_msg):
        schema.validate(df, lazy=True)


def test_validator_single_index():
    class Schema(pa.SchemaModel):
        a: Index[str]

        @pa.validator("a")
        def not_dog(idx: pd.Index) -> bool:
            return ~idx.str.contains("dog")

    df = pd.DataFrame(index=["cat", "dog"])
    schema = Schema.to_schema()
    err_msg = r"Index\s*<NA>\s*<Check not_dog>\s*\[dog\]\s*"
    with pytest.raises(pa.errors.SchemaErrors, match=err_msg):
        schema.validate(df, lazy=True)


def test_validator_and_check():
    class Schema(pa.SchemaModel):
        a: Series[int] = pa.Field(eq=1)

        @pa.validator("a")
        def int_column_lt_100(series: pd.Series) -> bool:
            return series < 100

    schema = Schema.to_schema()
    assert len(schema.columns["a"].checks) == 2


def test_validator_non_existing():
    class Schema(pa.SchemaModel):
        a: Series[int]

        @pa.validator("nope")
        def int_column_lt_100(series: pd.Series) -> bool:
            return series < 100

    err_msg = "Validator int_column_lt_100 is assigned to a non-existing field 'nope'"
    with pytest.raises(pa.errors.SchemaInitError, match=err_msg):
        Schema.to_schema()


def test_multiple_validators():
    class Schema(pa.SchemaModel):
        a: Series[int]

        @pa.validator("a")
        def int_column_lt_100(series: pd.Series) -> bool:
            return series < 100

        @pa.validator("a")
        def int_column_gt_0(series: pd.Series) -> bool:
            return series > 0

    schema = Schema.to_schema()
    assert len(schema.columns["a"].checks) == 2

    df = pd.DataFrame({"a": [0]})
    err_msg = r"Column\s*a\s*<Check int_column_gt_0>\s*\[0\]\s*1"
    with pytest.raises(pa.errors.SchemaErrors, match=err_msg):
        schema.validate(df, lazy=True)

    df = pd.DataFrame({"a": [101]})
    err_msg = r"Column\s*a\s*<Check int_column_lt_100>\s*\[101\]\s*1"
    with pytest.raises(pa.errors.SchemaErrors, match=err_msg):
        schema.validate(df, lazy=True)


def test_validator_multiple_columns():
    class Schema(pa.SchemaModel):
        a: Series[int]
        b: Series[int]

        @pa.validator("a", "b")
        def int_column_lt_100(series: pd.Series) -> bool:
            return series < 100

    df = pd.DataFrame({"a": [101], "b": [200]})
    schema = Schema.to_schema()
    with pytest.raises(pa.errors.SchemaErrors, match="2 schema errors were found"):
        schema.validate(df, lazy=True)


def test_validator_regex():
    class Schema(pa.SchemaModel):
        a: Series[int]
        abc: Series[int]
        cba: Series[int]

        @pa.validator("^a", regex=True)
        def int_column_lt_100(series: pd.Series) -> bool:
            return series < 100

    df = pd.DataFrame({"a": [101], "abc": [1], "cba": [200]})
    schema = Schema.to_schema()
    with pytest.raises(pa.errors.SchemaErrors, match="1 schema errors were found"):
        schema.validate(df, lazy=True)


def test_inherit_schemamodel_fields():
    """Tests that a SchemaModel.to_schema() can produce the correct schema."""

    class A(pa.SchemaModel):
        a: Series[int]
        idx: Index[str]

    class B(A):
        b: Series[str]
        idx: Index[str]

    class C(A):
        b: Series[int]

    expected = pa.DataFrameSchema(
        columns={"a": pa.Column(int), "b": pa.Column(int)},
        index=pa.Index(str),
    )

    assert expected == C.to_schema()


def test_inherit_schemamodel_fields_checks():
    """Tests that a SchemaModel.to_schema() can produce the correct schema."""

    class A(pa.SchemaModel):
        a: Series[int]

        @pa.validator("^a", regex=True)
        def int_column_lt_100(series: pd.Series) -> bool:
            return series < 100

    class B(A):
        b: Series[str]
        idx: Index[str]

        @pa.validator("a")
        def int_column_lt_5(series: pd.Series) -> bool:
            return series < 5

    class C(B):
        b: Series[int]
        abc: Series[int]

        @pa.validator("idx")
        def not_dog(idx: pd.Index) -> bool:
            return ~idx.str.contains("dog")

    schema = C.to_schema()
    assert len(schema.columns["a"].checks) == 2
    assert len(schema.columns["abc"].checks) == 1
    assert len(schema.index.checks) == 1


def test_dataframe_validator():
    class A(pa.SchemaModel):
        a: Series[int]
        b: Series[int]

        @pa.dataframe_validator
        def value_lt_100(df: pd.DataFrame) -> bool:
            return df < 100

    class B(A):
        @pa.dataframe_validator()
        def value_gt_0(df: pd.DataFrame) -> bool:
            return df > 0

    df = pd.DataFrame({"a": [101, 1], "b": [1, 0]})
    schema = B.to_schema()
    with pytest.raises(pa.errors.SchemaErrors, match="2 schema errors were found"):
        schema.validate(df, lazy=True)


def test_dataframe_transformer():
    class Schema(pa.SchemaModel):
        a: Series[int]

        @pa.dataframe_transformer
        def neg_a(df: pd.DataFrame) -> pd.DataFrame:
            df["a"] = -df["a"]
            return df

    df = pd.DataFrame({"a": [1]})
    actual = Schema.to_schema().validate(df)
    expected = pd.DataFrame({"a": [-1]})
    pd.testing.assert_frame_equal(actual, expected)


def test_multiple_dataframe_transformers():
    class A(pa.SchemaModel):
        a: Series[int]

        @pa.dataframe_transformer
        def neg_a(df: pd.DataFrame) -> pd.DataFrame:
            df["a"] = -df["a"]
            return df

    class B(A):
        b: Series[int]

        @pa.dataframe_transformer
        def neg_b(df: pd.DataFrame) -> pd.DataFrame:
            df["b"] = -df["b"]
            return df

    df = pd.DataFrame({"a": [1]})
    with pytest.raises(
        pa.errors.SchemaInitError, match="can only have one 'dataframe_transformer'"
    ):
        B.to_schema().validate(df)


def test_config():
    class A(pa.SchemaModel):
        a: Series[int]
        idx_1: Index[str]
        idx_2: Index[str]

        class Config:
            name = "A schema"
            coerce = True
            multiindex_coerce = True
            multiindex_strict = True
            multiindex_name: Optional[str] = "mi"

    class B(A):
        b: Series[int]

        class Config:
            name = "B schema"
            strict = True

    expected = pa.DataFrameSchema(
        columns={"a": pa.Column(int), "b": pa.Column(int)},
        index=pa.MultiIndex(
            [pa.Index(str, name="idx_1"), pa.Index(str, name="idx_2")],
            coerce=True,
            strict=True,
            name="mi",
        ),
        name="B schema",
        coerce=True,
        strict=True,
    )

    assert expected == B.to_schema()
