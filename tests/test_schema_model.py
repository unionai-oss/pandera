"""Tests schema creation and validation from type annotations."""
# pylint:disable=R0903,C0115,C0116

from typing import Any, Optional

import pandas as pd
import pytest

import pandera as pa
from pandera.typing import Index, Series


def test_simple_to_schema():
    """Test that SchemaModel.to_schema() can produce the correct schema."""

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
    """Test that SchemaModel.to_schema() fails if annotations or types are not
    recognized.
    """

    class IntSchema(pa.SchemaModel):
        a: int

    with pytest.raises(pa.errors.SchemaInitError, match="Invalid annotation"):
        IntSchema.to_schema()

    from decimal import Decimal # pylint:disable=C0415

    class InvalidDtypeSchema(pa.SchemaModel):
        d: Series[Decimal] # type: ignore

    with pytest.raises(TypeError, match="python type '<class 'decimal.Decimal'>"):
        InvalidDtypeSchema.to_schema()


def test_optional_column():
    """Test that optional columns are not required."""

    class Schema(pa.SchemaModel):
        a: Optional[Series[str]]
        b: Optional[Series[str]] = pa.Field(eq="b")

    schema = Schema.to_schema()
    assert not schema.columns["a"].required
    assert not schema.columns["b"].required


def test_optional_index():
    """Test that optional indices are not required."""

    class Schema(pa.SchemaModel):
        idx: Optional[Index[str]]

    with pytest.raises(
        pa.errors.SchemaInitError, match="Index 'idx' cannot be Optional."
    ):
        Schema.to_schema()


def test_schemamodel_with_fields():
    """Test that Fields are translated in the schema."""

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
    """Test that Field outputs the correct column options."""
    for flag in ["nullable", "allow_duplicates", "coerce", "regex"]:
        for value in [True, False]:
            col = pa.Field(**{flag: value}).to_column(pa.DateTime, required=value)
            assert isinstance(col, pa.Column)
            assert col.dtype == pa.DateTime.value
            assert col.properties[flag] == value
            assert col.required == value


def test_field_to_index():
    """Test that Field outputs the correct index options."""
    for flag in ["nullable", "allow_duplicates"]:
        for value in [True, False]:
            index = pa.Field(**{flag: value}).to_index(pa.DateTime)
            assert isinstance(index, pa.Index)
            assert index.dtype == pa.DateTime.value
            assert getattr(index, flag) == value


def test_field_no_checks():
    """Test Field without checks."""
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
    """Test that all built-in checks are available in a Field."""
    checks = pa.Field(**{arg: value}).to_column(str).checks
    assert len(checks) == 1
    assert checks[0] == expected


def test_multiindex():
    """Test that multiple Index annotations create a MultiIndex."""

    class Schema(pa.SchemaModel):
        a: Index[int]
        b: Index[str]

    expected = pa.DataFrameSchema(
        index=pa.MultiIndex([pa.Index(int, name="a"), pa.Index(str, name="b")])
    )
    assert expected == Schema.to_schema()


def test_check_single_column():
    """Test the behaviour of a check on a single column."""

    class Schema(pa.SchemaModel):
        a: Series[int]

        @pa.check("a")
        def int_column_lt_100(series: pd.Series) -> bool:
            return series < 100

    df = pd.DataFrame({"a": [101]})
    schema = Schema.to_schema()
    err_msg = r"Column\s*a\s*int_column_lt_100\s*\[101\]\s*1"
    with pytest.raises(pa.errors.SchemaErrors, match=err_msg):
        schema.validate(df, lazy=True)


def test_check_single_index():
    """Test the behaviour of a check on a single index."""

    class Schema(pa.SchemaModel):
        a: Index[str]

        @pa.check("a")
        def not_dog(idx: pd.Index) -> bool:
            return ~idx.str.contains("dog")

    df = pd.DataFrame(index=["cat", "dog"])
    schema = Schema.to_schema()
    err_msg = r"Index\s*<NA>\s*not_dog\s*\[dog\]\s*"
    with pytest.raises(pa.errors.SchemaErrors, match=err_msg):
        schema.validate(df, lazy=True)


def test_field_and_check():
    """Test the combination of a field and a check on the same column."""

    class Schema(pa.SchemaModel):
        a: Series[int] = pa.Field(eq=1)

        @pa.check("a")
        def int_column_lt_100(series: pd.Series) -> bool:
            return series < 100

    schema = Schema.to_schema()
    assert len(schema.columns["a"].checks) == 2


def test_check_non_existing():
    """Test a check on a non-existing column."""

    class Schema(pa.SchemaModel):
        a: Series[int]

        @pa.check("nope")
        def int_column_lt_100(series: pd.Series) -> bool:
            return series < 100

    err_msg = "Validator int_column_lt_100 is assigned to a non-existing field 'nope'"
    with pytest.raises(pa.errors.SchemaInitError, match=err_msg):
        Schema.to_schema()


def test_multiple_checks():
    """Test multiple checks on the same column."""

    class Schema(pa.SchemaModel):
        a: Series[int]

        @pa.check("a")
        def int_column_lt_100(series: pd.Series) -> bool:
            return series < 100

        @pa.check("a")
        def int_column_gt_0(series: pd.Series) -> bool:
            return series > 0

    schema = Schema.to_schema()
    assert len(schema.columns["a"].checks) == 2

    df = pd.DataFrame({"a": [0]})
    err_msg = r"Column\s*a\s*int_column_gt_0\s*\[0\]\s*1"
    with pytest.raises(pa.errors.SchemaErrors, match=err_msg):
        schema.validate(df, lazy=True)

    df = pd.DataFrame({"a": [101]})
    err_msg = r"Column\s*a\s*int_column_lt_100\s*\[101\]\s*1"
    with pytest.raises(pa.errors.SchemaErrors, match=err_msg):
        schema.validate(df, lazy=True)


def test_check_multiple_columns():
    """Test a single check decorator targeting multiple columns."""

    class Schema(pa.SchemaModel):
        a: Series[int]
        b: Series[int]

        @pa.check("a", "b")
        def int_column_lt_100(series: pd.Series) -> bool:
            return series < 100

    df = pd.DataFrame({"a": [101], "b": [200]})
    schema = Schema.to_schema()
    with pytest.raises(pa.errors.SchemaErrors, match="2 schema errors were found"):
        schema.validate(df, lazy=True)


def test_check_regex():
    """Test the regex argument of the check decorator."""

    class Schema(pa.SchemaModel):
        a: Series[int]
        abc: Series[int]
        cba: Series[int]

        @pa.check("^a", regex=True)
        def int_column_lt_100(series: pd.Series) -> bool:
            return series < 100

    df = pd.DataFrame({"a": [101], "abc": [1], "cba": [200]})
    schema = Schema.to_schema()
    with pytest.raises(pa.errors.SchemaErrors, match="1 schema errors were found"):
        schema.validate(df, lazy=True)


def test_inherit_schemamodel_fields():
    """Test that columns and indices are inherited."""

    class Base(pa.SchemaModel):
        a: Series[int]
        idx: Index[str]

    class Mid(Base):
        b: Series[str]
        idx: Index[str]

    class Child(Mid):
        b: Series[int]

    expected = pa.DataFrameSchema(
        columns={"a": pa.Column(int), "b": pa.Column(int)},
        index=pa.Index(str),
    )

    assert expected == Child.to_schema()


def test_inherit_schemamodel_checks():
    """Test that checks are inherited."""

    class Base(pa.SchemaModel):
        a: Series[int]

        @pa.check("^a", regex=True)
        def int_column_lt_100(series: pd.Series) -> bool:
            return series < 100

    class Mid(Base):
        b: Series[str]
        idx: Index[str]

        @pa.check("a")
        def int_column_lt_5(series: pd.Series) -> bool:
            return series < 5

    class Child(Mid):
        b: Series[int]
        abc: Series[int]

        @pa.check("idx")
        def not_dog(idx: pd.Index) -> bool:
            return ~idx.str.contains("dog")

    schema = Child.to_schema()
    assert len(schema.columns["a"].checks) == 2
    assert len(schema.columns["abc"].checks) == 1
    assert len(schema.index.checks) == 1


def test_dateframe_check():
    class Base(pa.SchemaModel):
        a: Series[int]
        b: Series[int]

        @pa.dateframe_check
        def value_lt_100(df: pd.DataFrame) -> bool:
            return df < 100

    class Child(Base):
        @pa.dateframe_check()
        def value_gt_0(df: pd.DataFrame) -> bool:
            return df > 0

    df = pd.DataFrame({"a": [101, 1], "b": [1, 0]})
    schema = Child.to_schema()
    with pytest.raises(pa.errors.SchemaErrors, match="2 schema errors were found"):
        schema.validate(df, lazy=True)


def test_config():
    """Test that Config can be inherited and translate into DataFramSchema options."""

    class Base(pa.SchemaModel):
        a: Series[int]
        idx_1: Index[str]
        idx_2: Index[str]

        class Config:
            name = "A schema"
            coerce = True
            multiindex_coerce = True
            multiindex_strict = True
            multiindex_name: Optional[str] = "mi"

    class Child(Base):
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
        name="Child schema",
        coerce=True,
        strict=True,
    )

    assert expected == Child.to_schema()
