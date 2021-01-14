"""Tests schema creation and validation from type annotations."""
# pylint:disable=missing-class-docstring,missing-function-docstring,too-few-public-methods
import re
from typing import Iterable, Optional

import pandas as pd
import pytest

import pandera as pa
from pandera.typing import DataFrame, Index, Series, String


def test_to_schema():
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

    with pytest.raises(TypeError):
        Schema()


def test_invalid_annotations():
    """Test that SchemaModel.to_schema() fails if annotations or types are not
    recognized.
    """

    class Missing(pa.SchemaModel):
        a = pa.Field()
        b: Series[int]
        c = pa.Field()
        _d = 0

    err_msg = re.escape("Found missing annotations: ['a', 'c']")
    with pytest.raises(pa.errors.SchemaInitError, match=err_msg):
        Missing.to_schema()

    class Invalid(pa.SchemaModel):
        a: int

    with pytest.raises(pa.errors.SchemaInitError, match="Invalid annotation"):
        Invalid.to_schema()

    from decimal import Decimal  # pylint:disable=C0415

    class InvalidDtype(pa.SchemaModel):
        d: Series[Decimal]  # type: ignore

    with pytest.raises(
        TypeError, match="python type '<class 'decimal.Decimal'>"
    ):
        InvalidDtype.to_schema()


def test_optional_column():
    """Test that optional columns are not required."""

    class Schema(pa.SchemaModel):
        a: Optional[Series[str]]
        b: Optional[Series[str]] = pa.Field(eq="b")
        c: Optional[Series[String]]  # test pandera.typing alias

    schema = Schema.to_schema()
    assert not schema.columns["a"].required
    assert not schema.columns["b"].required
    assert not schema.columns["c"].required


def test_optional_index():
    """Test that optional indices are not required."""

    class Schema(pa.SchemaModel):
        idx: Optional[Index[str]]

    class SchemaWithAliasDtype(pa.SchemaModel):
        idx: Optional[Index[String]]  # test pandera.typing alias

    for model in (Schema, SchemaWithAliasDtype):
        with pytest.raises(
            pa.errors.SchemaInitError, match="Index 'idx' cannot be Optional."
        ):
            model.to_schema()


def test_schemamodel_with_fields():
    """Test that Fields are translated in the schema."""

    class Schema(pa.SchemaModel):
        a: Series[int] = pa.Field(eq=9, ne=0)
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


def test_invalid_field():
    class Schema(pa.SchemaModel):
        a: Series[int] = 0

    with pytest.raises(
        pa.errors.SchemaInitError, match="'a' can only be assigned a 'Field'"
    ):
        Schema.to_schema()


def test_multiindex():
    """Test that multiple Index annotations create a MultiIndex."""

    class Schema(pa.SchemaModel):
        a: Index[int] = pa.Field(gt=0)
        b: Index[str]

    expected = pa.DataFrameSchema(
        index=pa.MultiIndex(
            [
                pa.Index(int, name="a", checks=pa.Check.gt(0)),
                pa.Index(str, name="b"),
            ]
        )
    )
    assert expected == Schema.to_schema()


def test_column_check_name():
    """Test that column name is mandatory."""

    class Schema(pa.SchemaModel):
        a: Series[int] = pa.Field(check_name=False)

    with pytest.raises(pa.errors.SchemaInitError):
        Schema.to_schema()


def test_single_index_check_name():
    """Test single index name."""
    df = pd.DataFrame(index=pd.Index(["cat", "dog"], name="animal"))

    class DefaultSchema(pa.SchemaModel):
        a: Index[str]

    assert isinstance(DefaultSchema.validate(df), pd.DataFrame)

    class DefaultFieldSchema(pa.SchemaModel):
        a: Index[str] = pa.Field(check_name=None)

    assert isinstance(DefaultFieldSchema.validate(df), pd.DataFrame)

    class NotCheckNameSchema(pa.SchemaModel):
        a: Index[str] = pa.Field(check_name=False)

    assert isinstance(NotCheckNameSchema.validate(df), pd.DataFrame)

    class SchemaNamedIndex(pa.SchemaModel):
        a: Index[str] = pa.Field(check_name=True)

    err_msg = "name 'a', found 'animal'"
    with pytest.raises(pa.errors.SchemaError, match=err_msg):
        SchemaNamedIndex.validate(df)


def test_multiindex_check_name():
    """Test a MultiIndex name."""

    df = pd.DataFrame(
        index=pd.MultiIndex.from_arrays(
            [["foo", "bar"], [0, 1]], names=["a", "b"]
        )
    )

    class DefaultSchema(pa.SchemaModel):
        a: Index[str]
        b: Index[int]

    assert isinstance(DefaultSchema.validate(df), pd.DataFrame)

    class CheckNameSchema(pa.SchemaModel):
        a: Index[str] = pa.Field(check_name=True)
        b: Index[int] = pa.Field(check_name=True)

    assert isinstance(CheckNameSchema.validate(df), pd.DataFrame)

    class NotCheckNameSchema(pa.SchemaModel):
        a: Index[str] = pa.Field(check_name=False)
        b: Index[int] = pa.Field(check_name=False)

    df = pd.DataFrame(
        index=pd.MultiIndex.from_arrays([["foo", "bar"], [0, 1]])
    )
    assert isinstance(NotCheckNameSchema.validate(df), pd.DataFrame)


def test_check_validate_method():
    """Test validate method on valid data."""

    class Schema(pa.SchemaModel):
        a: Series[int]

        @pa.check("a")
        def int_column_lt_100(cls, series: pd.Series) -> Iterable[bool]:
            # pylint:disable=no-self-argument
            assert cls is Schema
            return series < 100

    df = pd.DataFrame({"a": [99]})
    assert isinstance(Schema.validate(df, lazy=True), pd.DataFrame)


def test_check_single_column():
    """Test the behaviour of a check on a single column."""

    class Schema(pa.SchemaModel):
        a: Series[int]

        @pa.check("a")
        def int_column_lt_100(cls, series: pd.Series) -> Iterable[bool]:
            # pylint:disable=no-self-argument
            assert cls is Schema
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
        def not_dog(cls, idx: pd.Index) -> Iterable[bool]:
            # pylint:disable=no-self-argument
            assert cls is Schema
            return ~idx.str.contains("dog")

    df = pd.DataFrame(index=["cat", "dog"])
    err_msg = r"Index\s*<NA>\s*not_dog\s*\[dog\]\s*"
    with pytest.raises(pa.errors.SchemaErrors, match=err_msg):
        Schema.validate(df, lazy=True)


def test_field_and_check():
    """Test the combination of a field and a check on the same column."""

    class Schema(pa.SchemaModel):
        a: Series[int] = pa.Field(eq=1)

        @pa.check("a")
        @classmethod
        def int_column_lt_100(cls, series: pd.Series) -> Iterable[bool]:
            return series < 100

    schema = Schema.to_schema()
    assert len(schema.columns["a"].checks) == 2


def test_check_non_existing():
    """Test a check on a non-existing column."""

    class Schema(pa.SchemaModel):
        a: Series[int]

        @pa.check("nope")
        @classmethod
        def int_column_lt_100(cls, series: pd.Series) -> Iterable[bool]:
            return series < 100

    err_msg = (
        "Check int_column_lt_100 is assigned to a non-existing field 'nope'"
    )
    with pytest.raises(pa.errors.SchemaInitError, match=err_msg):
        Schema.to_schema()


def test_multiple_checks():
    """Test multiple checks on the same column."""

    class Schema(pa.SchemaModel):
        a: Series[int]

        @pa.check("a")
        @classmethod
        def int_column_lt_100(cls, series: pd.Series) -> Iterable[bool]:
            return series < 100

        @pa.check("a")
        @classmethod
        def int_column_gt_0(cls, series: pd.Series) -> Iterable[bool]:
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
        @classmethod
        def int_column_lt_100(cls, series: pd.Series) -> Iterable[bool]:
            return series < 100

    df = pd.DataFrame({"a": [101], "b": [200]})
    with pytest.raises(
        pa.errors.SchemaErrors, match="2 schema errors were found"
    ):
        Schema.validate(df, lazy=True)


def test_check_regex():
    """Test the regex argument of the check decorator."""

    class Schema(pa.SchemaModel):
        a: Series[int]
        abc: Series[int]
        cba: Series[int]

        @pa.check("^a", regex=True)
        @classmethod
        def int_column_lt_100(cls, series: pd.Series) -> Iterable[bool]:
            return series < 100

    df = pd.DataFrame({"a": [101], "abc": [1], "cba": [200]})
    with pytest.raises(
        pa.errors.SchemaErrors, match="1 schema errors were found"
    ):
        Schema.validate(df, lazy=True)


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


@pytest.mark.xfail(reason="Bug repro", strict=True)
def test_inherit_schemamodel_fields_alias():
    """Test that columns and indices are inherited."""

    class Base(pa.SchemaModel):
        a: Series[int]
        idx: Index[str]

    class Mid(Base):
        b: Series[str] = pa.Field(alias="_b")
        idx: Index[str]

    class Child(Mid):
        b: Series[int]  # This line does not pass the test. BUG: it should pass
        # b: Series[int]  = pa.Field()  # This line passes the test

    expected = pa.DataFrameSchema(
        columns={"a": pa.Column(int), "b": pa.Column(int)},
        index=pa.Index(str),
    )

    assert expected == Child.to_schema()


def test_inherit_field_checks():
    """Test that checks are inherited and overridden."""

    class Base(pa.SchemaModel):
        a: Series[int]
        abc: Series[int]

        @pa.check("^a", regex=True)
        @classmethod
        def a_max(cls, series: pd.Series) -> Iterable[bool]:
            return series < 100

        @pa.check("a")
        @classmethod
        def a_min(cls, series: pd.Series) -> Iterable[bool]:
            return series > 1

    class Child(Base):
        @pa.check("a")
        @classmethod
        def a_max(cls, series: pd.Series) -> Iterable[bool]:
            return series < 10

    schema = Child.to_schema()
    assert len(schema.columns["a"].checks) == 2
    assert len(schema.columns["abc"].checks) == 0

    df = pd.DataFrame({"a": [15], "abc": [100]})
    err_msg = r"Column\s*a\s*a_max\s*\[15\]\s*1"
    with pytest.raises(pa.errors.SchemaErrors, match=err_msg):
        schema.validate(df, lazy=True)


def test_dataframe_check():
    """Test dataframe checks."""

    class Base(pa.SchemaModel):
        a: Series[int]
        b: Series[int]

        @pa.dataframe_check
        @classmethod
        def value_max(cls, df: pd.DataFrame) -> Iterable[bool]:
            return df < 200

    class Child(Base):
        @pa.dataframe_check()
        @classmethod
        def value_min(cls, df: pd.DataFrame) -> Iterable[bool]:
            return df > 0

        @pa.dataframe_check
        @classmethod
        def value_max(cls, df: pd.DataFrame) -> Iterable[bool]:
            return df < 100

    schema = Child.to_schema()
    assert len(schema.checks) == 2

    df = pd.DataFrame({"a": [101, 1], "b": [1, 0]})
    with pytest.raises(
        pa.errors.SchemaErrors, match="2 schema errors were found"
    ):
        schema.validate(df, lazy=True)


def test_config():
    """Test that Config can be inherited and translate into DataFrameSchema options."""

    class Base(pa.SchemaModel):
        a: Series[int]
        idx_1: Index[str]
        idx_2: Index[str]

        class Config:
            name = "Base schema"
            coerce = True
            ordered = True
            multiindex_coerce = True
            multiindex_strict = True
            multiindex_name: Optional[str] = "mi"

    class Child(Base):
        b: Series[int]

        class Config:
            name = "Child schema"
            strict = True
            multiindex_strict = False

    expected = pa.DataFrameSchema(
        columns={"a": pa.Column(int), "b": pa.Column(int)},
        index=pa.MultiIndex(
            [pa.Index(str, name="idx_1"), pa.Index(str, name="idx_2")],
            coerce=True,
            strict=False,
            name="mi",
        ),
        name="Child schema",
        coerce=True,
        strict=True,
        ordered=True,
    )

    assert expected == Child.to_schema()


def test_check_types():
    class Input(pa.SchemaModel):
        a: Series[int]
        b: Series[int]
        idx: Index[str]

    class Output(Input):
        c: Series[int]

    @pa.check_types
    def transform(df: DataFrame[Input]) -> DataFrame[Output]:
        return df.assign(c=lambda x: x.a + x.b)

    data = pd.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6]}, index=pd.Index(["a", "b", "c"])
    )
    assert isinstance(transform(data), pd.DataFrame)

    for invalid_data in [
        data.drop("a", axis="columns"),
        data.drop("b", axis="columns"),
        data.assign(a=["a", "b", "c"]),
        data.assign(b=["a", "b", "c"]),
        data.reset_index(drop=True),
    ]:
        with pytest.raises(pa.errors.SchemaError):
            transform(invalid_data)


def test_alias():
    """Test that columns and indices can be aliased."""

    class Schema(pa.SchemaModel):
        col_2020: Series[int] = pa.Field(alias=2020)
        idx: Index[int] = pa.Field(alias="_idx", check_name=True)

        @pa.check(2020)
        @classmethod
        def int_column_lt_100(cls, series: pd.Series) -> Iterable[bool]:
            return series < 100

    schema = Schema.to_schema()
    assert len(schema.columns) == 1
    assert schema.columns.get(2020, None) is not None
    assert schema.index.name == "_idx"

    df = pd.DataFrame({2020: [99]}, index=[0])
    df.index.name = "_idx"
    assert isinstance(Schema.validate(df), pd.DataFrame)

    # test multiindex
    class MISchema(pa.SchemaModel):
        idx1: Index[int] = pa.Field(alias="index0")
        idx2: Index[int] = pa.Field(alias="index1")

    actual = [index.name for index in MISchema.to_schema().index.indexes]
    assert actual == ["index0", "index1"]


def test_inherit_alias():
    """Test that aliases are inherited and can be overwritten."""

    class Base(pa.SchemaModel):
        a: Series[int] = pa.Field(alias="_a")
        b: Series[int] = pa.Field(alias="_b")

    class Child(Base):
        a: Series[int] = pa.Field(alias="_a_child")

    schema = Child.to_schema()
    assert len(schema.columns) == 2
    assert schema.columns.get("_a_child", None) is not None
    assert schema.columns.get("_b", None) is not None


@pytest.mark.xfail(reason="Bug repro", strict=True)
def test_inherit_override():
    """Test that overrides are reflected properly as expected"""

    class Base(pa.SchemaModel):
        a: Series[int] = pa.Field(alias="_a")
        b: Series[int] = pa.Field(alias="_b")

    class Child(Base):
        a: Series[str] = pa.Field(alias="_a_child")
        b: Series[str]  # This line does not pass the test. BUG: it should pass
        # b: Series[str] = pa.Field()  # This line passes the test

    schema = Child.to_schema()
    print(schema.columns)
    assert len(schema.columns) == 2
    assert schema.columns.get("_a_child") == pa.Column(str, name="_a_child")
    assert schema.columns.get("b") == pa.Column(str, name="b")
