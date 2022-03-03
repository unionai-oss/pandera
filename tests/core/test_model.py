"""Tests schema creation and validation from type annotations."""
# pylint:disable=missing-class-docstring,missing-function-docstring,too-few-public-methods
import re
from copy import deepcopy
from decimal import Decimal  # pylint:disable=C0415
from typing import Any, Iterable, Optional

import pandas as pd
import pytest

import pandera as pa
import pandera.extensions as pax
from pandera.typing import DataFrame, Index, Series, String


def test_to_schema_and_validate() -> None:
    """
    Test that SchemaModel.to_schema() can produce the correct schema and
    can validate dataframe objects.
    """

    class Schema(pa.SchemaModel):
        a: Series[int]
        b: Series[str]
        c: Series[Any]
        idx: Index[str]

    expected = pa.DataFrameSchema(
        name="Schema",
        columns={"a": pa.Column(int), "b": pa.Column(str), "c": pa.Column()},
        index=pa.Index(str),
    )
    assert expected == Schema.to_schema()

    Schema(pd.DataFrame({"a": [1], "b": ["foo"], "c": [3.4]}, index=["1"]))
    with pytest.raises(pa.errors.SchemaError):
        Schema(pd.DataFrame({"a": [1]}))


def test_empty_schema() -> None:
    """Test that SchemaModel supports empty schemas."""

    empty_schema = pa.DataFrameSchema(name="EmptySchema")

    class EmptySchema(pa.SchemaModel):
        pass

    assert empty_schema == EmptySchema.to_schema()

    class Schema(pa.SchemaModel):
        a: Series[int]

    class EmptySubSchema(Schema):
        pass

    empty_sub_schema = pa.DataFrameSchema(
        name="EmptySubSchema",
        columns={"a": pa.Column(int)},
    )
    assert empty_sub_schema == EmptySubSchema.to_schema()

    empty_parent_schema = pa.DataFrameSchema(
        name="EmptyParentSchema",
        columns={"a": pa.Column(int)},
    )

    class EmptyParentSchema(EmptySchema):
        a: Series[int]

    assert empty_parent_schema == EmptyParentSchema.to_schema()


def test_invalid_annotations() -> None:
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

    class InvalidDtype(pa.SchemaModel):
        d: Series[Decimal]  # type: ignore

    with pytest.raises(
        TypeError, match="dtype '<class 'decimal.Decimal'>' not understood"
    ):
        InvalidDtype.to_schema()


def test_optional_column() -> None:
    """Test that optional columns are not required."""

    class Schema(pa.SchemaModel):
        a: Optional[Series[str]]
        b: Optional[Series[str]] = pa.Field(eq="b")
        c: Optional[Series[String]]  # test pandera.typing alias

    schema = Schema.to_schema()
    assert not schema.columns["a"].required
    assert not schema.columns["b"].required
    assert not schema.columns["c"].required


def test_optional_index() -> None:
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


def test_empty_dtype() -> None:
    expected = pa.DataFrameSchema(
        name="EmptyDtypeSchema",
        columns={"empty_column": pa.Column()},
    )

    class EmptyDtypeSchema(pa.SchemaModel):
        empty_column: pa.typing.Series

    assert EmptyDtypeSchema.to_schema() == expected


def test_schemamodel_with_fields() -> None:
    """Test that Fields are translated in the schema."""

    class Schema(pa.SchemaModel):
        a: Series[int] = pa.Field(eq=9, ne=0)
        b: Series[str]
        idx: Index[str] = pa.Field(str_length={"min_value": 1})

    actual = Schema.to_schema()
    expected = pa.DataFrameSchema(
        name="Schema",
        columns={
            "a": pa.Column(
                int, checks=[pa.Check.equal_to(9), pa.Check.not_equal_to(0)]
            ),
            "b": pa.Column(str),
        },
        index=pa.Index(str, pa.Check.str_length(1)),
    )

    assert actual == expected


def test_invalid_field() -> None:
    class Schema(pa.SchemaModel):
        a: Series[int] = 0  # type: ignore[assignment]  # mypy identifies the wrong usage correctly

    with pytest.raises(
        pa.errors.SchemaInitError, match="'a' can only be assigned a 'Field'"
    ):
        Schema.to_schema()


def test_multiindex() -> None:
    """Test that multiple Index annotations create a MultiIndex."""

    class Schema(pa.SchemaModel):
        a: Index[int] = pa.Field(gt=0)
        b: Index[str]

    expected = pa.DataFrameSchema(
        name="Schema",
        index=pa.MultiIndex(
            [
                pa.Index(int, name="a", checks=pa.Check.gt(0)),
                pa.Index(str, name="b"),
            ]
        ),
    )
    assert expected == Schema.to_schema()


def test_column_check_name() -> None:
    """Test that column name is mandatory."""

    class Schema(pa.SchemaModel):
        a: Series[int] = pa.Field(check_name=False)

    with pytest.raises(pa.errors.SchemaInitError):
        Schema.to_schema()


def test_single_index_check_name() -> None:
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


def test_multiindex_check_name() -> None:
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


def test_check_validate_method() -> None:
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


def test_check_validate_method_field() -> None:
    """Test validate method on valid data."""

    class Schema(pa.SchemaModel):
        a: Series[int] = pa.Field()
        b: Series[int]

        @pa.check(a)
        def int_column_lt_200(cls, series: pd.Series) -> Iterable[bool]:
            # pylint:disable=no-self-argument
            assert cls is Schema
            return series < 200

        @pa.check(a, "b")
        def int_column_lt_100(cls, series: pd.Series) -> Iterable[bool]:
            # pylint:disable=no-self-argument
            assert cls is Schema
            return series < 100

    df = pd.DataFrame({"a": [99], "b": [99]})
    assert isinstance(Schema.validate(df, lazy=True), pd.DataFrame)


def test_check_validate_method_aliased_field() -> None:
    """Test validate method on valid data."""

    class Schema(pa.SchemaModel):
        a: Series[int] = pa.Field(alias=2020, gt=50)

        @pa.check(a)
        def int_column_lt_100(cls, series: pd.Series) -> Iterable[bool]:
            # pylint:disable=no-self-argument
            assert cls is Schema
            return series < 100

    df = pd.DataFrame({2020: [99]})
    assert len(Schema.to_schema().columns[2020].checks) == 2
    assert isinstance(Schema.validate(df, lazy=True), pd.DataFrame)


def test_check_single_column() -> None:
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


def test_check_single_index() -> None:
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


def test_field_and_check() -> None:
    """Test the combination of a field and a check on the same column."""

    class Schema(pa.SchemaModel):
        a: Series[int] = pa.Field(eq=1)

        @pa.check("a")
        @classmethod
        def int_column_lt_100(cls, series: pd.Series) -> Iterable[bool]:
            return series < 100

    schema = Schema.to_schema()
    assert len(schema.columns["a"].checks) == 2


def test_check_non_existing() -> None:
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


def test_multiple_checks() -> None:
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


def test_check_multiple_columns() -> None:
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


def test_check_regex() -> None:
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


def test_inherit_schemamodel_fields() -> None:
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
        name="Child",
        columns={"a": pa.Column(int), "b": pa.Column(int)},
        index=pa.Index(str),
    )

    assert expected == Child.to_schema()


def test_inherit_schemamodel_fields_alias() -> None:
    """Test that columns and index aliases are inherited."""

    class Base(pa.SchemaModel):
        a: Series[int]
        idx: Index[str]

    class Mid(Base):
        b: Series[str] = pa.Field(alias="_b")
        idx: Index[str]

    class ChildOverrideAttr(Mid):
        b: Series[int]

    class ChildOverrideAlias(Mid):
        b: Series[str] = pa.Field(alias="new_b")

    class ChildNewAttr(Mid):
        c: Series[int]

    class ChildEmpty(Mid):
        pass

    expected_mid = pa.DataFrameSchema(
        name="Mid",
        columns={"a": pa.Column(int), "_b": pa.Column(str)},
        index=pa.Index(str),
    )
    expected_child_override_attr = expected_mid.rename_columns(
        {"_b": "b"}
    ).update_column("b", dtype=int)
    expected_child_override_attr.name = "ChildOverrideAttr"

    expected_child_override_alias = expected_mid.rename_columns(
        {"_b": "new_b"}
    )
    expected_child_override_alias.name = "ChildOverrideAlias"

    expected_child_new_attr = expected_mid.add_columns(
        {
            "c": pa.Column(int),
        }
    )
    expected_child_new_attr.name = "ChildNewAttr"

    expected_child_empty = deepcopy(expected_mid)
    expected_child_empty.name = "ChildEmpty"

    assert expected_mid == Mid.to_schema()
    assert expected_child_override_attr == ChildOverrideAttr.to_schema()
    assert expected_child_override_alias == ChildOverrideAlias.to_schema()
    assert expected_child_new_attr == ChildNewAttr.to_schema()
    assert expected_child_empty == ChildEmpty.to_schema()


def test_inherit_field_checks() -> None:
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


def test_dataframe_check() -> None:
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


def test_registered_dataframe_checks(
    extra_registered_checks: None,  # pylint: disable=unused-argument
) -> None:
    """Check that custom check inheritance works"""
    # pylint: disable=unused-variable

    @pax.register_check_method(statistics=["one_arg"])
    def base_check(df, *, one_arg):
        # pylint: disable=unused-argument
        return True

    @pax.register_check_method(statistics=["one_arg", "two_arg"])
    def child_check(df, *, one_arg, two_arg):
        # pylint: disable=unused-argument
        return True

    # pylint: enable=unused-variable

    check_vals = {
        "one_arg": 150,
        "two_arg": "hello",
        "one_arg_prime": "not_150",
    }

    class Base(pa.SchemaModel):
        a: Series[int]
        b: Series[int]

        class Config:
            no_param_check = ()
            base_check = check_vals["one_arg"]

    class Child(Base):
        class Config:
            base_check = check_vals["one_arg_prime"]
            child_check = {
                "one_arg": check_vals["one_arg"],
                "two_arg": check_vals["two_arg"],
            }

    base = Base.to_schema()
    child = Child.to_schema()

    expected_stats_base = {
        "no_param_check": {},
        "base_check": {"one_arg": check_vals["one_arg"]},
    }

    expected_stats_child = {
        "no_param_check": {},
        "base_check": {"one_arg": check_vals["one_arg_prime"]},
        "child_check": {
            "one_arg": check_vals["one_arg"],
            "two_arg": check_vals["two_arg"],
        },
    }

    assert {b.name: b.statistics for b in base.checks} == expected_stats_base
    assert {c.name: c.statistics for c in child.checks} == expected_stats_child

    # check that unregistered checks raise
    with pytest.raises(AttributeError, match=".*custom checks.*"):

        class ErrorSchema(pa.SchemaModel):
            class Config:
                unknown_check = {}  # type: ignore[var-annotated]

        # Check lookup happens at validation/to_schema conversion time
        # This means that you can register checks after defining a Config,
        # but also because of caching you can refer to a check that no longer
        # exists for some order of operations.
        ErrorSchema.to_schema()


def test_config() -> None:
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
            unique_column_names = True

    class Child(Base):
        b: Series[int]

        class Config:
            name = "Child schema"
            strict = True
            multiindex_strict = False
            description = "foo"
            title = "bar"

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
        unique_column_names=True,
        description="foo",
        title="bar",
    )

    assert expected == Child.to_schema()


def test_config_docstrings() -> None:
    class Model(pa.SchemaModel):
        """foo"""

        a: Series[int]

    assert Model.__doc__ == Model.to_schema().description


class Input(pa.SchemaModel):
    a: Series[int]
    b: Series[int]
    idx: Index[str]


class Output(Input):
    c: Series[int]


def test_check_types() -> None:
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


def test_alias() -> None:
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
    assert len(Schema.to_schema().columns[2020].checks) == 1
    assert isinstance(Schema.validate(df), pd.DataFrame)

    # test multiindex
    class MISchema(pa.SchemaModel):
        idx1: Index[int] = pa.Field(alias="index0")
        idx2: Index[int] = pa.Field(alias="index1")

    actual = [index.name for index in MISchema.to_schema().index.indexes]
    assert actual == ["index0", "index1"]


def test_inherit_alias() -> None:
    """Test that aliases are inherited and can be overwritten."""

    # Three cases to consider per annotation:
    #   - Field omitted
    #   - Field
    #   - Field with alias

    class Base(pa.SchemaModel):
        a: Series[int]
        b: Series[int] = pa.Field()
        c: Series[int] = pa.Field(alias="_c")

    class ChildExtend(Base):
        extra: Series[str]

    schema_ext = ChildExtend.to_schema()
    assert len(schema_ext.columns) == 4
    assert schema_ext.columns.get("a") == pa.Column(int, name="a")
    assert schema_ext.columns.get("b") == pa.Column(int, name="b")
    assert schema_ext.columns.get("_c") == pa.Column(int, name="_c")
    assert schema_ext.columns.get("extra") == pa.Column(str, name="extra")

    class ChildOmitted(Base):
        a: Series[str]
        b: Series[str]
        c: Series[str]

    schema_omitted = ChildOmitted.to_schema()
    assert len(schema_omitted.columns) == 3
    assert schema_omitted.columns.get("a") == pa.Column(str, name="a")
    assert schema_omitted.columns.get("b") == pa.Column(str, name="b")
    assert schema_omitted.columns.get("c") == pa.Column(str, name="c")

    class ChildField(Base):
        a: Series[str] = pa.Field()
        b: Series[str] = pa.Field()
        c: Series[str] = pa.Field()

    schema_field = ChildField.to_schema()
    assert len(schema_field.columns) == 3
    assert schema_field.columns.get("a") == pa.Column(str, name="a")
    assert schema_field.columns.get("b") == pa.Column(str, name="b")
    assert schema_field.columns.get("c") == pa.Column(str, name="c")

    class ChildAlias(Base):
        a: Series[str] = pa.Field(alias="_a")
        b: Series[str] = pa.Field(alias="_b")
        c: Series[str] = pa.Field(alias="_c")

    schema_alias = ChildAlias.to_schema()
    assert len(schema_alias.columns) == 3
    assert schema_alias.columns.get("_a") == pa.Column(str, name="_a")
    assert schema_alias.columns.get("_b") == pa.Column(str, name="_b")
    assert schema_alias.columns.get("_c") == pa.Column(str, name="_c")


def test_field_name_access():
    """Test that column and index names can be accessed through the class"""

    class Base(pa.SchemaModel):
        a: Series[int]
        b: Series[int] = pa.Field()
        c: Series[int] = pa.Field(alias="_c")
        d: Series[int] = pa.Field(alias=123)
        i1: Index[int]
        i2: Index[int] = pa.Field()

    assert Base.a == "a"
    assert Base.b == "b"
    assert Base.c == "_c"
    assert Base.d == 123
    assert Base.i1 == "i1"
    assert Base.i2 == "i2"


def test_field_name_access_inherit() -> None:
    """Test that column and index names can be accessed through the class"""

    class Base(pa.SchemaModel):
        a: Series[int]
        b: Series[int] = pa.Field()
        c: Series[int] = pa.Field(alias="_c")
        d: Series[int] = pa.Field(alias=123)
        i1: Index[int]
        i2: Index[int] = pa.Field()

    class Child(Base):
        b: Series[str] = pa.Field(alias="_b")
        c: Series[str]
        d: Series[str] = pa.Field()
        extra1: Series[int]
        extra2: Series[int] = pa.Field()
        extra3: Series[int] = pa.Field(alias="_extra3")
        i1: Index[str]
        i3: Index[int] = pa.Field(alias="_i3")

    expected_base = pa.DataFrameSchema(
        name="Base",
        columns={
            "a": pa.Column(int),
            "b": pa.Column(int),
            "_c": pa.Column(int),
            123: pa.Column(int),
        },
        index=pa.MultiIndex(
            [
                pa.Index(int, name="i1"),
                pa.Index(int, name="i2"),
            ]
        ),
    )

    expected_child = pa.DataFrameSchema(
        name="Child",
        columns={
            "a": pa.Column(int),
            "_b": pa.Column(str),
            "c": pa.Column(str),
            "d": pa.Column(str),
            "extra1": pa.Column(int),
            "extra2": pa.Column(int),
            "_extra3": pa.Column(int),
        },
        index=pa.MultiIndex(
            [
                pa.Index(str, name="i1"),
                pa.Index(int, name="i2"),
                pa.Index(int, name="_i3"),
            ]
        ),
    )

    assert expected_base == Base.to_schema()
    assert expected_child == Child.to_schema()
    assert Child.a == "a"  # pylint:disable=no-member
    assert Child.b == "_b"
    assert Child.c == "c"
    assert Child.d == "d"
    assert Child.extra1 == "extra1"
    assert Child.extra2 == "extra2"
    assert Child.extra3 == "_extra3"
    assert Child.i1 == "i1"
    assert Child.i2 == "i2"
    assert Child.i3 == "_i3"


def test_column_access_regex() -> None:
    """Test that column regex alias is reflected in schema attribute."""

    class Schema(pa.SchemaModel):
        col_regex: Series[str] = pa.Field(alias="column_([0-9])+", regex=True)

    assert Schema.col_regex == "column_([0-9])+"


def test_schema_name_override():
    """
    Test that setting name in Config manually does not propagate to other
    SchemaModels.
    """

    class Foo(pa.SchemaModel):
        pass

    class Bar(pa.SchemaModel):
        pass

    assert Foo.Config.name == "Foo"

    Foo.Config.name = "foo"

    assert Foo.Config.name == "foo"
    assert Bar.Config.name == "Bar"


def test_validate_coerce_on_init():
    """Test that DataFrame[Schema] validates and coerces on initialization."""

    class Schema(pa.SchemaModel):
        state: Series[str]
        city: Series[str]
        price: Series[float] = pa.Field(
            in_range={"min_value": 5, "max_value": 20}
        )

        class Config:
            coerce = True

    class SchemaNoCoerce(Schema):
        class Config:
            coerce = False

    raw_data = {
        "state": ["NY", "FL", "GA", "CA"],
        "city": ["New York", "Miami", "Atlanta", "San Francisco"],
        "price": [8, 12, 10, 16],
    }
    pandera_validated_df = DataFrame[Schema](raw_data)
    pandas_df = pd.DataFrame(raw_data)
    assert pandera_validated_df.equals(Schema.validate(pandas_df))
    assert isinstance(pandera_validated_df, DataFrame)
    assert isinstance(pandas_df, pd.DataFrame)

    with pytest.raises(
        pa.errors.SchemaError,
        match="^expected series 'price' to have type float64, got int64$",
    ):
        DataFrame[SchemaNoCoerce](raw_data)
