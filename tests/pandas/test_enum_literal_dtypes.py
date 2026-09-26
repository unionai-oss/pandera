"""Tests for ``Enum`` and ``Literal`` data types on the pandas backend."""

import enum
import typing

import pandas as pd
import pytest

import pandera.pandas as pa
from pandera.engines.pandas_engine import Engine
from pandera.errors import SchemaError, SchemaErrors
from pandera.typing import Series


class Department(enum.Enum):
    """A plain enum: members are *not* equal to their values."""

    billing = "billing"
    """Payment, invoices or subscription issues."""
    technical = "technical"
    """Bugs, outages or integration problems."""


class StrDepartment(
    str, enum.Enum
):  # StrEnum needs 3.11; pandera supports 3.10
    billing = "billing"
    technical = "technical"


class Frustration(enum.IntEnum):
    calm = 0
    annoyed = 1
    angry = 2


class Renamed(enum.Enum):
    """An enum whose values differ from its member names."""

    first = "a"
    second = "b"


# --------------------------------------------------------------------------
# Enum -> Category over member values
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "enum_type,categories,ordered",
    [
        (Department, ("billing", "technical"), False),
        (StrDepartment, ("billing", "technical"), False),
        (Frustration, (0, 1, 2), True),
        (Renamed, ("a", "b"), False),
    ],
)
def test_enum_maps_to_category_of_values(enum_type, categories, ordered):
    dtype = Engine.dtype(enum_type)
    assert dtype.categories == categories
    assert dtype.ordered is ordered


def test_plain_enum_validates_raw_values():
    """Regression: a plain ``Enum`` used to produce a category of *members*,
    so a column holding the values it was declared with failed validation."""

    class Model(pa.DataFrameModel):
        d: Department

    data = pd.DataFrame({"d": pd.Categorical(["billing", "technical"])})
    assert Model.validate(data)["d"].tolist() == ["billing", "technical"]


def test_plain_enum_coerces_from_values():
    schema = pa.DataFrameSchema({"d": pa.Column(Department, coerce=True)})
    out = schema.validate(pd.DataFrame({"d": ["billing", "technical"]}))
    assert out["d"].tolist() == ["billing", "technical"]


def test_plain_enum_coerces_from_members():
    schema = pa.DataFrameSchema({"d": pa.Column(Department, coerce=True)})
    out = schema.validate(
        pd.DataFrame({"d": [Department.billing, Department.technical]})
    )
    assert out["d"].tolist() == ["billing", "technical"]


def test_enum_coercion_still_rejects_unknown_values():
    schema = pa.DataFrameSchema({"d": pa.Column(Department, coerce=True)})
    with pytest.raises((SchemaError, SchemaErrors)):
        schema.validate(pd.DataFrame({"d": ["billing", "nope"]}))


def test_enum_coerce_value_accepts_member_value_and_name():
    dtype = Engine.dtype(Renamed)
    assert dtype.coerce_value(Renamed.first) == "a"
    assert dtype.coerce_value("a") == "a"
    assert dtype.coerce_value("first") == "a"
    with pytest.raises(TypeError):
        dtype.coerce_value("nope")


def test_int_enum_is_ordered_and_supports_comparison_checks():
    schema = pa.DataFrameSchema(
        {"f": pa.Column(Frustration, coerce=True, checks=pa.Check.ge(1))}
    )
    out = schema.validate(pd.DataFrame({"f": [1, 2]}))
    assert out["f"].tolist() == [1, 2]


def test_enum_dtype_equality_ignores_originating_enum():
    """A dtype built from an enum must compare equal to the plain categorical
    a dataframe reports, or every dtype check against real data would fail."""
    assert Engine.dtype(Department) == Engine.dtype(
        pd.CategoricalDtype(["billing", "technical"])
    )
    assert hash(Engine.dtype(Department)) == hash(
        Engine.dtype(pd.CategoricalDtype(["billing", "technical"]))
    )


# --------------------------------------------------------------------------
# Literal -> Category over literal arguments
# --------------------------------------------------------------------------


def test_literal_maps_to_category():
    dtype = Engine.dtype(typing.Literal["billing", "technical"])
    assert dtype.categories == ("billing", "technical")


def test_literal_bare_annotation():
    """Regression: a bare ``Literal`` annotation used to raise
    ``SchemaInitError: Invalid annotation``."""

    class Model(pa.DataFrameModel):
        x: typing.Literal["billing", "technical"]

    assert str(Model.to_schema().columns["x"].dtype) == "category"
    data = pd.DataFrame(
        {"x": pd.Categorical(["billing"], categories=["billing", "technical"])}
    )
    assert Model.validate(data)["x"].tolist() == ["billing"]


def test_literal_series_annotation():
    """Regression: ``Series[Literal[...]]`` used to hand the *first literal
    value* to the dtype engine, raising an opaque ``TypeError``."""

    class Model(pa.DataFrameModel):
        x: Series[typing.Literal["billing", "technical"]]

    assert Model.to_schema().columns["x"].dtype.categories == (
        "billing",
        "technical",
    )


def test_literal_of_dtype_names_is_not_silently_mistyped():
    """The dangerous case: every literal value here is a valid dtype string,
    so the option set used to be discarded and the column typed ``int64``."""

    class Model(pa.DataFrameModel):
        x: Series[typing.Literal["int64", "float64"]]

    dtype = Model.to_schema().columns["x"].dtype
    assert str(dtype) == "category"
    assert dtype.categories == ("int64", "float64")


def test_literal_membership_is_enforced():
    schema = pa.DataFrameSchema(
        {"x": pa.Column(typing.Literal["billing", "technical"], coerce=True)}
    )
    with pytest.raises((SchemaError, SchemaErrors)):
        schema.validate(pd.DataFrame({"x": ["nope"]}))


def test_heterogeneous_literal_raises():
    with pytest.raises(TypeError, match="mixes value types"):
        Engine.dtype(typing.Literal["a", 1])


def test_int_literal_maps_to_category_of_ints():
    dtype = Engine.dtype(typing.Literal[1, 2, 3])
    assert dtype.categories == (1, 2, 3)
