"""Enum/Literal dtypes on the polars backend.

This module must not import pandas, directly or indirectly: the polars extra is
disjoint from the pandas extra and ``test_polars_no_pandas`` blocks both in
process. Cross-backend conformance lives in
``tests/pandas/test_enum_literal_conformance.py``.
"""

import enum
import typing

import polars as pl
import pytest

import pandera.polars as pa
from pandera.config import CONFIG
from pandera.engines.polars_engine import Engine as PolarsEngine


class Department(enum.Enum):
    billing = "billing"
    technical = "technical"


class StrDepartment(
    str, enum.Enum
):  # StrEnum needs 3.11; pandera supports 3.10
    billing = "billing"
    technical = "technical"


class Frustration(enum.IntEnum):
    calm = 0
    annoyed = 1
    angry = 2


# --------------------------------------------------------------------------
# dtype mapping
# --------------------------------------------------------------------------


@pytest.mark.parametrize("enum_type", [Department, StrDepartment])
def test_string_enum_maps_to_polars_enum(enum_type):
    dtype = PolarsEngine.dtype(enum_type)
    assert dtype.type == pl.Enum(["billing", "technical"])


def test_int_enum_maps_to_value_dtype():
    """Polars categoricals are string-only, so an ``IntEnum`` maps to the
    dtype of its values rather than failing outright."""
    assert PolarsEngine.dtype(Frustration).type == pl.Int64


def test_string_literal_maps_to_polars_enum():
    dtype = PolarsEngine.dtype(typing.Literal["billing", "technical"])
    assert dtype.type == pl.Enum(["billing", "technical"])


def test_int_literal_maps_to_value_dtype():
    assert PolarsEngine.dtype(typing.Literal[1, 2]).type == pl.Int64


def test_heterogeneous_literal_raises():
    with pytest.raises(TypeError, match="mixes value types"):
        PolarsEngine.dtype(typing.Literal["a", 1])


# --------------------------------------------------------------------------
# end-to-end
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    condition=CONFIG.use_narwhals_backend,
    reason=(
        "The narwhals backend does not validate pl.Enum columns, and does not "
        "apply coerce=True; a native pl.Enum column fails the same way on main."
    ),
    strict=True,
)
def test_enum_model_validates():
    class Model(pa.DataFrameModel):
        d: Department

    data = pl.DataFrame(
        {
            "d": pl.Series(
                ["billing", "technical"],
                dtype=pl.Enum(["billing", "technical"]),
            )
        }
    )
    assert Model.validate(data).to_dicts() == [
        {"d": "billing"},
        {"d": "technical"},
    ]


@pytest.mark.xfail(
    condition=CONFIG.use_narwhals_backend,
    reason=(
        "The narwhals backend does not validate pl.Enum columns, and does not "
        "apply coerce=True; a native pl.Enum column fails the same way on main."
    ),
    strict=True,
)
def test_enum_coerces_from_strings():
    schema = pa.DataFrameSchema({"d": pa.Column(Department, coerce=True)})
    out = schema.validate(pl.DataFrame({"d": ["billing", "technical"]}))
    assert out["d"].to_list() == ["billing", "technical"]


@pytest.mark.xfail(
    condition=CONFIG.use_narwhals_backend,
    reason=(
        "The narwhals backend does not validate pl.Enum columns, and does not "
        "apply coerce=True; a native pl.Enum column fails the same way on main."
    ),
    strict=True,
)
def test_literal_coerces_from_strings():
    schema = pa.DataFrameSchema(
        {"x": pa.Column(typing.Literal["billing", "technical"], coerce=True)}
    )
    out = schema.validate(pl.DataFrame({"x": ["billing"]}))
    assert out["x"].to_list() == ["billing"]
