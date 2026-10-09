"""Cross-backend conformance for ``Enum`` and ``Literal`` data types.

The pandas and polars engines used to disagree about enums -- polars derived
categories from member values, pandas from the members themselves -- so the
same ``DataFrameModel`` meant different things on each backend. Nothing
exercised both engines with the same type, which is why the divergence went
unnoticed; this module is that test.

It lives under ``tests/pandas`` rather than ``tests/polars`` because polars
test modules must stay importable without pandas (see
``tests/polars/test_polars_no_pandas.py``).
"""

import enum
import typing

import pandas as pd
import pytest

import pandera.pandas as pandas_pa
from pandera._enum_literal import enum_categories, literal_categories
from pandera.engines.pandas_engine import Engine as PandasEngine

pl = pytest.importorskip("polars")
polars_pa = pytest.importorskip("pandera.polars")
polars_engine = pytest.importorskip("pandera.engines.polars_engine")


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


ALL_TYPES = [
    Department,
    StrDepartment,
    Frustration,
    typing.Literal["billing", "technical"],
    typing.Literal[1, 2],
]


def _expected_categories(data_type):
    if isinstance(data_type, type) and issubclass(data_type, enum.Enum):
        return enum_categories(data_type)
    return literal_categories(data_type)


@pytest.mark.parametrize("data_type", ALL_TYPES)
def test_backends_agree_on_categories(data_type):
    """Both engines derive the same option set from the same Python type.

    The dtype *objects* differ where a backend has no equivalent -- polars has
    no integer categorical -- but the option set must match.
    """
    expected = _expected_categories(data_type)

    assert tuple(PandasEngine.dtype(data_type).categories) == expected

    polars_dtype = polars_engine.Engine.dtype(data_type)
    if isinstance(polars_dtype.type, pl.Enum):
        assert tuple(polars_dtype.type.categories) == expected
    else:
        # Non-string options fall back to the value dtype; check the values
        # actually round-trip through it.
        series = pl.Series(list(expected), dtype=polars_dtype.type)
        assert series.to_list() == list(expected)


@pytest.mark.parametrize("data_type", [Department, StrDepartment, Frustration])
def test_backends_agree_on_coercion(data_type):
    values = list(enum_categories(data_type))

    pandas_out = pandas_pa.DataFrameSchema(
        {"d": pandas_pa.Column(data_type, coerce=True)}
    ).validate(pd.DataFrame({"d": values}))
    polars_out = polars_pa.DataFrameSchema(
        {"d": polars_pa.Column(data_type, coerce=True)}
    ).validate(pl.DataFrame({"d": values}))

    assert pandas_out["d"].tolist() == polars_out["d"].to_list() == values


@pytest.mark.parametrize("data_type", ALL_TYPES)
def test_neither_backend_accepts_heterogeneous_literals(data_type):
    """Whatever each backend does with a valid option set, both must refuse an
    ambiguous one rather than silently keeping one value type."""
    del data_type  # the parametrization only keeps the ids readable
    for engine in (PandasEngine, polars_engine.Engine):
        with pytest.raises(TypeError, match="mixes value types"):
            engine.dtype(typing.Literal["a", 1])
