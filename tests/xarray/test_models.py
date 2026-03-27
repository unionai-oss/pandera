"""Tests for declarative xarray models (DataArrayModel, DatasetModel)."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

import pandera.errors  # noqa: E402
import pandera.xarray as pa  # noqa: E402
from pandera.typing.xarray import Coordinate  # noqa: E402
from tests.xarray.conftest import GridModel  # noqa: E402


def test_data_array_model_to_schema_and_validate(grid_da):
    s = GridModel.to_schema()
    assert isinstance(s, pa.DataArraySchema)
    assert s.dims == ("x", "y")
    assert s.name == "values"
    assert "x" in (s.coords or {})

    out = GridModel.validate(grid_da)
    assert out.identical(grid_da)


def test_data_array_model_column_name_accessor():
    class M(pa.DataArrayModel):
        data: np.float64 = pa.Field()
        t: Coordinate[np.float64]

        class Config:
            dims = ("t",)
            name = "a"

    assert M.t == "t"


def test_dataset_model_flat_fields():
    class DS(pa.DatasetModel):
        a: np.float64 = pa.Field(dims=("x",))
        x: Coordinate[np.float64]

    ds = xr.Dataset(
        {"a": (("x",), np.ones(2, dtype=np.float64))},
        coords={"x": np.arange(2, dtype=np.float64)},
    )
    DS.validate(ds)


def test_dataset_model_nested_data_array_model():
    class Var(pa.DataArrayModel):
        data: np.float64 = pa.Field()
        x: Coordinate[np.float64]

        class Config:
            dims = ("x",)
            name = "v"

    class DS(pa.DatasetModel):
        v: Var
        x: Coordinate[np.float64]

    da = xr.DataArray(
        np.ones(2),
        dims=("x",),
        coords={"x": np.arange(2, dtype=np.float64)},
        name="v",
    )
    ds = xr.Dataset({"v": da})
    DS.validate(ds)
    assert DS.v == "v"


def test_dataset_model_optional_var():
    class DS(pa.DatasetModel):
        a: np.float64 | None = pa.Field(dims=("x",), required=False)
        x: Coordinate[np.float64]

    ds = xr.Dataset(coords={"x": np.arange(2, dtype=np.float64)})
    DS.validate(ds)


def test_data_array_model_requires_data_field():
    class Bad(pa.DataArrayModel):
        x: Coordinate[np.float64]

        class Config:
            dims = ("x",)

    with pytest.raises(pandera.errors.SchemaInitError, match="data"):
        Bad.to_schema()
