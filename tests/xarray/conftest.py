"""Shared fixtures and model definitions for xarray tests."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

import pandera.xarray as pa  # noqa: E402
from pandera.typing.xarray import Coordinate  # noqa: E402


class GridModel(pa.DataArrayModel):
    data: np.float64 = pa.Field()
    x: Coordinate[np.float64]
    y: Coordinate[np.float64]

    class Config:
        dims = ("x", "y")
        name = "values"


class SurfaceModel(pa.DatasetModel):
    temperature: np.float64 = pa.Field(dims=("x",))
    x: Coordinate[np.float64]


@pytest.fixture
def grid_da():
    return xr.DataArray(
        np.ones((2, 3)),
        dims=("x", "y"),
        coords={
            "x": np.arange(2, dtype=np.float64),
            "y": np.arange(3, dtype=np.float64),
        },
        name="values",
    )


@pytest.fixture
def surface_ds():
    return xr.Dataset(
        {"temperature": (("x",), np.ones(3))},
        coords={"x": np.arange(3, dtype=np.float64)},
    )
