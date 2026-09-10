"""Tests for xarray hypothesis strategies."""

import numpy as np
import pytest
import xarray as xr

from pandera.api.checks import Check
from pandera.api.xarray.components import Coordinate, DataVar
from pandera.api.xarray.container import DataArraySchema, DatasetSchema

pytest.importorskip("hypothesis")

from hypothesis import given, settings

from pandera.strategies.xarray_strategies import (
    data_array_schema_strategy,
    data_array_strategy,
    dataset_schema_strategy,
    dataset_strategy,
    xarray_dtype_strategy,
)


class TestDtypeStrategy:
    @given(xarray_dtype_strategy("float64"))
    def test_float64(self, value):
        assert isinstance(value, (np.floating, float))

    @given(xarray_dtype_strategy("int32"))
    def test_int32(self, value):
        assert isinstance(value, (np.integer, int))

    @given(xarray_dtype_strategy(np.dtype("bool")))
    def test_bool(self, value):
        assert isinstance(value, (np.bool_, bool))

    def test_none_defaults_to_float64(self):
        strat = xarray_dtype_strategy(None)
        assert strat is not None


class TestDataArrayStrategy:
    @given(data_array_strategy(dtype="float64", dims=("x",)))
    @settings(max_examples=5)
    def test_basic(self, da):
        assert isinstance(da, xr.DataArray)
        assert da.dims == ("x",)
        assert da.dtype == np.float64

    @given(
        data_array_strategy(
            dtype="float64",
            dims=("x", "y"),
            sizes={"x": 4, "y": 5},
        )
    )
    @settings(max_examples=5)
    def test_with_sizes(self, da):
        assert da.sizes["x"] == 4
        assert da.sizes["y"] == 5

    @given(
        data_array_strategy(
            dtype="float64",
            dims=("x",),
            shape=(7,),
        )
    )
    @settings(max_examples=5)
    def test_with_shape(self, da):
        assert da.shape == (7,)

    @given(
        data_array_strategy(
            dtype="float64",
            dims=("x",),
            name="temperature",
        )
    )
    @settings(max_examples=5)
    def test_with_name(self, da):
        assert da.name == "temperature"

    @given(
        data_array_strategy(
            dtype="int64",
            dims=("x",),
            coords={"x": {"dtype": "float64"}},
        )
    )
    @settings(max_examples=5)
    def test_with_coords(self, da):
        assert "x" in da.coords

    @given(data_array_strategy(dtype="float64", dims=("a", "b"), size=2))
    @settings(max_examples=5)
    def test_default_size(self, da):
        assert da.sizes["a"] == 2
        assert da.sizes["b"] == 2


class TestDatasetStrategy:
    @given(
        dataset_strategy(
            data_vars={
                "temp": {"dtype": "float64", "dims": ("x", "y")},
                "pressure": {"dtype": "float64", "dims": ("x", "y")},
            },
            sizes={"x": 3, "y": 4},
        )
    )
    @settings(max_examples=5)
    def test_basic(self, ds):
        assert isinstance(ds, xr.Dataset)
        assert "temp" in ds.data_vars
        assert "pressure" in ds.data_vars
        assert ds.sizes["x"] == 3
        assert ds.sizes["y"] == 4

    @given(dataset_strategy())
    @settings(max_examples=5)
    def test_defaults(self, ds):
        assert isinstance(ds, xr.Dataset)
        assert "var_0" in ds.data_vars

    @given(
        dataset_strategy(
            data_vars={"v": {"dtype": "float64", "dims": ("x",)}},
            coords={"x": {"dtype": "float64"}},
            sizes={"x": 5},
        )
    )
    @settings(max_examples=5)
    def test_with_coords(self, ds):
        assert "x" in ds.coords


class TestDataArraySchemaStrategy:
    @given(
        data_array_schema_strategy(
            DataArraySchema(
                dtype="float64",
                dims=("x", "y"),
                sizes={"x": 3, "y": 4},
                name="temp",
            )
        )
    )
    @settings(max_examples=5)
    def test_from_schema(self, da):
        assert isinstance(da, xr.DataArray)
        assert da.dims == ("x", "y")
        assert da.sizes["x"] == 3
        assert da.sizes["y"] == 4
        assert da.name == "temp"

    @given(
        data_array_schema_strategy(
            DataArraySchema(
                dtype="float64",
                dims=("x",),
                coords={"x": Coordinate(dtype="float64")},
            )
        )
    )
    @settings(max_examples=5)
    def test_with_coords(self, da):
        assert "x" in da.coords

    @given(
        data_array_schema_strategy(
            DataArraySchema(
                dtype="int64",
                dims=("x",),
            ),
            size=10,
        )
    )
    @settings(max_examples=5)
    def test_custom_size(self, da):
        assert da.sizes["x"] == 10


class TestDatasetSchemaStrategy:
    @given(
        dataset_schema_strategy(
            DatasetSchema(
                data_vars={
                    "a": DataVar(dtype="float64", dims=("x",)),
                    "b": DataVar(dtype="int32", dims=("x", "y")),
                },
                sizes={"x": 3, "y": 4},
            )
        )
    )
    @settings(max_examples=5)
    def test_from_schema(self, ds):
        assert isinstance(ds, xr.Dataset)
        assert "a" in ds.data_vars
        assert "b" in ds.data_vars
        assert ds["a"].dims == ("x",)
        assert ds["b"].dims == ("x", "y")

    @given(
        dataset_schema_strategy(
            DatasetSchema(
                data_vars={"v": DataVar(dtype="float64", dims=("x",))},
                coords={"x": Coordinate(dtype="float64")},
                sizes={"x": 5},
            )
        )
    )
    @settings(max_examples=5)
    def test_with_coords(self, ds):
        assert "x" in ds.coords

    @given(
        dataset_schema_strategy(
            DatasetSchema(
                data_vars={"v": None},
            ),
            size=4,
        )
    )
    @settings(max_examples=5)
    def test_none_data_var(self, ds):
        assert "v" in ds.data_vars
