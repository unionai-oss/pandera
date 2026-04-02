"""Tests for xarray schema inference."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pandera.xarray as pa
from pandera.api.xarray.components import Coordinate, DataVar
from pandera.api.xarray.container import DataArraySchema, DatasetSchema
from pandera.schema_inference.xarray import (
    infer_data_array_schema,
    infer_dataset_schema,
    infer_schema,
)


class TestInferDataArraySchema:
    def test_basic_float(self):
        da = xr.DataArray(
            np.array([1.0, 2.0, 3.0]),
            dims=("x",),
            name="temperature",
        )
        schema = infer_data_array_schema(da)
        assert isinstance(schema, DataArraySchema)
        assert schema.dims == ("x",)
        assert schema.name == "temperature"
        assert schema.nullable is False
        assert schema.coerce is True

    def test_infers_dtype(self):
        da = xr.DataArray(np.array([1, 2, 3], dtype=np.int32), dims=("x",))
        schema = infer_data_array_schema(da)
        assert "int32" in str(schema.dtype)

    def test_nullable_detection(self):
        da = xr.DataArray(np.array([1.0, np.nan, 3.0]), dims=("x",))
        schema = infer_data_array_schema(da)
        assert schema.nullable is True

    def test_multidim(self):
        da = xr.DataArray(
            np.zeros((3, 4)),
            dims=("y", "x"),
        )
        schema = infer_data_array_schema(da)
        assert schema.dims == ("y", "x")

    def test_coords_inferred(self):
        da = xr.DataArray(
            np.zeros((3,)),
            dims=("x",),
            coords={"x": [10.0, 20.0, 30.0]},
        )
        schema = infer_data_array_schema(da)
        assert schema.coords is not None
        assert "x" in schema.coords
        assert isinstance(schema.coords["x"], Coordinate)  # type: ignore[index]

    def test_check_statistics_for_numeric(self):
        da = xr.DataArray(np.array([5.0, 10.0, 15.0]), dims=("x",))
        schema = infer_data_array_schema(da)
        assert schema.checks is not None
        check_names = [c.name for c in schema.checks]
        assert "greater_than_or_equal_to" in check_names
        assert "less_than_or_equal_to" in check_names

    def test_validates_with_inferred_schema(self):
        da = xr.DataArray(
            np.array([1.0, 2.0, 3.0]),
            dims=("x",),
            coords={"x": [0.0, 1.0, 2.0]},
            name="vals",
        )
        schema = infer_schema(da)
        result = schema.validate(da)
        assert isinstance(result, xr.DataArray)


class TestInferDatasetSchema:
    def test_basic(self):
        ds = xr.Dataset(
            {
                "temperature": (("x", "y"), np.zeros((3, 4))),
                "pressure": (("x", "y"), np.ones((3, 4))),
            }
        )
        schema = infer_dataset_schema(ds)
        assert isinstance(schema, DatasetSchema)
        assert schema.data_vars is not None
        assert "temperature" in schema.data_vars
        assert "pressure" in schema.data_vars

    def test_data_var_dims(self):
        ds = xr.Dataset(
            {
                "a": (("x",), [1.0, 2.0]),
                "b": (("x", "y"), [[1.0, 2.0], [3.0, 4.0]]),
            }
        )
        schema = infer_dataset_schema(ds)
        a_spec = schema.data_vars["a"]
        b_spec = schema.data_vars["b"]
        assert isinstance(a_spec, DataVar)
        assert isinstance(b_spec, DataVar)
        assert a_spec.dims == ("x",)
        assert b_spec.dims == ("x", "y")

    def test_coords_inferred(self):
        ds = xr.Dataset(
            {"var": (("x",), [1.0, 2.0, 3.0])},
            coords={"x": [10.0, 20.0, 30.0]},
        )
        schema = infer_dataset_schema(ds)
        assert schema.coords is not None
        assert "x" in schema.coords

    def test_sizes_inferred(self):
        ds = xr.Dataset({"v": (("a", "b"), np.zeros((2, 5)))})
        schema = infer_dataset_schema(ds)
        assert schema.sizes == {"a": 2, "b": 5}

    def test_validates_with_inferred_schema(self):
        ds = xr.Dataset(
            {
                "temp": (("x",), [1.0, 2.0, 3.0]),
            },
            coords={"x": [0.0, 1.0, 2.0]},
        )
        schema = infer_schema(ds)
        result = schema.validate(ds)
        assert result is not None


class TestInferSchemaDispatch:
    def test_dispatch_data_array(self):
        da = xr.DataArray(np.zeros(3), dims=("x",))
        schema = infer_schema(da)
        assert isinstance(schema, DataArraySchema)

    def test_dispatch_dataset(self):
        ds = xr.Dataset({"v": (("x",), [1.0])})
        schema = infer_schema(ds)
        assert isinstance(schema, DatasetSchema)

    def test_dispatch_unknown_raises(self):
        with pytest.raises(TypeError, match="not recognized"):
            infer_schema("not_xarray")  # type: ignore[arg-type]


class TestInferSchemaViaEntryPoint:
    def test_via_pa_module(self):
        da = xr.DataArray(np.array([1.0, 2.0]), dims=("x",))
        schema = pa.infer_schema(da)
        assert isinstance(schema, DataArraySchema)

    def test_via_pa_module_dataset(self):
        ds = xr.Dataset({"a": (("x",), [1.0, 2.0])})
        schema = pa.infer_schema(ds)
        assert isinstance(schema, DatasetSchema)
