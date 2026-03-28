"""Tests for DatasetSchema."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

import pandera.errors
from pandera.xarray import Check, Coordinate, DatasetSchema, DataVar


def test_data_vars_and_coords():
    ds = xr.Dataset(
        {
            "a": (["x"], np.zeros(3)),
            "b": (["x"], np.ones(3)),
        },
        coords={"x": np.arange(3)},
    )
    schema = DatasetSchema(
        data_vars={
            "a": DataVar(dtype=np.float64, dims=("x",)),
            "b": DataVar(dtype=np.float64, dims=("x",)),
        },
        coords={"x": Coordinate()},
        dims=("x",),
        sizes={"x": 3},
    )
    schema.validate(ds)


def test_optional_default_fill():
    ds = xr.Dataset({"a": (["x"], np.zeros(2))}, coords={"x": [0, 1]})
    schema = DatasetSchema(
        data_vars={
            "a": DataVar(dims=("x",)),
            "b": DataVar(
                dtype=np.float64,
                dims=("x",),
                required=False,
                default=2.0,
            ),
        },
    )
    out = schema.validate(ds)
    assert "b" in out.data_vars
    assert (out["b"] == 2.0).all()


def test_aligned_with():
    ds = xr.Dataset(
        {
            "u": (["x", "y"], np.zeros((2, 3))),
            "v": (["x", "y"], np.ones((2, 3))),
        },
    )
    schema = DatasetSchema(
        data_vars={
            "u": DataVar(dims=("x", "y"), aligned_with=("v",)),
            "v": DataVar(dims=("x", "y")),
        },
    )
    schema.validate(ds)


def test_broadcastable_with():
    ds = xr.Dataset(
        {
            "u": (["x"], np.zeros(2)),
            "v": (["x", "y"], np.zeros((2, 1))),
        },
    )
    schema = DatasetSchema(
        data_vars={
            "u": DataVar(dims=("x",)),
            "v": DataVar(dims=("x", "y"), broadcastable_with=("u",)),
        },
    )
    schema.validate(ds)


def test_strict_filter():
    ds = xr.Dataset({"a": (["x"], np.zeros(2)), "extra": (["x"], np.ones(2))})
    schema = DatasetSchema(
        data_vars={"a": DataVar()},
        strict="filter",
    )
    out = schema.validate(ds)
    assert set(out.data_vars) == {"a"}


def test_alias():
    ds = xr.Dataset({"temp_c": (["x"], np.zeros(2))})
    schema = DatasetSchema(
        data_vars={
            "temperature": DataVar(alias="temp_c", dims=("x",)),
        },
    )
    schema.validate(ds)


def test_duplicate_alias_raises():
    """Two logical data_vars that resolve to the same actual name."""
    ds = xr.Dataset({"temp_c": (["x"], np.zeros(2))})
    schema = DatasetSchema(
        data_vars={
            "temperature": DataVar(alias="temp_c", dims=("x",)),
            "temp_alias": DataVar(alias="temp_c", dims=("x",)),
        },
    )
    with pytest.raises(
        pandera.errors.SchemaError,
        match="multiple data_vars resolve to the same actual variable name",
    ):
        schema.validate(ds)


def test_alias_collides_with_logical_name_raises():
    """An alias that matches another logical key's resolved name."""
    ds = xr.Dataset({"temp_c": (["x"], np.zeros(2))})
    schema = DatasetSchema(
        data_vars={
            "temp_c": DataVar(dims=("x",)),
            "temperature": DataVar(alias="temp_c", dims=("x",)),
        },
    )
    with pytest.raises(
        pandera.errors.SchemaError,
        match="multiple data_vars resolve to the same actual variable name",
    ):
        schema.validate(ds)


def test_distinct_aliases_pass():
    """Different aliases should not trigger the duplicate check."""
    ds = xr.Dataset(
        {
            "temp_c": (["x"], np.zeros(2)),
            "pres_hpa": (["x"], np.ones(2)),
        }
    )
    schema = DatasetSchema(
        data_vars={
            "temperature": DataVar(alias="temp_c", dims=("x",)),
            "pressure": DataVar(alias="pres_hpa", dims=("x",)),
        },
    )
    schema.validate(ds)


def test_validate_leaves_shared_datavar_spec_unmutated():
    dv = DataVar(dtype=np.float64, dims=("x",))
    schema = DatasetSchema(
        data_vars={"logical": dv},
        coords={"x": Coordinate()},
    )
    assert dv.name is None
    ds = xr.Dataset(
        {"logical": (["x"], np.zeros(2))},
        coords={"x": [0, 1]},
    )
    schema.validate(ds)
    assert dv.name is None
    assert schema.data_vars["logical"] is dv


def test_dataset_level_check():
    ds = xr.Dataset({"a": (["x"], np.array([1.0, 2.0]))})
    schema = DatasetSchema(
        data_vars={"a": DataVar()},
        checks=Check(lambda d: d["a"].max() < 3),
    )
    schema.validate(ds)
    bad = DatasetSchema(
        data_vars={"a": DataVar()},
        checks=Check(lambda d: d["a"].max() < 1),
    )
    with pytest.raises(pandera.errors.SchemaError):
        bad.validate(ds)
