"""Tests for DataTreeModel."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

import pandera.errors  # noqa: E402
import pandera.xarray as pa  # noqa: E402
from pandera.typing.xarray import Coordinate, DataTree  # noqa: E402


class SurfaceModel(pa.DatasetModel):
    temperature: np.float64 = pa.Field(dims=("x",))
    x: Coordinate[np.float64]


class UpperModel(pa.DatasetModel):
    wind: np.float64 = pa.Field(dims=("x",))
    x: Coordinate[np.float64]


@pytest.fixture
def simple_tree():
    ds_root = xr.Dataset(attrs={"conventions": "CF-1.8"})
    ds_surface = xr.Dataset(
        {"temperature": (("x",), np.ones(3, dtype=np.float64))},
        coords={"x": np.arange(3, dtype=np.float64)},
    )
    ds_upper = xr.Dataset(
        {"wind": (("x",), np.ones(3, dtype=np.float64))},
        coords={"x": np.arange(3, dtype=np.float64)},
    )
    return xr.DataTree.from_dict(
        {
            "/": ds_root,
            "/surface": ds_surface,
            "/upper": ds_upper,
        }
    )


# ===================================================================
# DataTreeModel basic usage
# ===================================================================


class TestDataTreeModel:
    def test_to_schema(self):
        class ClimateTree(pa.DataTreeModel):
            surface: SurfaceModel
            upper: UpperModel

        schema = ClimateTree.to_schema()
        assert isinstance(schema, pa.DataTreeSchema)
        assert "surface" in schema.children
        assert "upper" in schema.children

    def test_validate(self, simple_tree):
        class ClimateTree(pa.DataTreeModel):
            surface: SurfaceModel
            upper: UpperModel

        ClimateTree.validate(simple_tree)

    def test_field_name_access(self):
        class ClimateTree(pa.DataTreeModel):
            surface: SurfaceModel
            upper: UpperModel

        assert ClimateTree.surface == "surface"
        assert ClimateTree.upper == "upper"

    def test_strict_mode(self, simple_tree):
        class StrictTree(pa.DataTreeModel):
            surface: SurfaceModel

            class Config:
                strict = True

        with pytest.raises(pandera.errors.SchemaError):
            StrictTree.validate(simple_tree)

    def test_strict_all_children(self, simple_tree):
        class StrictTree(pa.DataTreeModel):
            surface: SurfaceModel
            upper: UpperModel

            class Config:
                strict = True

        StrictTree.validate(simple_tree)

    def test_validation_error(self, simple_tree):
        class BadTree(pa.DataTreeModel):
            surface: UpperModel  # temperature != wind

        with pytest.raises(pandera.errors.SchemaError):
            BadTree.validate(simple_tree)

    def test_config_attrs(self, simple_tree):
        class Tree(pa.DataTreeModel):
            surface: SurfaceModel

            class Config:
                attrs = {"conventions": "CF-1.8"}

        Tree.validate(simple_tree)

    def test_config_attrs_fail(self, simple_tree):
        class Tree(pa.DataTreeModel):
            surface: SurfaceModel

            class Config:
                attrs = {"conventions": "wrong"}

        with pytest.raises(pandera.errors.SchemaError):
            Tree.validate(simple_tree)


# ===================================================================
# check_types with DataTree
# ===================================================================


class TestCheckTypesDataTree:
    def test_check_types_data_tree(self, simple_tree):
        class Tree(pa.DataTreeModel):
            surface: SurfaceModel
            upper: UpperModel

        @pa.check_types
        def process(dt: DataTree[Tree]) -> DataTree[Tree]:
            return dt

        out = process(simple_tree)
        assert out is not None

    def test_check_types_invalid_input(self):
        class Tree(pa.DataTreeModel):
            surface: SurfaceModel

        bad_tree = xr.DataTree.from_dict(
            {"/": xr.Dataset(), "/wrong": xr.Dataset({"a": (("x",), [1])})}
        )

        @pa.check_types
        def process(dt: DataTree[Tree]) -> DataTree[Tree]:
            return dt

        with pytest.raises(pandera.errors.SchemaError):
            process(bad_tree)
