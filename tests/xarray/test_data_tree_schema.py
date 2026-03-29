"""Tests for DataTreeSchema."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

import pandera.errors  # noqa: E402
from pandera.xarray import (  # noqa: E402
    Coordinate,
    DatasetSchema,
    DataTreeSchema,
    DataVar,
)


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


@pytest.fixture
def nested_tree():
    ds_root = xr.Dataset(attrs={"conventions": "CF-1.8"})
    ds_surface = xr.Dataset(
        {"temperature": (("x",), np.ones(3, dtype=np.float64))},
        coords={"x": np.arange(3, dtype=np.float64)},
    )
    ds_diag = xr.Dataset(
        {"rmse": (("x",), np.ones(3, dtype=np.float64))},
        coords={"x": np.arange(3, dtype=np.float64)},
    )
    return xr.DataTree.from_dict(
        {
            "/": ds_root,
            "/surface": ds_surface,
            "/surface/diagnostics": ds_diag,
        }
    )


# ===================================================================
# Basic validation
# ===================================================================


class TestBasicValidation:
    def test_empty_schema_passes(self, simple_tree):
        schema = DataTreeSchema()
        schema.validate(simple_tree)

    def test_root_attrs(self, simple_tree):
        schema = DataTreeSchema(
            attrs={"conventions": "CF-1.8"},
        )
        schema.validate(simple_tree)

    def test_root_attrs_fail(self, simple_tree):
        schema = DataTreeSchema(
            attrs={"conventions": "wrong"},
        )
        with pytest.raises(pandera.errors.SchemaError):
            schema.validate(simple_tree)

    def test_child_dataset_schema(self, simple_tree):
        schema = DataTreeSchema(
            children={
                "surface": DatasetSchema(
                    data_vars={
                        "temperature": DataVar(
                            dtype=np.float64, dims=("x",)
                        ),
                    },
                ),
            },
        )
        schema.validate(simple_tree)

    def test_child_missing_raises(self, simple_tree):
        schema = DataTreeSchema(
            children={
                "nonexistent": DatasetSchema(),
            },
        )
        with pytest.raises(pandera.errors.SchemaError):
            schema.validate(simple_tree)

    def test_child_wrong_dtype(self, simple_tree):
        schema = DataTreeSchema(
            children={
                "surface": DatasetSchema(
                    data_vars={
                        "temperature": DataVar(
                            dtype=np.int64, dims=("x",)
                        ),
                    },
                ),
            },
        )
        with pytest.raises(pandera.errors.SchemaError):
            schema.validate(simple_tree)


# ===================================================================
# Path-based children
# ===================================================================


class TestPathBasedChildren:
    def test_slash_separated_path(self, nested_tree):
        schema = DataTreeSchema(
            children={
                "surface/diagnostics": DatasetSchema(
                    data_vars={
                        "rmse": DataVar(
                            dtype=np.float64, dims=("x",)
                        ),
                    },
                ),
            },
        )
        schema.validate(nested_tree)

    def test_path_to_nonexistent_node(self, nested_tree):
        schema = DataTreeSchema(
            children={
                "surface/missing": DatasetSchema(),
            },
        )
        with pytest.raises(pandera.errors.SchemaError):
            schema.validate(nested_tree)


# ===================================================================
# Strict mode
# ===================================================================


class TestStrictMode:
    def test_strict_fails_on_extra_children(self, simple_tree):
        schema = DataTreeSchema(
            children={
                "surface": DatasetSchema(),
            },
            strict=True,
        )
        with pytest.raises(pandera.errors.SchemaError):
            schema.validate(simple_tree)

    def test_strict_passes_with_all_children(self, simple_tree):
        schema = DataTreeSchema(
            children={
                "surface": DatasetSchema(),
                "upper": DatasetSchema(),
            },
            strict=True,
        )
        schema.validate(simple_tree)

    def test_non_strict_allows_extra_children(self, simple_tree):
        schema = DataTreeSchema(
            children={
                "surface": DatasetSchema(),
            },
            strict=False,
        )
        schema.validate(simple_tree)


# ===================================================================
# Nested DataTreeSchema children
# ===================================================================


class TestNestedTreeSchema:
    def test_nested_datatree_schema(self, nested_tree):
        schema = DataTreeSchema(
            attrs={"conventions": "CF-1.8"},
            children={
                "surface": DataTreeSchema(
                    children={
                        "diagnostics": DatasetSchema(
                            data_vars={
                                "rmse": DataVar(
                                    dtype=np.float64, dims=("x",)
                                ),
                            },
                        ),
                    },
                    dataset=DatasetSchema(
                        data_vars={
                            "temperature": DataVar(
                                dtype=np.float64, dims=("x",)
                            ),
                        },
                    ),
                ),
            },
        )
        schema.validate(nested_tree)


# ===================================================================
# Lazy validation
# ===================================================================


class TestLazyValidation:
    def test_lazy_collects_errors(self, simple_tree):
        schema = DataTreeSchema(
            attrs={"missing_key": "val"},
            children={
                "nonexistent": DatasetSchema(),
            },
        )
        with pytest.raises(pandera.errors.SchemaErrors) as exc_info:
            schema.validate(simple_tree, lazy=True)
        assert len(exc_info.value.schema_errors) >= 2


# ===================================================================
# Combined root dataset + children
# ===================================================================


class TestRootDataset:
    def test_root_dataset_validation(self, simple_tree):
        schema = DataTreeSchema(
            dataset=DatasetSchema(
                attrs={"conventions": "CF-1.8"},
            ),
            children={
                "surface": DatasetSchema(
                    data_vars={
                        "temperature": DataVar(
                            dtype=np.float64, dims=("x",)
                        ),
                    },
                ),
            },
        )
        schema.validate(simple_tree)

    def test_root_dataset_attrs_fail(self, simple_tree):
        schema = DataTreeSchema(
            dataset=DatasetSchema(
                attrs={"conventions": "wrong"},
            ),
        )
        with pytest.raises(pandera.errors.SchemaError):
            schema.validate(simple_tree)
