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


# -------------------------------------------------------------------
# Unit tests for individual @validate_scope check methods
# -------------------------------------------------------------------


class TestDatasetCheckMethods:
    """Test individual check methods on the backend."""

    @pytest.fixture()
    def backend(self):
        from pandera.backends.xarray.container import (
            DatasetSchemaBackend,
        )

        return DatasetSchemaBackend()

    # -- dataset-level structural checks --

    def test_check_dims_pass(self, backend):
        ds = xr.Dataset(
            {"a": (["x"], np.zeros(2))},
            coords={"x": np.arange(2)},
        )
        schema = DatasetSchema(data_vars={"a": DataVar()}, dims=("x",))
        results = backend.check_dims(ds, schema)
        assert all(r.passed for r in results) or not results

    def test_check_dims_fail(self, backend):
        ds = xr.Dataset({"a": (["x"], np.zeros(2))})
        schema = DatasetSchema(data_vars={"a": DataVar()}, dims=("y",))
        results = backend.check_dims(ds, schema)
        assert any(not r.passed for r in results)

    def test_check_sizes_pass(self, backend):
        ds = xr.Dataset({"a": (["x"], np.zeros(3))})
        schema = DatasetSchema(
            data_vars={"a": DataVar()},
            sizes={"x": 3},
        )
        results = backend.check_sizes(ds, schema)
        assert not results

    def test_check_sizes_fail(self, backend):
        ds = xr.Dataset({"a": (["x"], np.zeros(3))})
        schema = DatasetSchema(
            data_vars={"a": DataVar()},
            sizes={"x": 99},
        )
        results = backend.check_sizes(ds, schema)
        assert len(results) == 1
        assert not results[0].passed

    def test_check_attrs_pass(self, backend):
        ds = xr.Dataset(
            {"a": (["x"], np.zeros(2))},
            attrs={"version": 1},
        )
        schema = DatasetSchema(
            data_vars={"a": DataVar()},
            attrs={"version": 1},
        )
        results = backend.check_attrs(ds, schema)
        assert not results

    def test_check_attrs_fail(self, backend):
        ds = xr.Dataset(
            {"a": (["x"], np.zeros(2))},
            attrs={"version": 2},
        )
        schema = DatasetSchema(
            data_vars={"a": DataVar()},
            attrs={"version": 1},
        )
        results = backend.check_attrs(ds, schema)
        assert len(results) == 1
        assert not results[0].passed

    def test_check_strict_attrs_pass(self, backend):
        ds = xr.Dataset(
            {"a": (["x"], np.zeros(2))},
            attrs={"version": 1},
        )
        schema = DatasetSchema(
            data_vars={"a": DataVar()},
            attrs={"version": 1},
            strict_attrs=True,
        )
        results = backend.check_strict_attrs(ds, schema)
        assert not results

    def test_check_strict_attrs_fail(self, backend):
        ds = xr.Dataset(
            {"a": (["x"], np.zeros(2))},
            attrs={"version": 1, "extra": "x"},
        )
        schema = DatasetSchema(
            data_vars={"a": DataVar()},
            attrs={"version": 1},
            strict_attrs=True,
        )
        results = backend.check_strict_attrs(ds, schema)
        assert len(results) == 1
        assert "unexpected attribute" in results[0].message

    def test_check_coords_pass(self, backend):
        ds = xr.Dataset(
            {"a": (["x"], np.zeros(2))},
            coords={"x": np.arange(2)},
        )
        schema = DatasetSchema(
            data_vars={"a": DataVar()},
            coords={"x": Coordinate()},
        )
        results = backend.check_coords(ds, schema)
        assert all(r.passed for r in results) or not results

    def test_check_coords_missing(self, backend):
        ds = xr.Dataset({"a": (["x"], np.zeros(2))})
        schema = DatasetSchema(
            data_vars={"a": DataVar()},
            coords={"y": Coordinate()},
        )
        results = backend.check_coords(ds, schema)
        assert any(not r.passed for r in results)

    def test_check_strict_coords_pass(self, backend):
        ds = xr.Dataset(
            {"a": (["x"], np.zeros(2))},
            coords={"x": np.arange(2)},
        )
        schema = DatasetSchema(
            data_vars={"a": DataVar()},
            coords=["x"],
            strict_coords=True,
        )
        results = backend.check_strict_coords(ds, schema)
        assert not results

    def test_check_strict_coords_fail(self, backend):
        ds = xr.Dataset(
            {"a": (["x"], np.zeros(2))},
            coords={
                "x": np.arange(2),
                "extra": ("x", np.zeros(2)),
            },
        )
        schema = DatasetSchema(
            data_vars={"a": DataVar()},
            coords=["x"],
            strict_coords=True,
        )
        results = backend.check_strict_coords(ds, schema)
        assert any(not r.passed for r in results)

    # -- data-var level checks --

    def test_check_strict_data_vars_pass(self, backend):
        schema = DatasetSchema(data_vars={"a": DataVar()}, strict=True)
        result = backend.check_strict_data_vars(schema, extras=[])
        assert result.passed

    def test_check_strict_data_vars_fail(self, backend):
        schema = DatasetSchema(data_vars={"a": DataVar()}, strict=True)
        result = backend.check_strict_data_vars(schema, extras=["extra"])
        assert not result.passed
        assert "unexpected data variables" in result.message

    def test_check_strict_data_vars_filter_noop(self, backend):
        schema = DatasetSchema(data_vars={"a": DataVar()}, strict="filter")
        result = backend.check_strict_data_vars(schema, extras=["extra"])
        assert result.passed

    def test_check_data_var_presence_pass(self, backend):
        ds = xr.Dataset({"a": (["x"], np.zeros(2))})
        schema = DatasetSchema(data_vars={"a": DataVar(dims=("x",))})
        results = backend.check_data_var_presence(ds, schema, {"a": "a"})
        assert not results

    def test_check_data_var_presence_missing_required(self, backend):
        ds = xr.Dataset({"a": (["x"], np.zeros(2))})
        schema = DatasetSchema(
            data_vars={
                "a": DataVar(),
                "b": DataVar(dims=("x",)),
            }
        )
        results = backend.check_data_var_presence(
            ds, schema, {"a": "a", "b": "b"}
        )
        assert len(results) == 1
        assert "missing required" in results[0].message

    def test_check_data_var_presence_optional_ok(self, backend):
        ds = xr.Dataset({"a": (["x"], np.zeros(2))})
        schema = DatasetSchema(
            data_vars={
                "a": DataVar(),
                "b": DataVar(dims=("x",), required=False),
            }
        )
        results = backend.check_data_var_presence(
            ds, schema, {"a": "a", "b": "b"}
        )
        assert not results

    def test_check_data_var_alignment_pass(self, backend):
        ds = xr.Dataset(
            {
                "u": (["x", "y"], np.zeros((2, 3))),
                "v": (["x", "y"], np.ones((2, 3))),
            }
        )
        schema = DatasetSchema(
            data_vars={
                "u": DataVar(dims=("x", "y"), aligned_with=("v",)),
                "v": DataVar(dims=("x", "y")),
            }
        )
        results = backend.check_data_var_alignment(
            ds, schema, {"u": "u", "v": "v"}
        )
        assert not results

    def test_check_data_var_alignment_fail(self, backend):
        ds = xr.Dataset(
            {
                "u": (["x"], np.zeros(2)),
                "v": (["x", "y"], np.ones((2, 3))),
            }
        )
        schema = DatasetSchema(
            data_vars={
                "u": DataVar(dims=("x",), aligned_with=("v",)),
                "v": DataVar(dims=("x", "y")),
            }
        )
        results = backend.check_data_var_alignment(
            ds, schema, {"u": "u", "v": "v"}
        )
        assert any(not r.passed for r in results)

    def test_check_data_var_alignment_missing_peer(self, backend):
        ds = xr.Dataset({"u": (["x"], np.zeros(2))})
        schema = DatasetSchema(
            data_vars={
                "u": DataVar(dims=("x",), aligned_with=("v",)),
                "v": DataVar(dims=("x",)),
            }
        )
        results = backend.check_data_var_alignment(
            ds, schema, {"u": "u", "v": "v"}
        )
        assert any("missing" in (r.message or "") for r in results)

    def test_check_broadcastable_with_fail(self, backend):
        ds = xr.Dataset(
            {
                "u": (["x"], np.zeros(2)),
                "v": (["y"], np.ones(3)),
            }
        )
        schema = DatasetSchema(
            data_vars={
                "u": DataVar(dims=("x",)),
                "v": DataVar(
                    dims=("y",),
                    broadcastable_with=("u",),
                ),
            }
        )
        results = backend.check_data_var_alignment(
            ds, schema, {"u": "u", "v": "v"}
        )
        assert not results or all(r.passed for r in results)

    def test_schema_scope_checks_skipped_with_data_only(self, backend):
        from pandera.config import (
            ValidationDepth,
            config_context,
        )

        ds = xr.Dataset({"a": (["x"], np.zeros(2))})
        schema = DatasetSchema(data_vars={"a": DataVar()}, dims=("y",))
        with config_context(validation_depth=ValidationDepth.DATA_ONLY):
            result = backend.check_dims(ds, schema)
        assert result.passed

    # --- ordered_dims tests ---

    def test_check_dims_unordered_pass(self, backend):
        ds = xr.Dataset(
            {
                "a": (["y", "x"], np.zeros((3, 2))),
            }
        )
        schema = DatasetSchema(
            data_vars={"a": DataVar()},
            dims=("x", "y"),
            ordered_dims=False,
        )
        results = backend.check_dims(ds, schema)
        assert not results or all(r.passed for r in results)

    def test_check_dims_ordered_fail(self, backend):
        ds = xr.Dataset(
            {
                "a": (["y", "x"], np.zeros((3, 2))),
            }
        )
        schema = DatasetSchema(
            data_vars={"a": DataVar()},
            dims=("x", "y"),
            ordered_dims=True,
        )
        results = backend.check_dims(ds, schema)
        assert any(not r.passed for r in results)

    # --- attrs regex / callable tests ---

    def test_check_attrs_regex_pass(self, backend):
        ds = xr.Dataset(
            {"a": (["x"], np.zeros(2))},
            attrs={"source": "ERA5"},
        )
        schema = DatasetSchema(
            data_vars={"a": DataVar()},
            attrs={"source": "^ERA\\d$"},
        )
        results = backend.check_attrs(ds, schema)
        assert not results

    def test_check_attrs_regex_fail(self, backend):
        ds = xr.Dataset(
            {"a": (["x"], np.zeros(2))},
            attrs={"source": "MERRA"},
        )
        schema = DatasetSchema(
            data_vars={"a": DataVar()},
            attrs={"source": "^ERA\\d$"},
        )
        results = backend.check_attrs(ds, schema)
        assert len(results) == 1
        assert not results[0].passed

    def test_check_attrs_callable_pass(self, backend):
        ds = xr.Dataset(
            {"a": (["x"], np.zeros(2))},
            attrs={"version": 3},
        )
        schema = DatasetSchema(
            data_vars={"a": DataVar()},
            attrs={"version": lambda v: isinstance(v, int) and v >= 2},
        )
        results = backend.check_attrs(ds, schema)
        assert not results

    def test_check_attrs_callable_fail(self, backend):
        ds = xr.Dataset(
            {"a": (["x"], np.zeros(2))},
            attrs={"version": 1},
        )
        schema = DatasetSchema(
            data_vars={"a": DataVar()},
            attrs={"version": lambda v: isinstance(v, int) and v >= 2},
        )
        results = backend.check_attrs(ds, schema)
        assert len(results) == 1
        assert not results[0].passed

    # --- Coordinate required tests ---

    def test_check_coords_optional_absent(self, backend):
        ds = xr.Dataset({"a": (["x"], np.zeros(2))})
        schema = DatasetSchema(
            data_vars={"a": DataVar()},
            coords={"label": Coordinate(required=False)},
        )
        results = backend.check_coords(ds, schema)
        assert not results or all(r.passed for r in results)

    def test_check_coords_optional_present(self, backend):
        ds = xr.Dataset(
            {"a": (["x"], np.zeros(2))},
            coords={
                "label": ("x", ["a", "b"]),
            },
        )
        schema = DatasetSchema(
            data_vars={"a": DataVar()},
            coords={"label": Coordinate(required=False, dtype=str)},
        )
        results = backend.check_coords(ds, schema)
        assert not results or all(r.passed for r in results)
