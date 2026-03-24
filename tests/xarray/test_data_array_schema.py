"""Tests for DataArraySchema."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

import pandera.errors
from pandera.xarray import Check, Coordinate, DataArraySchema


def test_dtype_and_dims_pass():
    da = xr.DataArray(
        np.zeros((2, 3)),
        dims=("x", "y"),
        name="a",
    )
    schema = DataArraySchema(
        dtype=np.float64,
        dims=("x", "y"),
        name="a",
    )
    out = schema.validate(da)
    assert out.identical(da)


def test_dims_mismatch_raises():
    da = xr.DataArray(np.zeros((2, 3)), dims=("x", "y"))
    schema = DataArraySchema(dims=("x", "z"))
    with pytest.raises(pandera.errors.SchemaError):
        schema.validate(da)


def test_sizes():
    da = xr.DataArray(np.zeros((2, 3)), dims=("x", "y"))
    schema = DataArraySchema(sizes={"x": 2, "y": 3})
    schema.validate(da)
    bad = DataArraySchema(sizes={"x": 99})
    with pytest.raises(pandera.errors.SchemaError):
        bad.validate(da)


def test_coordinate_dimension_and_indexed():
    da = xr.DataArray(
        np.arange(3.0),
        dims="t",
        coords={
            "t": ("t", np.arange(3)),
            "aux": ("t", np.array([1.0, 2.0, 3.0])),
        },
    )
    schema = DataArraySchema(
        coords={
            "t": Coordinate(dimension=True, indexed=True),
            "aux": Coordinate(dimension=False, indexed=False),
        },
    )
    schema.validate(da)


def test_strict_coords():
    da = xr.DataArray(
        np.zeros((2,)),
        dims=("x",),
        coords={"x": ("x", np.arange(2))},
    )
    schema = DataArraySchema(coords=["x"], strict_coords=True)
    schema.validate(da)
    schema_bad = DataArraySchema(
        coords=["x"],
        strict_coords=True,
    )
    da_extra = da.assign_coords(extra=("x", np.zeros(2)))
    with pytest.raises(pandera.errors.SchemaError):
        schema_bad.validate(da_extra)


def test_data_check_and_lazy():
    da = xr.DataArray(np.array([1.0, 5.0, 3.0]), dims=("x",))
    schema = DataArraySchema(checks=Check(lambda x: x.max() < 4))
    with pytest.raises(pandera.errors.SchemaError):
        schema.validate(da)
    with pytest.raises(pandera.errors.SchemaErrors):
        schema.validate(da, lazy=True)


def test_schema_only_skips_data_checks():
    da = xr.DataArray(np.array([1.0, 99.0]), dims=("x",))
    schema = DataArraySchema(checks=Check(lambda x: x.max() < 4))
    from pandera.config import ValidationDepth, config_context

    with config_context(validation_depth=ValidationDepth.SCHEMA_ONLY):
        schema.validate(da)


def test_coerce_dtype():
    da = xr.DataArray(np.array([1.0, 2.0]), dims=("x",))
    schema = DataArraySchema(dtype=np.int64, coerce=True)
    out = schema.validate(da)
    assert np.issubdtype(out.dtype, np.integer)


def test_sizes_and_shape_mutually_exclusive():
    from pandera.errors import SchemaDefinitionError

    with pytest.raises(SchemaDefinitionError):
        DataArraySchema(sizes={"x": 1}, shape=(1,))


def test_nullable_rejects_nan():
    da = xr.DataArray(np.array([1.0, np.nan]), dims=("x",))
    schema = DataArraySchema(nullable=False)
    with pytest.raises(pandera.errors.SchemaError):
        schema.validate(da)


def test_nullable_true_allows_nan():
    da = xr.DataArray(np.array([1.0, np.nan]), dims=("x",))
    schema = DataArraySchema(nullable=True)
    schema.validate(da)


def test_in_range_ignore_na_skips_nan_cells():
    da = xr.DataArray(np.array([1.0, np.nan, 2.0]), dims=("x",))
    # nulls allowed at schema level; ignore_na only affects the in_range mask.
    schema = DataArraySchema(
        nullable=True,
        checks=Check.in_range(0, 3, ignore_na=True),
    )
    schema.validate(da)
    strict = DataArraySchema(
        nullable=True,
        checks=Check.in_range(0, 3, ignore_na=False),
    )
    with pytest.raises(pandera.errors.SchemaError):
        strict.validate(da)


def test_strict_attrs():
    da = xr.DataArray(np.zeros(2), dims="x", attrs={"a": 1, "b": 2})
    schema = DataArraySchema(attrs={"a": 1}, strict_attrs=True)
    with pytest.raises(pandera.errors.SchemaError):
        schema.validate(da)


def test_head_subsample_runs_checks_on_subset():
    da = xr.DataArray(np.arange(10.0), dims=("x",))
    schema = DataArraySchema(
        checks=Check(lambda z: float(z.max()) < 5),
    )
    schema.validate(da, head=5)


def test_head_on_zero_dim_array_is_noop_for_subsample():
    da = xr.DataArray(np.float64(42.0))
    schema = DataArraySchema(checks=Check(lambda z: float(z) == 42.0))
    schema.validate(da, head=3)


def test_in_range_builtin():
    da = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims=("x",))
    schema = DataArraySchema(
        checks=Check.in_range(0, 4),
    )
    schema.validate(da)


def test_lazy_schema_errors_multi_message():
    da = xr.DataArray(np.array([1.0, 5.0]), dims=("x",))
    schema = DataArraySchema(
        checks=[
            Check(lambda x: x.max() < 4),
            Check(lambda x: x.min() > 2),
        ],
    )
    with pytest.raises(pandera.errors.SchemaErrors) as excinfo:
        schema.validate(da, lazy=True)
    assert len(excinfo.value.schema_errors) >= 2


def test_parser_runs():
    from pandera.api.parsers import Parser

    da = xr.DataArray(np.array([1.0, 2.0]), dims=("x",))
    schema = DataArraySchema(parsers=Parser(lambda x: x * 2))
    out = schema.validate(da)
    assert float(out.max().item()) == 4.0


def test_chunked_data_checks_skipped_by_default():
    pytest.importorskip("dask.array")
    import dask.array as dda

    da = xr.DataArray(dda.ones(4, chunks=2), dims="x")
    schema = DataArraySchema(
        checks=Check(lambda x: False),
    )
    schema.validate(da)


def test_chunked_data_checks_when_validation_depth_includes_data():
    pytest.importorskip("dask.array")
    import dask.array as dda

    from pandera.config import ValidationDepth, config_context

    da = xr.DataArray(dda.ones(4, chunks=2), dims="x")
    schema = DataArraySchema(
        checks=Check(lambda x: False),
    )
    with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
        with pytest.raises(pandera.errors.SchemaErrors):
            schema.validate(da, lazy=True)
