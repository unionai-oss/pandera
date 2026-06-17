"""Tests for Dask-backed (duck array) xarray validation.

Covers ``chunked``, ``array_type``, structural checks without ``.compute()``,
and configurable data-level check behavior via ``ValidationDepth``.
"""

import numpy as np
import pytest

dask = pytest.importorskip("dask")
import dask.array as dda

xr = pytest.importorskip("xarray")

import pandera.errors
from pandera.config import ValidationDepth, config_context
from pandera.xarray import (
    Check,
    Coordinate,
    DataArraySchema,
    DatasetSchema,
    DataVar,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _dask_da(name="temp", dims=("x",), values=None):
    if values is None:
        values = [1.0, 2.0, 3.0]
    return xr.DataArray(
        dda.from_array(np.array(values), chunks=2),
        dims=dims,
        name=name,
    )


def _eager_da(name="temp", dims=("x",), values=None):
    if values is None:
        values = [1.0, 2.0, 3.0]
    return xr.DataArray(np.array(values), dims=dims, name=name)


_SchemaErrors = (
    pandera.errors.SchemaError,
    pandera.errors.SchemaErrors,
)


# ==================================================================
# 1. DataArraySchema – chunked parameter
# ==================================================================


class TestChunkedParameter:
    """``chunked=True/False/None`` on DataArraySchema."""

    def test_chunked_true_passes_dask(self):
        schema = DataArraySchema(dims=("x",), name="temp", chunked=True)
        schema.validate(_dask_da())

    def test_chunked_true_fails_eager(self):
        schema = DataArraySchema(dims=("x",), name="temp", chunked=True)
        with pytest.raises(_SchemaErrors):
            schema.validate(_eager_da())

    def test_chunked_false_passes_eager(self):
        schema = DataArraySchema(dims=("x",), name="temp", chunked=False)
        schema.validate(_eager_da())

    def test_chunked_false_fails_dask(self):
        schema = DataArraySchema(dims=("x",), name="temp", chunked=False)
        with pytest.raises(_SchemaErrors):
            schema.validate(_dask_da())

    @pytest.mark.parametrize("factory", [_dask_da, _eager_da])
    def test_chunked_none_accepts_both(self, factory):
        schema = DataArraySchema(dims=("x",), name="temp", chunked=None)
        schema.validate(factory())


# ==================================================================
# 2. array_type parameter
# ==================================================================


class TestArrayType:
    """``array_type`` on DataArraySchema."""

    def test_dask_array_type_passes_dask(self):
        schema = DataArraySchema(
            dims=("x",), name="temp", array_type=dda.Array
        )
        schema.validate(_dask_da())

    def test_ndarray_type_fails_dask(self):
        schema = DataArraySchema(
            dims=("x",),
            name="temp",
            array_type=np.ndarray,
        )
        with pytest.raises(_SchemaErrors):
            schema.validate(_dask_da())

    def test_ndarray_type_passes_eager(self):
        schema = DataArraySchema(
            dims=("x",),
            name="temp",
            array_type=np.ndarray,
        )
        schema.validate(_eager_da())

    def test_dask_array_type_fails_eager(self):
        schema = DataArraySchema(
            dims=("x",), name="temp", array_type=dda.Array
        )
        with pytest.raises(_SchemaErrors):
            schema.validate(_eager_da())


# ==================================================================
# 3. Structural checks work without .compute()
# ==================================================================


class TestStructuralChecksNonCompute:
    """Schema-level checks on Dask-backed DataArrays.

    These must work without triggering ``.compute()`` because they
    inspect metadata only (dtype, dims, sizes, coords, name, attrs).
    """

    def test_dtype_check(self):
        schema = DataArraySchema(dtype=np.float64, dims=("x",), name="temp")
        schema.validate(_dask_da())

    def test_dtype_check_fails(self):
        schema = DataArraySchema(dtype=np.int32, dims=("x",), name="temp")
        with pytest.raises(_SchemaErrors):
            schema.validate(_dask_da())

    def test_dims_check(self):
        schema = DataArraySchema(dims=("x",), name="temp")
        schema.validate(_dask_da())

    def test_dims_check_fails(self):
        schema = DataArraySchema(dims=("y",), name="temp")
        with pytest.raises(_SchemaErrors):
            schema.validate(_dask_da())

    def test_sizes_check(self):
        schema = DataArraySchema(dims=("x",), name="temp", sizes={"x": 3})
        schema.validate(_dask_da())

    def test_sizes_check_fails(self):
        schema = DataArraySchema(dims=("x",), name="temp", sizes={"x": 99})
        with pytest.raises(_SchemaErrors):
            schema.validate(_dask_da())

    def test_coords_check(self):
        da = xr.DataArray(
            dda.from_array(np.ones(3), chunks=2),
            dims=("x",),
            coords={"x": np.arange(3, dtype=np.float64)},
            name="temp",
        )
        schema = DataArraySchema(
            dims=("x",),
            name="temp",
            coords={
                "x": Coordinate(dtype=np.float64, required=True),
            },
        )
        schema.validate(da)

    def test_name_check(self):
        schema = DataArraySchema(dims=("x",), name="temp")
        schema.validate(_dask_da())

    def test_name_check_fails(self):
        schema = DataArraySchema(dims=("x",), name="wrong")
        with pytest.raises(_SchemaErrors):
            schema.validate(_dask_da())

    def test_attrs_check(self):
        da = _dask_da()
        da.attrs["units"] = "K"
        schema = DataArraySchema(
            dims=("x",), name="temp", attrs={"units": "K"}
        )
        schema.validate(da)

    def test_attrs_check_fails(self):
        da = _dask_da()
        da.attrs["units"] = "C"
        schema = DataArraySchema(
            dims=("x",), name="temp", attrs={"units": "K"}
        )
        with pytest.raises(_SchemaErrors):
            schema.validate(da)


# ==================================================================
# 4. Data-level checks and validation depth
# ==================================================================


class TestDaskValidationDepth:
    """Dask arrays default to SCHEMA_ONLY; data checks need opt-in."""

    def test_default_depth_skips_data_checks(self):
        """By default, Dask-backed arrays skip user checks."""
        schema = DataArraySchema(
            dims=("x",),
            name="temp",
            checks=Check.ge(0),
        )
        da = _dask_da(values=[-1.0, -2.0, -3.0])
        schema.validate(da)

    def test_schema_and_data_runs_data_checks(self):
        """Opt-in SCHEMA_AND_DATA executes user checks."""
        schema = DataArraySchema(
            dims=("x",),
            name="temp",
            checks=Check.ge(0),
        )
        da = _dask_da(values=[-1.0, -2.0, -3.0])
        with config_context(
            validation_depth=ValidationDepth.SCHEMA_AND_DATA,
        ):
            with pytest.raises(_SchemaErrors):
                schema.validate(da)

    def test_user_check_skipped_schema_only(self):
        """A lambda check is skipped under SCHEMA_ONLY."""
        schema = DataArraySchema(
            dims=("x",),
            name="temp",
            checks=Check(lambda x: False, error="always fails"),
        )
        with config_context(
            validation_depth=ValidationDepth.SCHEMA_ONLY,
        ):
            schema.validate(_dask_da())

    def test_nullable_skipped_schema_only(self):
        """nullable=False is DATA-scoped; skipped for Dask default."""
        da = xr.DataArray(
            dda.from_array(np.array([1.0, np.nan, 3.0]), chunks=2),
            dims=("x",),
            name="temp",
        )
        schema = DataArraySchema(dims=("x",), name="temp", nullable=False)
        schema.validate(da)


# ==================================================================
# 5. DatasetSchema with Dask-backed variables
# ==================================================================


class TestDaskDataset:
    """DatasetSchema validation for Datasets with Dask variables."""

    @staticmethod
    def _dask_ds():
        return xr.Dataset(
            {
                "a": (
                    "x",
                    dda.from_array(np.array([1.0, 2.0, 3.0]), chunks=2),
                ),
                "b": (
                    "x",
                    dda.from_array(np.array([4.0, 5.0, 6.0]), chunks=2),
                ),
            }
        )

    @staticmethod
    def _mixed_ds():
        return xr.Dataset(
            {
                "eager_var": ("x", np.array([1.0, 2.0, 3.0])),
                "dask_var": (
                    "x",
                    dda.from_array(np.array([4.0, 5.0, 6.0]), chunks=2),
                ),
            }
        )

    def test_dataset_validates_dask_data_vars(self):
        schema = DatasetSchema(
            data_vars={
                "a": DataVar(dtype=np.float64, dims=("x",)),
                "b": DataVar(dtype=np.float64, dims=("x",)),
            },
            dims=("x",),
        )
        schema.validate(self._dask_ds())

    def test_per_var_chunked_true(self):
        schema = DatasetSchema(
            data_vars={
                "a": DataVar(
                    dtype=np.float64,
                    dims=("x",),
                    chunked=True,
                ),
                "b": DataVar(
                    dtype=np.float64,
                    dims=("x",),
                    chunked=True,
                ),
            },
        )
        schema.validate(self._dask_ds())

    def test_per_var_chunked_true_fails_eager(self):
        ds = xr.Dataset(
            {
                "a": ("x", np.array([1.0, 2.0, 3.0])),
            }
        )
        schema = DatasetSchema(
            data_vars={
                "a": DataVar(dims=("x",), chunked=True),
            },
        )
        with pytest.raises(_SchemaErrors):
            schema.validate(ds)

    def test_mixed_eager_dask(self):
        schema = DatasetSchema(
            data_vars={
                "eager_var": DataVar(
                    dtype=np.float64,
                    dims=("x",),
                    chunked=False,
                ),
                "dask_var": DataVar(
                    dtype=np.float64,
                    dims=("x",),
                    chunked=True,
                ),
            },
        )
        schema.validate(self._mixed_ds())

    def test_structural_checks_no_compute(self):
        """Dims, sizes, dtype checked without computing."""
        schema = DatasetSchema(
            data_vars={
                "a": DataVar(
                    dtype=np.float64,
                    dims=("x",),
                ),
                "b": DataVar(
                    dtype=np.float64,
                    dims=("x",),
                ),
            },
            dims=("x",),
            sizes={"x": 3},
        )
        schema.validate(self._dask_ds())

    def test_structural_wrong_dims_fails(self):
        schema = DatasetSchema(
            data_vars={
                "a": DataVar(dims=("x",)),
                "b": DataVar(dims=("x",)),
            },
            dims=("y",),
        )
        with pytest.raises(_SchemaErrors):
            schema.validate(self._dask_ds())


# ==================================================================
# 6. Lazy validation (error collection) with Dask
# ==================================================================


class TestLazyValidationDask:
    """``lazy=True`` error collection for Dask-backed objects."""

    def test_lazy_collects_schema_errors(self):
        da = _dask_da()
        schema = DataArraySchema(dims=("y",), name="wrong", dtype=np.int32)
        with config_context(
            validation_depth=ValidationDepth.SCHEMA_ONLY,
        ):
            with pytest.raises(
                pandera.errors.SchemaErrors,
            ) as exc_info:
                schema.validate(da, lazy=True)
            assert len(exc_info.value.schema_errors) >= 2

    def test_lazy_collects_schema_and_data_errors(self):
        da = xr.DataArray(
            dda.from_array(np.array([1.0, np.nan, 3.0]), chunks=2),
            dims=("x",),
            name="temp",
        )
        schema = DataArraySchema(
            dims=("y",),
            name="temp",
            nullable=False,
        )
        with config_context(
            validation_depth=ValidationDepth.SCHEMA_AND_DATA,
        ):
            with pytest.raises(
                pandera.errors.SchemaErrors,
            ) as exc_info:
                schema.validate(da, lazy=True)
            reasons = {e.reason_code for e in exc_info.value.schema_errors}
            assert pandera.errors.SchemaErrorReason.MISMATCH_INDEX in reasons
            assert (
                pandera.errors.SchemaErrorReason.SERIES_CONTAINS_NULLS
                in reasons
            )
