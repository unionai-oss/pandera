"""Tests for xarray :class:`~pandera.config.ValidationDepth` resolution.

Also covers end-to-end validation behaviour under each depth setting
for DataArray and Dataset schemas.
"""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

import pandera.errors
from pandera.api.xarray.utils import get_validation_depth
from pandera.config import (
    CONFIG,
    ValidationDepth,
    config_context,
    reset_config_context,
)
from pandera.xarray import (
    Check,
    Coordinate,
    DataArraySchema,
    DatasetSchema,
    DataVar,
)


@pytest.fixture
def validation_depth_none():
    """Global ``validation_depth`` unset (like no ``PANDERA_VALIDATION_DEPTH``)."""
    previous = CONFIG.validation_depth
    CONFIG.validation_depth = None
    reset_config_context()
    try:
        yield
    finally:
        CONFIG.validation_depth = previous
        reset_config_context()


def test_get_depth_eager_defaults_schema_and_data(validation_depth_none):
    da = xr.DataArray(np.ones(2), dims="x")
    assert get_validation_depth(da) == ValidationDepth.SCHEMA_AND_DATA


def test_get_depth_chunked_defaults_schema_only(validation_depth_none):
    pytest.importorskip("dask.array")
    import dask.array as dda

    da = xr.DataArray(dda.ones(2, chunks=1), dims="x")
    assert get_validation_depth(da) == ValidationDepth.SCHEMA_ONLY


def test_get_depth_dataset_any_chunked_defaults_schema_only(
    validation_depth_none,
):
    pytest.importorskip("dask.array")
    import dask.array as dda

    ds = xr.Dataset(
        {"a": ("x", np.ones(2)), "b": ("x", dda.ones(2, chunks=1))}
    )
    assert get_validation_depth(ds) == ValidationDepth.SCHEMA_ONLY


def test_get_depth_context_overrides_chunked_default(validation_depth_none):
    pytest.importorskip("dask.array")
    import dask.array as dda

    da = xr.DataArray(dda.ones(2, chunks=1), dims="x")
    with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
        assert get_validation_depth(da) == ValidationDepth.SCHEMA_AND_DATA


def test_get_depth_global_overrides_chunked_default(validation_depth_none):
    pytest.importorskip("dask.array")
    import dask.array as dda

    previous = CONFIG.validation_depth
    CONFIG.validation_depth = ValidationDepth.DATA_ONLY
    reset_config_context()
    try:
        da = xr.DataArray(dda.ones(2, chunks=1), dims="x")
        assert get_validation_depth(da) == ValidationDepth.DATA_ONLY
    finally:
        CONFIG.validation_depth = previous
        reset_config_context()


# ================================================================
# DataArraySchema – validation depth end-to-end
# ================================================================


class TestDataArrayValidationDepth:
    """Verify SCHEMA / DATA check gating for DataArraySchema."""

    @staticmethod
    def _da_with_nulls():
        """DataArray that has NaN values (for nullable tests)."""
        return xr.DataArray([1.0, np.nan, 3.0], dims="x", name="arr")

    @staticmethod
    def _da_clean():
        """DataArray with no NaN values."""
        return xr.DataArray([1.0, 2.0, 3.0], dims="x", name="arr")

    # --- SCHEMA_ONLY -------------------------------------------------

    def test_schema_only_skips_nullable(self):
        """nullable=False is DATA-scoped; it should be skipped."""
        da = self._da_with_nulls()
        schema = DataArraySchema(
            dtype=np.float64,
            dims=("x",),
            name="arr",
            nullable=False,
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_ONLY):
            schema.validate(da)

    def test_schema_only_skips_user_checks(self):
        """User-defined Checks are DATA-scoped; they should be skipped."""
        da = self._da_clean()
        schema = DataArraySchema(
            dims=("x",),
            name="arr",
            checks=Check(lambda x: False, error="always fails"),
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_ONLY):
            schema.validate(da)

    def test_schema_only_catches_wrong_dims(self):
        """SCHEMA check_dims should still fire under SCHEMA_ONLY."""
        da = self._da_clean()
        schema = DataArraySchema(dims=("y",), name="arr")
        with config_context(validation_depth=ValidationDepth.SCHEMA_ONLY):
            with pytest.raises(
                (pandera.errors.SchemaError, pandera.errors.SchemaErrors)
            ):
                schema.validate(da)

    def test_schema_only_catches_wrong_name(self):
        """SCHEMA check_name should still fire under SCHEMA_ONLY."""
        da = self._da_clean()
        schema = DataArraySchema(dims=("x",), name="wrong")
        with config_context(validation_depth=ValidationDepth.SCHEMA_ONLY):
            with pytest.raises(
                (pandera.errors.SchemaError, pandera.errors.SchemaErrors)
            ):
                schema.validate(da)

    def test_schema_only_catches_wrong_dtype(self):
        """SCHEMA check_dtype should still fire under SCHEMA_ONLY."""
        da = self._da_clean()
        schema = DataArraySchema(
            dtype=np.int32,
            dims=("x",),
            name="arr",
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_ONLY):
            with pytest.raises(
                (pandera.errors.SchemaError, pandera.errors.SchemaErrors)
            ):
                schema.validate(da)

    def test_schema_only_catches_wrong_sizes(self):
        """SCHEMA check_sizes should still fire under SCHEMA_ONLY."""
        da = self._da_clean()
        schema = DataArraySchema(
            dims=("x",),
            name="arr",
            sizes={"x": 99},
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_ONLY):
            with pytest.raises(
                (pandera.errors.SchemaError, pandera.errors.SchemaErrors)
            ):
                schema.validate(da)

    # --- DATA_ONLY ---------------------------------------------------

    def test_data_only_skips_wrong_name(self):
        """SCHEMA check_name should be skipped under DATA_ONLY."""
        da = self._da_clean()
        schema = DataArraySchema(name="wrong")
        with config_context(validation_depth=ValidationDepth.DATA_ONLY):
            schema.validate(da)

    def test_data_only_skips_wrong_dims(self):
        """SCHEMA check_dims should be skipped under DATA_ONLY."""
        da = self._da_clean()
        schema = DataArraySchema(dims=("a", "b"))
        with config_context(validation_depth=ValidationDepth.DATA_ONLY):
            schema.validate(da)

    def test_data_only_skips_wrong_dtype(self):
        """SCHEMA check_dtype should be skipped under DATA_ONLY."""
        da = self._da_clean()
        schema = DataArraySchema(dtype=np.int32)
        with config_context(validation_depth=ValidationDepth.DATA_ONLY):
            schema.validate(da)

    def test_data_only_skips_wrong_sizes(self):
        """SCHEMA check_sizes should be skipped under DATA_ONLY."""
        da = self._da_clean()
        schema = DataArraySchema(
            dims=("x",),
            sizes={"x": 99},
        )
        with config_context(validation_depth=ValidationDepth.DATA_ONLY):
            schema.validate(da)

    def test_data_only_catches_nullable(self):
        """DATA check_nullable should still fire under DATA_ONLY."""
        da = self._da_with_nulls()
        schema = DataArraySchema(nullable=False)
        with config_context(validation_depth=ValidationDepth.DATA_ONLY):
            with pytest.raises(
                (pandera.errors.SchemaError, pandera.errors.SchemaErrors)
            ):
                schema.validate(da)

    def test_data_only_catches_user_check(self):
        """DATA run_checks should still fire under DATA_ONLY."""
        da = self._da_clean()
        schema = DataArraySchema(
            checks=Check(lambda x: False, error="always fails"),
        )
        with config_context(validation_depth=ValidationDepth.DATA_ONLY):
            with pytest.raises(
                (pandera.errors.SchemaError, pandera.errors.SchemaErrors)
            ):
                schema.validate(da)

    # --- SCHEMA_AND_DATA (both fire) ---------------------------------

    def test_schema_and_data_catches_schema_error(self):
        da = self._da_clean()
        schema = DataArraySchema(dims=("y",), name="arr")
        with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
            with pytest.raises(
                (pandera.errors.SchemaError, pandera.errors.SchemaErrors)
            ):
                schema.validate(da)

    def test_schema_and_data_catches_data_error(self):
        da = self._da_with_nulls()
        schema = DataArraySchema(
            dims=("x",),
            name="arr",
            nullable=False,
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
            with pytest.raises(
                (pandera.errors.SchemaError, pandera.errors.SchemaErrors)
            ):
                schema.validate(da)

    def test_schema_and_data_catches_both(self):
        """Both SCHEMA and DATA errors collected in lazy mode."""
        da = self._da_with_nulls()
        schema = DataArraySchema(
            dims=("y",),
            name="arr",
            nullable=False,
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
            with pytest.raises(pandera.errors.SchemaErrors) as exc_info:
                schema.validate(da, lazy=True)
            reasons = {e.reason_code for e in exc_info.value.schema_errors}
            assert pandera.errors.SchemaErrorReason.MISMATCH_INDEX in reasons
            assert (
                pandera.errors.SchemaErrorReason.SERIES_CONTAINS_NULLS
                in reasons
            )

    # --- Default (None) behaves like SCHEMA_AND_DATA for eager -------

    def test_default_depth_catches_both_for_eager(self):
        """No explicit depth → eager data uses SCHEMA_AND_DATA."""
        da = self._da_with_nulls()
        schema = DataArraySchema(
            dims=("y",),
            name="arr",
            nullable=False,
        )
        with pytest.raises(pandera.errors.SchemaErrors) as exc_info:
            schema.validate(da, lazy=True)
        reasons = {e.reason_code for e in exc_info.value.schema_errors}
        assert pandera.errors.SchemaErrorReason.MISMATCH_INDEX in reasons
        assert (
            pandera.errors.SchemaErrorReason.SERIES_CONTAINS_NULLS in reasons
        )


# ================================================================
# DatasetSchema – validation depth end-to-end
# ================================================================


class TestDatasetValidationDepth:
    """Verify SCHEMA / DATA check gating for DatasetSchema."""

    @staticmethod
    def _ds_with_nulls():
        """Dataset containing a variable with NaN."""
        return xr.Dataset(
            {"a": ("x", [1.0, np.nan, 3.0])},
            coords={"x": np.arange(3)},
        )

    @staticmethod
    def _ds_clean():
        """Dataset with no NaN values."""
        return xr.Dataset(
            {"a": ("x", [1.0, 2.0, 3.0])},
            coords={"x": np.arange(3)},
        )

    # --- SCHEMA_ONLY -------------------------------------------------

    def test_schema_only_skips_per_var_nullable(self):
        """Per-variable nullable=False (DATA) should be skipped."""
        ds = self._ds_with_nulls()
        schema = DatasetSchema(
            data_vars={
                "a": DataVar(
                    dtype=np.float64,
                    dims=("x",),
                    nullable=False,
                ),
            },
            dims=("x",),
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_ONLY):
            schema.validate(ds)

    def test_schema_only_skips_per_var_user_check(self):
        """Per-variable user Check (DATA) should be skipped."""
        ds = self._ds_clean()
        schema = DatasetSchema(
            data_vars={
                "a": DataVar(
                    dtype=np.float64,
                    dims=("x",),
                    checks=Check(lambda x: False, error="always fails"),
                ),
            },
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_ONLY):
            schema.validate(ds)

    def test_schema_only_skips_dataset_user_check(self):
        """Dataset-level user Check (DATA) should be skipped."""
        ds = self._ds_clean()
        schema = DatasetSchema(
            data_vars={"a": DataVar(dtype=np.float64, dims=("x",))},
            checks=Check(lambda x: False, error="always fails"),
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_ONLY):
            schema.validate(ds)

    def test_schema_only_catches_wrong_dims(self):
        """SCHEMA check_dims should still fire."""
        ds = self._ds_clean()
        schema = DatasetSchema(
            data_vars={"a": DataVar(dims=("x",))},
            dims=("y",),
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_ONLY):
            with pytest.raises(
                (pandera.errors.SchemaError, pandera.errors.SchemaErrors)
            ):
                schema.validate(ds)

    def test_schema_only_catches_wrong_sizes(self):
        """SCHEMA check_sizes should still fire."""
        ds = self._ds_clean()
        schema = DatasetSchema(
            data_vars={"a": DataVar(dims=("x",))},
            sizes={"x": 99},
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_ONLY):
            with pytest.raises(
                (pandera.errors.SchemaError, pandera.errors.SchemaErrors)
            ):
                schema.validate(ds)

    def test_schema_only_catches_missing_data_var(self):
        """SCHEMA check_data_var_presence should still fire."""
        ds = self._ds_clean()
        schema = DatasetSchema(
            data_vars={
                "a": DataVar(dims=("x",)),
                "missing": DataVar(dims=("x",), required=True),
            },
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_ONLY):
            with pytest.raises(
                (pandera.errors.SchemaError, pandera.errors.SchemaErrors)
            ):
                schema.validate(ds)

    def test_schema_only_catches_strict_extra_var(self):
        """SCHEMA check_strict_data_vars should still fire."""
        ds = xr.Dataset(
            {
                "a": ("x", [1.0, 2.0]),
                "extra": ("x", [3.0, 4.0]),
            }
        )
        schema = DatasetSchema(
            data_vars={"a": DataVar(dims=("x",))},
            strict=True,
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_ONLY):
            with pytest.raises(
                (pandera.errors.SchemaError, pandera.errors.SchemaErrors)
            ):
                schema.validate(ds)

    # --- DATA_ONLY ---------------------------------------------------

    def test_data_only_skips_wrong_dims(self):
        """SCHEMA check_dims should be skipped under DATA_ONLY."""
        ds = self._ds_clean()
        schema = DatasetSchema(
            data_vars={"a": DataVar(dims=("x",))},
            dims=("y",),
        )
        with config_context(validation_depth=ValidationDepth.DATA_ONLY):
            schema.validate(ds)

    def test_data_only_skips_wrong_sizes(self):
        """SCHEMA check_sizes should be skipped under DATA_ONLY."""
        ds = self._ds_clean()
        schema = DatasetSchema(
            data_vars={"a": DataVar(dims=("x",))},
            sizes={"x": 99},
        )
        with config_context(validation_depth=ValidationDepth.DATA_ONLY):
            schema.validate(ds)

    def test_data_only_skips_missing_data_var(self):
        """SCHEMA check_data_var_presence should be skipped."""
        ds = self._ds_clean()
        schema = DatasetSchema(
            data_vars={
                "a": DataVar(dims=("x",)),
                "missing": DataVar(dims=("x",), required=True),
            },
        )
        with config_context(validation_depth=ValidationDepth.DATA_ONLY):
            schema.validate(ds)

    def test_data_only_skips_strict_extra_var(self):
        """SCHEMA check_strict_data_vars should be skipped."""
        ds = xr.Dataset(
            {
                "a": ("x", [1.0, 2.0]),
                "extra": ("x", [3.0, 4.0]),
            }
        )
        schema = DatasetSchema(
            data_vars={"a": DataVar(dims=("x",))},
            strict=True,
        )
        with config_context(validation_depth=ValidationDepth.DATA_ONLY):
            schema.validate(ds)

    def test_data_only_skips_per_var_schema_checks(self):
        """Per-variable SCHEMA checks (wrong dims) should be skipped."""
        ds = self._ds_clean()
        schema = DatasetSchema(
            data_vars={
                "a": DataVar(dtype=np.float64, dims=("y",)),
            },
        )
        with config_context(validation_depth=ValidationDepth.DATA_ONLY):
            schema.validate(ds)

    def test_data_only_catches_per_var_nullable(self):
        """Per-variable nullable (DATA) should still fire."""
        ds = self._ds_with_nulls()
        schema = DatasetSchema(
            data_vars={
                "a": DataVar(
                    dtype=np.float64,
                    dims=("x",),
                    nullable=False,
                ),
            },
        )
        with config_context(validation_depth=ValidationDepth.DATA_ONLY):
            with pytest.raises(
                (pandera.errors.SchemaError, pandera.errors.SchemaErrors)
            ):
                schema.validate(ds)

    def test_data_only_catches_per_var_user_check(self):
        """Per-variable user Check (DATA) should still fire."""
        ds = self._ds_clean()
        schema = DatasetSchema(
            data_vars={
                "a": DataVar(
                    dims=("x",),
                    checks=Check(lambda x: False, error="always fails"),
                ),
            },
        )
        with config_context(validation_depth=ValidationDepth.DATA_ONLY):
            with pytest.raises(
                (pandera.errors.SchemaError, pandera.errors.SchemaErrors)
            ):
                schema.validate(ds)

    def test_data_only_catches_dataset_user_check(self):
        """Dataset-level user Check (DATA) should still fire."""
        ds = self._ds_clean()
        schema = DatasetSchema(
            data_vars={"a": DataVar(dims=("x",))},
            checks=Check(lambda x: False, error="always fails"),
        )
        with config_context(validation_depth=ValidationDepth.DATA_ONLY):
            with pytest.raises(
                (pandera.errors.SchemaError, pandera.errors.SchemaErrors)
            ):
                schema.validate(ds)

    # --- SCHEMA_AND_DATA (both fire) ---------------------------------

    def test_schema_and_data_catches_schema_error(self):
        ds = self._ds_clean()
        schema = DatasetSchema(
            data_vars={"a": DataVar(dims=("x",))},
            dims=("y",),
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
            with pytest.raises(
                (pandera.errors.SchemaError, pandera.errors.SchemaErrors)
            ):
                schema.validate(ds)

    def test_schema_and_data_catches_data_error(self):
        ds = self._ds_with_nulls()
        schema = DatasetSchema(
            data_vars={
                "a": DataVar(
                    dtype=np.float64,
                    dims=("x",),
                    nullable=False,
                ),
            },
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
            with pytest.raises(
                (pandera.errors.SchemaError, pandera.errors.SchemaErrors)
            ):
                schema.validate(ds)

    def test_schema_and_data_catches_both(self):
        """Both SCHEMA and DATA errors collected in lazy mode."""
        ds = self._ds_with_nulls()
        schema = DatasetSchema(
            data_vars={
                "a": DataVar(
                    dtype=np.float64,
                    dims=("x",),
                    nullable=False,
                ),
            },
            dims=("y",),
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
            with pytest.raises(pandera.errors.SchemaErrors) as exc_info:
                schema.validate(ds, lazy=True)
            reasons = {e.reason_code for e in exc_info.value.schema_errors}
            assert pandera.errors.SchemaErrorReason.MISMATCH_INDEX in reasons
            assert (
                pandera.errors.SchemaErrorReason.SERIES_CONTAINS_NULLS
                in reasons
            )

    # --- Default (None) behaves like SCHEMA_AND_DATA for eager -------

    def test_default_depth_catches_both_for_eager(self):
        ds = self._ds_with_nulls()
        schema = DatasetSchema(
            data_vars={
                "a": DataVar(
                    dtype=np.float64,
                    dims=("x",),
                    nullable=False,
                ),
            },
            dims=("y",),
        )
        with pytest.raises(pandera.errors.SchemaErrors) as exc_info:
            schema.validate(ds, lazy=True)
        reasons = {e.reason_code for e in exc_info.value.schema_errors}
        assert pandera.errors.SchemaErrorReason.MISMATCH_INDEX in reasons
        assert (
            pandera.errors.SchemaErrorReason.SERIES_CONTAINS_NULLS in reasons
        )
