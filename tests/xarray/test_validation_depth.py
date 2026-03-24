"""Tests for xarray :class:`~pandera.config.ValidationDepth` resolution."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from pandera.api.xarray.utils import get_validation_depth
from pandera.config import (
    CONFIG,
    ValidationDepth,
    config_context,
    reset_config_context,
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

    ds = xr.Dataset({"a": ("x", np.ones(2)), "b": ("x", dda.ones(2, chunks=1))})
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
