"""Xarray schema helpers (validation depth, etc.)."""

from __future__ import annotations

from typing import Any

from pandera.config import (
    ValidationDepth,
    get_config_context,
    get_config_global,
)


def _is_chunked_xarray(check_obj: Any) -> bool:
    try:
        import xarray as xr
    except ImportError:
        return False
    if isinstance(check_obj, xr.DataArray):
        return getattr(check_obj, "chunks", None) is not None
    if isinstance(check_obj, xr.Dataset):
        return any(
            getattr(v, "chunks", None) is not None
            for v in check_obj.data_vars.values()
        )
    return False


def get_validation_depth(check_obj: Any) -> ValidationDepth:
    """Resolve :class:`ValidationDepth` for an xarray object (Polars-style).

    Precedence matches :func:`pandera.api.polars.utils.get_validation_depth`:
    context ``validation_depth``, then global config, then defaults — chunked
    (Dask-backed) objects default to :attr:`ValidationDepth.SCHEMA_ONLY` so
    data-level checks do not compute lazy storage unless the user sets
    ``PANDERA_VALIDATION_DEPTH`` or uses :func:`~pandera.config.config_context`.
    Eager objects default to :attr:`ValidationDepth.SCHEMA_AND_DATA`.
    """
    config_global = get_config_global()
    config_ctx = get_config_context(validation_depth_default=None)

    if config_ctx.validation_depth is not None:
        return config_ctx.validation_depth

    if config_global.validation_depth is not None:
        return config_global.validation_depth

    if _is_chunked_xarray(check_obj):
        return ValidationDepth.SCHEMA_ONLY

    return ValidationDepth.SCHEMA_AND_DATA
