"""Utilities for the DataFusion schema API."""

from __future__ import annotations

import datafusion

from pandera.api.pyarrow.utils import resolve_dtype
from pandera.config import (
    ValidationDepth,
    get_config_context,
    get_config_global,
)

__all__ = ["get_validation_depth", "resolve_dtype"]


def get_validation_depth(check_obj: datafusion.DataFrame) -> ValidationDepth:
    """Get the validation depth for a DataFusion DataFrame.

    A ``datafusion.DataFrame`` is a lazy query plan, like ``pl.LazyFrame`` or
    ``ibis.Table``: running data-level checks executes it. Unless the context
    or global configuration says otherwise, only schema-level checks run by
    default.
    """
    config_ctx = get_config_context(validation_depth_default=None)
    if config_ctx.validation_depth is not None:
        return config_ctx.validation_depth

    config_global = get_config_global()
    if config_global.validation_depth is not None:
        return config_global.validation_depth

    return ValidationDepth.SCHEMA_ONLY
