"""Polars validation engine utilities."""

import polars as pl

from pandera.api.polars.types import PolarsCheckObjects
from pandera.config import (
    get_config_context,
    get_config_global,
    ValidationDepth,
)


def get_validation_depth(check_obj: PolarsCheckObjects) -> ValidationDepth:
    is_dataframe = isinstance(check_obj, pl.DataFrame)

    config = get_config_global()
    config_from_ctx = get_config_context()

    if isinstance(check_obj, pl.LazyFrame) and config.validation_depth is None:
        # if global validation depth is not set, use schema only validation
        validation_depth = ValidationDepth.SCHEMA_ONLY
    elif is_dataframe and config_from_ctx.validation_depth is None:
        validation_depth = ValidationDepth.SCHEMA_AND_DATA
    elif is_dataframe or config_from_ctx.validation_depth is not None:
        # when validating pl.DataFrame, or when context-level config is set,
        # propagate the context-level configuration
        validation_depth = config_from_ctx.validation_depth  # type: ignore[assignment]
    else:
        validation_depth = ValidationDepth.SCHEMA_ONLY

    return validation_depth
