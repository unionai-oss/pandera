"""Polars validation engine utilities."""

import polars as pl

from pandera.api.polars.types import PolarsCheckObjects
from pandera.config import (
    get_config_context,
    get_config_global,
    ValidationDepth,
)


def get_validation_depth(check_obj: PolarsCheckObjects) -> ValidationDepth:
    """Get validation depth for a given polars check object."""
    is_dataframe = isinstance(check_obj, pl.DataFrame)

    config_global = get_config_global()
    config_ctx = get_config_context()

    if (
        isinstance(check_obj, pl.LazyFrame)
        and config_global.validation_depth is None
    ):
        # if global validation depth is not set, use schema only validation
        # when validating LazyFrames
        validation_depth = ValidationDepth.SCHEMA_ONLY
    elif is_dataframe and config_ctx.validation_depth is None:
        # use schema and data validation when validating pl.DataFrame
        validation_depth = ValidationDepth.SCHEMA_AND_DATA
    elif is_dataframe or config_ctx.validation_depth is not None:
        # when validating pl.DataFrame, or when context-level config is set,
        # propagate the context-level configuration
        validation_depth = config_ctx.validation_depth  # type: ignore[assignment]
    else:
        validation_depth = ValidationDepth.SCHEMA_ONLY

    return validation_depth
