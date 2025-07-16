# pylint: disable=cyclic-import
"""Narwhals validation engine utilities."""

from typing import Dict, List, Any

import narwhals as nw

from pandera.api.narwhals.types import NarwhalsCheckObjects
from pandera.config import (
    ValidationDepth,
    get_config_context,
    get_config_global,
)


def get_dataframe_schema(df: nw.DataFrame[Any]) -> Dict[str, nw.Dtype]:
    """Get a dict of column names and dtypes from a narwhals DataFrame."""
    # Placeholder implementation - would need proper narwhals schema access
    return {col: df.select(col).dtype for col in df.columns}


def get_dataframe_column_dtypes(df: nw.DataFrame[Any]) -> List[nw.Dtype]:
    """Get a list of column dtypes from a narwhals DataFrame."""
    # Placeholder implementation - would need proper narwhals dtype access
    return [df.select(col).dtype for col in df.columns]


def get_dataframe_column_names(df: nw.DataFrame[Any]) -> List[str]:
    """Get a list of column names from a narwhals DataFrame."""
    return df.columns


def get_validation_depth(check_obj: NarwhalsCheckObjects) -> ValidationDepth:
    """Get validation depth for a given narwhals check object."""
    is_dataframe = isinstance(check_obj, nw.DataFrame)

    config_global = get_config_global()
    config_ctx = get_config_context(validation_depth_default=None)

    if config_ctx.validation_depth is not None:
        # use context configuration if specified
        return config_ctx.validation_depth

    if config_global.validation_depth is not None:
        # use global configuration if specified
        return config_global.validation_depth

    if (
        isinstance(check_obj, nw.LazyFrame)
        and config_global.validation_depth is None
    ):
        # if global validation depth is not set, use schema only validation
        # when validating LazyFrames
        validation_depth = ValidationDepth.SCHEMA_ONLY
    elif is_dataframe and (
        config_ctx.validation_depth is None
        or config_ctx.validation_depth is None
    ):
        # if context validation depth is not set, use schema and data validation
        # when validating DataFrames
        validation_depth = ValidationDepth.SCHEMA_AND_DATA
    else:
        validation_depth = ValidationDepth.SCHEMA_ONLY

    return validation_depth
