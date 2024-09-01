# pylint: disable=cyclic-import
"""Polars validation engine utilities."""

from typing import Dict, List

import polars as pl

from pandera.api.polars.types import PolarsCheckObjects
from pandera.engines.polars_engine import polars_version
from pandera.config import (
    ValidationDepth,
    get_config_context,
    get_config_global,
)


def get_lazyframe_schema(lf: pl.LazyFrame) -> Dict[str, pl.DataType]:
    """Get a dict of column names and  dtypes from a polars LazyFrame."""
    if polars_version().release >= (1, 0, 0):
        return lf.collect_schema()
    return lf.schema


def get_lazyframe_column_dtypes(lf: pl.LazyFrame) -> List[pl.DataType]:
    """Get a list of column dtypes from a polars LazyFrame."""
    if polars_version().release >= (1, 0, 0):
        return lf.collect_schema().dtypes()
    return [*lf.schema.values()]


def get_lazyframe_column_names(lf: pl.LazyFrame) -> List[str]:
    """Get a list of column names from a polars LazyFrame."""
    if polars_version().release >= (1, 0, 0):
        return lf.collect_schema().names()
    return lf.columns


def get_validation_depth(check_obj: PolarsCheckObjects) -> ValidationDepth:
    """Get validation depth for a given polars check object."""
    is_dataframe = isinstance(check_obj, pl.DataFrame)

    config_global = get_config_global()
    config_ctx = get_config_context(validation_depth_default=None)

    if config_ctx.validation_depth is not None:
        # use context configuration if specified
        return config_ctx.validation_depth

    if config_global.validation_depth is not None:
        # use global configuration if specified
        return config_global.validation_depth

    if (
        isinstance(check_obj, pl.LazyFrame)
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
