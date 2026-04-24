"""Polars validation engine utilities."""

from typing import Any

import polars as pl

from pandera.config import (
    ValidationDepth,
    get_config_context,
    get_config_global,
)
from pandera.engines.polars_engine import polars_version


def get_lazyframe_schema(lf: pl.LazyFrame) -> dict[str, pl.DataType]:
    """Get a dict of column names and  dtypes from a polars LazyFrame."""
    if polars_version().release >= (1, 0, 0):
        return lf.collect_schema()
    return lf.schema


def get_lazyframe_column_dtypes(lf: pl.LazyFrame) -> list[pl.DataType]:
    """Get a list of column dtypes from a polars LazyFrame."""
    if polars_version().release >= (1, 0, 0):
        return lf.collect_schema().dtypes()
    return [*lf.schema.values()]


def get_lazyframe_column_names(lf: pl.LazyFrame) -> list[str]:
    """Get a list of column names from a polars LazyFrame."""
    if polars_version().release >= (1, 0, 0):
        return lf.collect_schema().names()
    return lf.columns


def get_validation_depth(check_obj: Any) -> ValidationDepth:
    """Get validation depth for a given check object.

    The narwhals backend can validate any frame supported by narwhals
    (``pl.DataFrame``, ``pl.LazyFrame``, ``pd.DataFrame``,
    ``pyarrow.Table``, ``ibis.Table``, etc.). LazyFrame-like inputs default
    to ``SCHEMA_ONLY`` so we don't force a full collect; eager inputs
    default to ``SCHEMA_AND_DATA``.
    """
    config_global = get_config_global()
    config_ctx = get_config_context(validation_depth_default=None)

    if config_ctx.validation_depth is not None:
        return config_ctx.validation_depth

    if config_global.validation_depth is not None:
        return config_global.validation_depth

    # Treat polars LazyFrame and SQL-lazy backends (ibis.Table) as lazy.
    # ``ibis.Table`` exposes ``.execute`` rather than ``.collect``; both should
    # default to SCHEMA_ONLY to avoid materializing the full frame.
    is_lazy = isinstance(check_obj, pl.LazyFrame) or _is_sql_lazy(check_obj)
    if is_lazy:
        return ValidationDepth.SCHEMA_ONLY

    return ValidationDepth.SCHEMA_AND_DATA


def _is_sql_lazy(check_obj: Any) -> bool:
    """Detect SQL-lazy frames (e.g. ``ibis.Table``) without importing ibis."""
    try:
        import ibis

        return isinstance(check_obj, ibis.Table)
    except ImportError:
        return False
