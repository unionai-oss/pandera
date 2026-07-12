"""Utilities for the polars backend."""

from collections.abc import Sequence

import polars as pl
from packaging import version


def polars_version() -> version.Version:
    """Return the polars version."""

    return version.parse(pl.__version__)


def horizontal_concat(
    items: Sequence[pl.LazyFrame],
) -> pl.LazyFrame:
    """Concat LazyFrames horizontally across supported polars versions."""
    try:
        return pl.concat(items, how="horizontal_extend")  # type: ignore[arg-type]
    except ValueError as exc:
        if "got 'horizontal_extend'" not in str(exc):
            raise
        return pl.concat(items, how="horizontal")
