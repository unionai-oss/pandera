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
    if polars_version().release >= (1, 42, 1):
        return pl.concat(items, how="horizontal_extend")  # type: ignore[arg-type]
    return pl.concat(items, how="horizontal")
