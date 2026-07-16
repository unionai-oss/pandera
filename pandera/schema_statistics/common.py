"""Shared, dataframe-library-agnostic statistics helpers.

This module must stay free of pandas (and other dataframe-library) imports so
that pandas-free entrypoints like ``pandera.polars`` can use it.
"""

from __future__ import annotations

from typing import Any


def string_length_check_statistics(
    min_len: int, max_len: int
) -> dict[str, Any]:
    """Build ``parse_check_statistics``-compatible stats for
    :meth:`~pandera.api.checks.Check.str_length`.
    """
    return {
        "str_length": {
            "min_value": min_len,
            "max_value": max_len,
        },
    }
