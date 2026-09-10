"""Regression tests for _concat_failure_cases polars branch — TQ-02.

Verifies that the polars branch of _concat_failure_cases merges pl_items
(native pl.DataFrame from _build_eager_failure_case / _build_scalar_failure_case)
with nw_items (nw.LazyFrame from _build_lazy_failure_case) instead of silently
dropping pl_items.
"""

import warnings

import narwhals.stable.v1 as nw
import polars as pl
import pytest

from pandera.backends.narwhals.base import _concat_failure_cases
from pandera.errors import SchemaWarning


def test_concat_failure_cases_polars_merges_pl_items_and_nw_items():
    """Polars branch merges pl_items + nw_items into a single frame.

    A validation run with both schema-level failures (pl.DataFrame from
    _build_eager_failure_case / _build_scalar_failure_case) and data-check
    failures (nw.LazyFrame from _build_lazy_failure_case) must return a
    combined frame containing rows from both sources.

    Regression test for the bug where pl_items were silently dropped.
    """
    nw_item = nw.from_native(
        pl.LazyFrame({"col_x": [1, 2]}), eager_or_interchange_only=False
    )
    pl_item = pl.DataFrame({"col_x": [3]})

    result = _concat_failure_cases([nw_item, pl_item])

    assert result.height == 3
    assert set(result["col_x"].to_list()) == {1, 2, 3}


def test_concat_failure_cases_polars_only_nw_items_returns_lazy_native():
    """When only nw_items are present (no pl_items), the polars branch stays lazy.

    The all-lazy path must return a native pl.LazyFrame to preserve laziness.
    This test must pass BEFORE and AFTER the TQ-02 fix.
    """
    nw_item_a = nw.from_native(
        pl.LazyFrame({"col_x": [1, 2]}), eager_or_interchange_only=False
    )
    nw_item_b = nw.from_native(
        pl.LazyFrame({"col_x": [3, 4]}), eager_or_interchange_only=False
    )

    result = _concat_failure_cases([nw_item_a, nw_item_b])

    assert isinstance(result, pl.LazyFrame)
    assert set(result.collect()["col_x"].to_list()) == {1, 2, 3, 4}


def test_concat_failure_cases_polars_emits_no_warning():
    """Polars branch must NOT emit a SchemaWarning when merging pl_items + nw_items.

    The PySpark branch emits a SchemaWarning because it cannot convert
    pl.DataFrame to PySpark without a SparkSession. Polars has no such barrier
    and can merge both sources cleanly — no warning should be emitted.
    """
    nw_item = nw.from_native(
        pl.LazyFrame({"col_x": [1, 2]}), eager_or_interchange_only=False
    )
    pl_item = pl.DataFrame({"col_x": [3]})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _concat_failure_cases([nw_item, pl_item])

    assert not any(issubclass(w.category, SchemaWarning) for w in caught)
