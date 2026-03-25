"""Narwhals API types."""
from typing import NamedTuple

import narwhals.stable.v1 as nw


class NarwhalsData(NamedTuple):
    """Data container for narwhals-backed validation.

    Note: field is named ``frame`` (not ``lazyframe``) to distinguish from
    the Polars ``PolarsData`` naming convention.
    """

    frame: nw.LazyFrame
    key: str = "*"


class NarwhalsCheckResult(NamedTuple):
    """Check result for user-defined checks on narwhals frames."""

    check_output: nw.LazyFrame
    check_passed: nw.LazyFrame
    checked_object: nw.LazyFrame
    failure_cases: nw.LazyFrame
