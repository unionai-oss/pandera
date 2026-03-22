"""Tests for ColumnBackend — COLUMN-01, COLUMN-02.

All tests are marked xfail(strict=False) until Plan 03-02 implements
ColumnBackend in pandera/backends/narwhals/components.py.
Once the implementation lands, all stubs flip to passing.
"""
import pytest
import polars as pl
import narwhals.stable.v1 as nw
from types import SimpleNamespace

from pandera.api.checks import Check
from pandera.backends.base import CoreCheckResult
from pandera.errors import SchemaErrorReason

# ---------------------------------------------------------------------------
# Guard: import ColumnBackend only if components.py exists (Plan 03-02)
# ---------------------------------------------------------------------------
try:
    from pandera.backends.narwhals.components import ColumnBackend
except ImportError:
    ColumnBackend = None


# ---------------------------------------------------------------------------
# Schema stub — mimics pandera Column schema with the fields ColumnBackend needs
# ---------------------------------------------------------------------------

def _make_schema(
    selector="col",
    nullable=True,
    unique=False,
    dtype=None,
    checks=None,
):
    """Return a minimal schema-like namespace for testing."""
    return SimpleNamespace(
        selector=selector,
        name=selector,
        nullable=nullable,
        unique=unique,
        dtype=dtype,
        checks=checks or [],
    )


# ---------------------------------------------------------------------------
# xfail marker — applied to every test below
# ---------------------------------------------------------------------------

_xfail = pytest.mark.xfail(
    ColumnBackend is None,
    reason="components.py not yet implemented (Plan 03-02)",
    strict=False,
)


# ---------------------------------------------------------------------------
# check_nullable tests
# ---------------------------------------------------------------------------

@_xfail
def test_check_nullable_passes(make_narwhals_frame):
    """COLUMN-01: nullable=True column with no nulls passes."""
    frame = make_narwhals_frame({"col": [1, 2, 3]})
    schema = _make_schema(selector="col", nullable=True)

    backend = ColumnBackend()
    results = backend.check_nullable(frame, schema)

    assert len(results) == 1
    assert results[0].passed is True


@_xfail
def test_check_nullable_fails_on_null(make_narwhals_frame):
    """COLUMN-01: nullable=False column with None values fails with SERIES_CONTAINS_NULLS."""
    frame = make_narwhals_frame({"col": [1, None, 3]})
    schema = _make_schema(selector="col", nullable=False)

    backend = ColumnBackend()
    results = backend.check_nullable(frame, schema)

    assert len(results) == 1
    assert results[0].passed is False
    assert results[0].reason_code == SchemaErrorReason.SERIES_CONTAINS_NULLS


@_xfail
def test_check_nullable_catches_nan(make_narwhals_frame):
    """COLUMN-01: nullable=False float column with NaN fails — NaN treated as null."""
    import math

    frame = make_narwhals_frame({"col": pl.Series([1.0, float("nan"), 3.0])})
    schema = _make_schema(selector="col", nullable=False)

    backend = ColumnBackend()
    results = backend.check_nullable(frame, schema)

    assert len(results) == 1
    assert results[0].passed is False
    assert results[0].reason_code == SchemaErrorReason.SERIES_CONTAINS_NULLS


# ---------------------------------------------------------------------------
# check_unique tests
# ---------------------------------------------------------------------------

@_xfail
def test_check_unique_passes(make_narwhals_frame):
    """COLUMN-02: unique=True column with all distinct values passes."""
    frame = make_narwhals_frame({"col": [1, 2, 3]})
    schema = _make_schema(selector="col", unique=True)

    backend = ColumnBackend()
    results = backend.check_unique(frame, schema)

    assert len(results) == 1
    assert results[0].passed is True


@_xfail
def test_check_unique_fails(make_narwhals_frame):
    """COLUMN-02: unique=True column with duplicate values fails with SERIES_CONTAINS_DUPLICATES."""
    frame = make_narwhals_frame({"col": [1, 2, 2]})
    schema = _make_schema(selector="col", unique=True)

    backend = ColumnBackend()
    results = backend.check_unique(frame, schema)

    assert len(results) == 1
    assert results[0].passed is False
    assert results[0].reason_code == SchemaErrorReason.SERIES_CONTAINS_DUPLICATES


# ---------------------------------------------------------------------------
# check_dtype tests
# ---------------------------------------------------------------------------

@_xfail
def test_check_dtype_correct(make_narwhals_frame):
    """COLUMN-02: column dtype matching schema.dtype passes."""
    from pandera.engines import narwhals_engine

    frame = make_narwhals_frame({"col": [1, 2, 3]})
    schema = _make_schema(selector="col", dtype=narwhals_engine.Int64())

    backend = ColumnBackend()
    results = backend.check_dtype(frame, schema)

    assert len(results) == 1
    assert results[0].passed is True


@_xfail
def test_check_dtype_wrong(make_narwhals_frame):
    """COLUMN-02: column dtype not matching schema.dtype fails with WRONG_DATATYPE."""
    from pandera.engines import narwhals_engine

    # Frame has Int64 but schema expects Float64
    frame = make_narwhals_frame({"col": [1, 2, 3]})
    schema = _make_schema(selector="col", dtype=narwhals_engine.Float64())

    backend = ColumnBackend()
    results = backend.check_dtype(frame, schema)

    assert len(results) == 1
    assert results[0].passed is False
    assert results[0].reason_code == SchemaErrorReason.WRONG_DATATYPE


@_xfail
def test_check_dtype_none(make_narwhals_frame):
    """COLUMN-02: schema.dtype=None short-circuits and returns passed=True."""
    frame = make_narwhals_frame({"col": [1, 2, 3]})
    schema = _make_schema(selector="col", dtype=None)

    backend = ColumnBackend()
    results = backend.check_dtype(frame, schema)

    assert len(results) == 1
    assert results[0].passed is True


# ---------------------------------------------------------------------------
# run_checks test
# ---------------------------------------------------------------------------

@_xfail
def test_run_checks(make_narwhals_frame):
    """COLUMN-02: run_checks executes Check objects and returns list[CoreCheckResult]."""
    frame = make_narwhals_frame({"col": [1, 2, 3]})
    schema = _make_schema(
        selector="col",
        checks=[Check.greater_than(0)],
    )

    backend = ColumnBackend()
    results = backend.run_checks(frame, schema)

    assert isinstance(results, list)
    assert len(results) >= 1
    assert all(isinstance(r, CoreCheckResult) for r in results)
    assert results[0].passed is True
