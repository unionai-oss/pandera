"""Tests for ColumnBackend (pandera/backends/narwhals/components.py)."""

from types import SimpleNamespace

import narwhals.stable.v1 as nw
import polars as pl
import pytest

from pandera.api.checks import Check
from pandera.backends.base import CoreCheckResult
from pandera.backends.narwhals.base import NarwhalsSchemaBackend
from pandera.backends.narwhals.components import ColumnBackend
from pandera.errors import SchemaError, SchemaErrorReason

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
# check_nullable tests
# ---------------------------------------------------------------------------


def test_check_nullable_passes(make_narwhals_frame):
    """nullable=True column with no nulls passes."""
    frame = make_narwhals_frame({"col": [1, 2, 3]})
    schema = _make_schema(selector="col", nullable=True)

    backend = ColumnBackend()
    results = backend.check_nullable(frame, schema)

    assert len(results) == 1
    assert results[0].passed is True


def test_check_nullable_fails_on_null(make_narwhals_frame):
    """nullable=False column with None values fails with SERIES_CONTAINS_NULLS."""
    frame = make_narwhals_frame({"col": [1, None, 3]})
    schema = _make_schema(selector="col", nullable=False)

    backend = ColumnBackend()
    results = backend.check_nullable(frame, schema)

    assert len(results) == 1
    assert results[0].passed is False
    assert results[0].reason_code == SchemaErrorReason.SERIES_CONTAINS_NULLS


def test_check_nullable_catches_nan(make_narwhals_frame):
    """nullable=False float column with NaN fails — NaN treated as null."""
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


def test_check_unique_passes(make_narwhals_frame):
    """unique=True column with all distinct values passes."""
    frame = make_narwhals_frame({"col": [1, 2, 3]})
    schema = _make_schema(selector="col", unique=True)

    backend = ColumnBackend()
    results = backend.check_unique(frame, schema)

    assert len(results) == 1
    assert results[0].passed is True


def test_check_unique_fails(make_narwhals_frame):
    """unique=True column with duplicate values fails with SERIES_CONTAINS_DUPLICATES."""
    frame = make_narwhals_frame({"col": [1, 2, 2]})
    schema = _make_schema(selector="col", unique=True)

    backend = ColumnBackend()
    results = backend.check_unique(frame, schema)

    assert len(results) == 1
    assert results[0].passed is False
    assert (
        results[0].reason_code == SchemaErrorReason.SERIES_CONTAINS_DUPLICATES
    )


# ---------------------------------------------------------------------------
# check_dtype tests
# ---------------------------------------------------------------------------


def test_check_dtype_correct(make_narwhals_frame):
    """column dtype matching schema.dtype passes."""
    from pandera.engines import narwhals_engine

    frame = make_narwhals_frame({"col": [1, 2, 3]})
    schema = _make_schema(selector="col", dtype=narwhals_engine.Int64())

    backend = ColumnBackend()
    results = backend.check_dtype(frame, schema)

    assert len(results) == 1
    assert results[0].passed is True


def test_check_dtype_wrong(make_narwhals_frame):
    """column dtype not matching schema.dtype fails with WRONG_DATATYPE."""
    from pandera.engines import narwhals_engine

    # Frame has Int64 but schema expects Float64
    frame = make_narwhals_frame({"col": [1, 2, 3]})
    schema = _make_schema(selector="col", dtype=narwhals_engine.Float64())

    backend = ColumnBackend()
    results = backend.check_dtype(frame, schema)

    assert len(results) == 1
    assert results[0].passed is False
    assert results[0].reason_code == SchemaErrorReason.WRONG_DATATYPE


def test_check_dtype_none(make_narwhals_frame):
    """schema.dtype=None short-circuits and returns passed=True."""
    frame = make_narwhals_frame({"col": [1, 2, 3]})
    schema = _make_schema(selector="col", dtype=None)

    backend = ColumnBackend()
    results = backend.check_dtype(frame, schema)

    assert len(results) == 1
    assert results[0].passed is True


# ---------------------------------------------------------------------------
# run_checks test
# ---------------------------------------------------------------------------


def test_run_checks(make_narwhals_frame):
    """run_checks executes Check objects and returns list[CoreCheckResult]."""
    import narwhals.stable.v1 as nw

    import pandera.backends.narwhals.builtin_checks  # noqa: F401 — registers nw.Expr dispatcher
    from pandera.backends.narwhals.checks import NarwhalsCheckBackend

    # Ensure narwhals frame types are registered with the Check backend.
    # In CI this is done by PANDERA_USE_NARWHALS_BACKEND=True; here we
    # register directly so the test works in any environment.
    Check.register_backend(nw.DataFrame, NarwhalsCheckBackend)
    Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend)

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


# ---------------------------------------------------------------------------
# subsample() lazy-first contracts
# ---------------------------------------------------------------------------

# Guard: skip ibis-specific tests if ibis is not installed
try:
    import ibis as _ibis_mod

    HAS_IBIS = True
except ImportError:
    HAS_IBIS = False

ibis_only = pytest.mark.skipif(not HAS_IBIS, reason="ibis not installed")


class TestSubsample:
    """Tests for subsample() lazy-first contracts."""

    def _polars_lazy_frame(self):
        """Return an nw.LazyFrame backed by polars."""
        return nw.from_native(
            pl.LazyFrame({"x": [1, 2, 3, 4, 5]}),
            eager_or_interchange_only=False,
        )

    def test_subsample_head_stays_lazy(self):
        """LAZY-FIRST: subsample(head=2) must return nw.LazyFrame, not nw.DataFrame."""
        frame = self._polars_lazy_frame()
        backend = NarwhalsSchemaBackend()
        result = backend.subsample(frame, head=2)
        assert isinstance(result, nw.LazyFrame), (
            f"expected nw.LazyFrame from subsample(head=), got {type(result)}"
        )

    def test_subsample_tail_stays_lazy(self):
        """LAZY-FIRST: subsample(tail=2) must return nw.LazyFrame, not nw.DataFrame."""
        frame = self._polars_lazy_frame()
        backend = NarwhalsSchemaBackend()
        result = backend.subsample(frame, tail=2)
        assert isinstance(result, nw.LazyFrame), (
            f"expected nw.LazyFrame from subsample(tail=), got {type(result)}"
        )

    def test_subsample_both_head_and_tail(self):
        """LAZY-FIRST: subsample(head=2, tail=2) must return nw.LazyFrame."""
        frame = self._polars_lazy_frame()
        backend = NarwhalsSchemaBackend()
        result = backend.subsample(frame, head=2, tail=2)
        assert isinstance(result, nw.LazyFrame), (
            f"expected nw.LazyFrame from subsample(head=, tail=), got {type(result)}"
        )

    @ibis_only
    def test_subsample_ibis_tail_raises(self):
        """SQL-lazy backends: subsample(tail=) must raise NotImplementedError."""
        import pandas as pd

        ibis_frame = nw.from_native(
            _ibis_mod.memtable(pd.DataFrame({"x": [1, 2, 3, 4, 5]})),
            eager_or_interchange_only=False,
        )
        backend = NarwhalsSchemaBackend()
        with pytest.raises(NotImplementedError, match="tail="):
            backend.subsample(ibis_frame, tail=2)

    def test_subsample_no_params_returns_unchanged(self):
        """subsample() with no params returns the original frame unchanged."""
        frame = self._polars_lazy_frame()
        backend = NarwhalsSchemaBackend()
        result = backend.subsample(frame)
        assert result is frame

    def test_subsample_sample_eager_polars(self):
        """sample= works on eager polars frames wrapped as nw.DataFrame."""
        frame = nw.from_native(
            pl.DataFrame({"x": list(range(20))}),
            eager_only=True,
        )
        backend = NarwhalsSchemaBackend()
        result = backend.subsample(frame, sample=5, random_state=42)
        assert isinstance(result, nw.DataFrame)
        native = nw.to_native(result)
        assert native.shape[0] == 5
        # Deterministic for the same seed (unique() may reorder the rows,
        # so compare the row set rather than the exact row order).
        result2 = backend.subsample(frame, sample=5, random_state=42)
        assert set(nw.to_native(result2)["x"].to_list()) == set(
            native["x"].to_list()
        )

    def test_subsample_sample_lazy_raises(self):
        """sample= on a polars LazyFrame raises NotImplementedError."""
        frame = self._polars_lazy_frame()
        backend = NarwhalsSchemaBackend()
        with pytest.raises(NotImplementedError, match="sample="):
            backend.subsample(frame, sample=2)

    @ibis_only
    def test_subsample_sample_ibis_raises(self):
        """sample= on a SQL-lazy backend (ibis) raises NotImplementedError."""
        import pandas as pd

        ibis_frame = nw.from_native(
            _ibis_mod.memtable(pd.DataFrame({"x": [1, 2, 3, 4, 5]})),
            eager_or_interchange_only=False,
        )
        backend = NarwhalsSchemaBackend()
        with pytest.raises(NotImplementedError, match="sample="):
            backend.subsample(ibis_frame, sample=2)

    def test_subsample_head_and_sample_combined(self):
        """head= and sample= can be combined on an eager polars frame.

        ``sample(5)`` draws from the *full* frame, not from head(10), so the
        union of head(10) + sample(5) contains at most 15 distinct rows
        (fewer if the sample happens to overlap with the head).
        """
        frame = nw.from_native(
            pl.DataFrame({"x": list(range(20))}),
            eager_only=True,
        )
        backend = NarwhalsSchemaBackend()
        result = backend.subsample(frame, head=10, sample=5, random_state=0)
        native = nw.to_native(result)
        n_rows = native.shape[0]
        assert 10 <= n_rows <= 15
        xs = set(native["x"].to_list())
        # head(10) values must all be present.
        assert set(range(10)).issubset(xs)


def test_validate_with_sample_works_on_eager_polars(monkeypatch, request):
    """``schema.validate(eager_df, sample=N)`` runs end-to-end via the
    Narwhals container.

    Regression test: the container originally always passed
    ``nw.LazyFrame`` to ``subsample()``, which made the eager-only
    ``sample=`` path unreachable in the public API. The container now
    materializes the LazyFrame back to ``nw.DataFrame`` when the caller
    originally passed an eager polars frame.
    """
    import pandera.polars as pa
    from pandera.backends.polars.register import register_polars_backends
    from pandera.config import CONFIG

    monkeypatch.setattr(CONFIG, "use_narwhals_backend", True)
    request.addfinalizer(register_polars_backends.cache_clear)
    register_polars_backends.cache_clear()
    register_polars_backends()

    schema = pa.DataFrameSchema({"x": pa.Column(pl.Int64)})
    df = pl.DataFrame({"x": list(range(20))})
    out = schema.validate(df, sample=5)
    assert isinstance(out, pl.DataFrame)
    assert out.shape == (20, 1)


def test_validate_with_sample_on_lazyframe_raises(monkeypatch, request):
    """``sample=`` on a LazyFrame still raises (sample is eager-only)."""
    import pandera.polars as pa
    from pandera.backends.polars.register import register_polars_backends
    from pandera.config import CONFIG

    monkeypatch.setattr(CONFIG, "use_narwhals_backend", True)
    request.addfinalizer(register_polars_backends.cache_clear)
    register_polars_backends.cache_clear()
    register_polars_backends()

    schema = pa.DataFrameSchema({"x": pa.Column(pl.Int64)})
    lf = pl.LazyFrame({"x": list(range(20))})
    with pytest.raises(NotImplementedError, match="sample="):
        schema.validate(lf, sample=5)


# ---------------------------------------------------------------------------
# failure_cases_metadata() ibis return type
# ---------------------------------------------------------------------------


@ibis_only
def test_failure_cases_metadata_ibis_returns_ibis_table():
    """failure_cases_metadata() preserves ibis.Table for ibis input."""
    import ibis
    import pandas as pd

    failure_cases_df = nw.from_native(
        ibis.memtable(pd.DataFrame({"x": [-1, -3]})),
        eager_or_interchange_only=False,
    )

    # Build a schema stub whose __class__.__name__ == "Column" so that
    # failure_cases_metadata() can use err.schema.__class__.__name__ safely.
    ColumnStub = type("Column", (), {})
    schema_stub = ColumnStub()
    schema_stub.name = "x"

    schema_error = SchemaError(
        schema=schema_stub,
        data=None,
        message="Check failed",
        failure_cases=failure_cases_df,
        check=None,
        check_index=0,
        check_output=None,
        reason_code=SchemaErrorReason.DATAFRAME_CHECK,
    )

    backend = NarwhalsSchemaBackend()
    result = backend.failure_cases_metadata("test_schema", [schema_error])

    assert isinstance(result.failure_cases, ibis.Table), (
        f"expected ibis.Table for ibis input, got {type(result.failure_cases)}"
    )
