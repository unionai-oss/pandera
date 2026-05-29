"""Shared fixtures for narwhals backend tests.

CI Matrix (TEST-01, TEST-02, TEST-03):

- tests/polars/ and tests/ibis/ run with PANDERA_USE_NARWHALS_BACKEND unset
  (defaults to False), so the native polars/ibis backends are used regardless
  of whether narwhals is installed.
- tests/backends/narwhals/ (this directory) runs WITH narwhals + polars + ibis all
  installed together and PANDERA_USE_NARWHALS_BACKEND=True
  (CI job: unit-tests-narwhals, extra=narwhals). The `make_narwhals_frame` fixture
  below parametrizes every test across the three supported native frame types
  (pl.DataFrame, pl.LazyFrame, ibis.Table) so each test runs 3 times and no frame
  type is silently skipped (TEST-02).

See .github/workflows/ci-tests.yml for the full matrix and .planning/REQUIREMENTS.md
for TEST-01, TEST-02, and TEST-03 definitions.
"""

import os

import narwhals.stable.v1 as nw
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# pyspark is an optional dependency
# ---------------------------------------------------------------------------
try:
    import pyspark.sql
    from pyspark.sql import SparkSession

    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False


@pytest.fixture(autouse=True, scope="module")
def _ensure_narwhals_backends_registered():
    """Initialise narwhals backends before each test module in this directory.

    Clears the lru_cache on the register functions and re-registers so that:
    - builtin_checks side-effect runs (populates Dispatcher._function_registry)
    - NarwhalsCheckBackend, ColumnBackend, DataFrameSchemaBackend are registered
    - Tests that call NarwhalsCheckBackend directly do not need to trigger
      schema.validate() first.

    Requires PANDERA_USE_NARWHALS_BACKEND=True (set by the nox session).
    """
    from pandera.backends.ibis.register import register_ibis_backends
    from pandera.backends.polars.register import register_polars_backends

    register_polars_backends.cache_clear()
    register_ibis_backends.cache_clear()
    register_polars_backends()
    register_ibis_backends()

    try:
        import pyspark.sql  # noqa: F401

        from pandera.backends.pyspark.register import register_pyspark_backends

        register_pyspark_backends.cache_clear()
        register_pyspark_backends()
    except ImportError:
        pass

    yield


@pytest.fixture(
    params=["polars_eager", "polars_lazy", "ibis_table"],
    ids=["polars_eager", "polars_lazy", "ibis_table"],
)
def make_narwhals_frame(request):
    """Return a callable that creates an nw frame across all 3 supported native types.

    TEST-02: parametrizes Narwhals backend tests across pl.DataFrame (eager),
    pl.LazyFrame (lazy), and ibis.Table — all three supported native frame types.
    """
    backend = request.param

    def _make(data: dict):
        if backend == "polars_eager":
            return nw.from_native(pl.DataFrame(data), eager_only=True)
        elif backend == "polars_lazy":
            return nw.from_native(
                pl.LazyFrame(data), eager_or_interchange_only=False
            )
        elif backend == "ibis_table":
            import ibis
            import pandas as pd

            return nw.from_native(
                ibis.memtable(pd.DataFrame(data)),
                eager_or_interchange_only=False,
            )

    return _make


# ===========================================================================
# PySpark fixtures (shared across tests/narwhals/ test modules)
# ===========================================================================


@pytest.fixture(autouse=True, scope="function")
def _spark_env_vars():
    """Set environment variables required by PySpark before each test.

    No-ops when pyspark is not installed so polars/ibis tests are unaffected.
    """
    if not HAS_PYSPARK:
        yield
        return  # noqa: return-after-yield needed to prevent fall-through
    prev = {k: os.environ.get(k) for k in ("SPARK_LOCAL_IP", "PYARROW_IGNORE_TIMEZONE")}
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
    yield
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="module")
def spark():
    """Create a SparkSession for the module, mirroring tests/pyspark/conftest.py."""
    pytest.importorskip("pyspark")
    import pyspark
    from packaging import version

    PYSPARK_VERSION = version.parse(pyspark.__version__)
    builder = SparkSession.builder.config("spark.sql.ansi.enabled", False)
    if PYSPARK_VERSION >= version.parse("4.0.0"):
        builder = builder.config("spark.hadoop.fs.defaultFS", "file:///")
        builder = builder.config(
            "spark.sql.warehouse.dir", "file:///tmp/spark-warehouse"
        )
    spark_session = builder.getOrCreate()
    yield spark_session
    spark_session.stop()
