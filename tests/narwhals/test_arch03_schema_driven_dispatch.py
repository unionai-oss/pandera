"""Behavioral tests for ARCH-03: schema-driven dispatch in ColumnBackend.check_dtype.

ColumnBackend.check_dtype dispatches on schema.dtype type (schema-driven), not on
check_obj.implementation (frame-driven). These tests verify the behavioral contract:

- Narwhals-engine dtype path: polars LazyFrame + narwhals_engine.Int64 schema -> pass
- PySpark-engine dtype path: PySpark frame + pyspark_engine-wrapped T.LongType schema
  -> pass when types match, fail with WRONG_DATATYPE when they don't

Source-inspection tests that used the inspect module to grep ColumnBackend.check_dtype for
internal variable names (is_pyspark, uses_pyspark_dtype, etc.) were removed in Phase 08
because they tested intermediate implementation state, not the behavioral contract.
"""

import os

import polars as pl
import pytest

# ---------------------------------------------------------------------------
# PySpark optional-dependency guard
# ---------------------------------------------------------------------------
try:
    import pyspark.sql
    from pyspark.sql import SparkSession

    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False

pyspark_only = pytest.mark.skipif(not HAS_PYSPARK, reason="pyspark not installed")


# ---------------------------------------------------------------------------
# PySpark fixtures (copied inline from tests/narwhals/test_e2e.py — narwhals
# conftest.py does not define a spark fixture)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Narwhals-engine dtype path (always runs — no pyspark dependency)
# ---------------------------------------------------------------------------


def test_check_dtype_narwhals_schema_takes_narwhals_engine_path():
    """check_dtype with a narwhals-native dtype does NOT attempt PySpark operations.

    A schema configured with narwhals_engine.Int64 should use the narwhals_engine
    comparison path, even if the frame happens to be non-PySpark. This is consistent
    with schema-driven dispatch: the dtype configured in the schema determines the path.
    """
    from types import SimpleNamespace

    import narwhals.stable.v1 as nw
    import polars as pl

    from pandera.backends.narwhals.components import ColumnBackend
    from pandera.engines import narwhals_engine

    frame = nw.from_native(
        pl.LazyFrame({"col": [1, 2, 3]}), eager_or_interchange_only=False
    )
    schema = SimpleNamespace(
        selector="col",
        name="col",
        nullable=True,
        unique=False,
        dtype=narwhals_engine.Int64(),
        checks=[],
    )

    backend = ColumnBackend()
    results = backend.check_dtype(frame, schema)

    # The narwhals path should pass — col is Int64 and schema expects Int64
    assert len(results) == 1
    assert results[0].passed is True, (
        "check_dtype with narwhals_engine.Int64 schema should pass for Int64 column"
    )


# ---------------------------------------------------------------------------
# PySpark-engine dtype path (gated — skipped when pyspark not installed)
# ---------------------------------------------------------------------------


@pyspark_only
def test_check_dtype_pyspark_schema_pass(spark):
    """check_dtype with matching PySpark dtype passes (schema-driven dispatch)."""
    import pyspark.sql.types as T
    from types import SimpleNamespace

    import narwhals.stable.v1 as nw

    from pandera.backends.narwhals.components import ColumnBackend
    from pandera.engines import pyspark_engine

    df = spark.createDataFrame([(1,), (2,)], schema=["col"])
    frame = nw.from_native(df, eager_or_interchange_only=False)
    schema = SimpleNamespace(
        selector="col",
        name="col",
        nullable=True,
        unique=False,
        dtype=pyspark_engine.Engine.dtype(T.LongType()),
        checks=[],
    )
    backend = ColumnBackend()
    results = backend.check_dtype(frame, schema)
    assert len(results) == 1
    assert results[0].passed is True


@pyspark_only
def test_check_dtype_pyspark_schema_fail(spark):
    """check_dtype with mismatched PySpark dtype fails (schema-driven dispatch)."""
    import pyspark.sql.types as T
    from types import SimpleNamespace

    import narwhals.stable.v1 as nw

    from pandera.backends.narwhals.components import ColumnBackend
    from pandera.engines import pyspark_engine
    from pandera.errors import SchemaErrorReason

    df = spark.createDataFrame([(1,), (2,)], schema=["col"])  # LongType column
    frame = nw.from_native(df, eager_or_interchange_only=False)
    schema = SimpleNamespace(
        selector="col",
        name="col",
        nullable=True,
        unique=False,
        dtype=pyspark_engine.Engine.dtype(T.StringType()),  # wrong type
        checks=[],
    )
    backend = ColumnBackend()
    results = backend.check_dtype(frame, schema)
    assert len(results) == 1
    assert results[0].passed is False
    assert results[0].reason_code == SchemaErrorReason.WRONG_DATATYPE
