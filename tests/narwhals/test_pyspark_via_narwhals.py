"""Cross-backend validation: PySpark DataFrames via the narwhals backend.

These tests demonstrate that a ``pyspark.sql.DataFrame`` can be validated
against a ``pandera.polars`` or ``pandera.ibis`` schema by routing through
the Narwhals backend (``nw.from_native(pyspark_df) -> nw.LazyFrame``).

Notes
-----
- Tests are skipped if PySpark is not installed or a local SparkSession
  cannot be created (e.g. unsupported Java version).
- PySpark DataFrames are SQL-lazy; the default validation depth is
  ``SCHEMA_ONLY``. Tests that exercise data-level checks explicitly opt
  into ``SCHEMA_AND_DATA`` via ``config_context``.
- ``sample=`` and ``tail=`` are not supported on SQL-lazy backends (Spark);
  use ``head=`` only.
"""

from __future__ import annotations

import os

import pytest

from pandera.config import ValidationDepth, config_context
from pandera.errors import SchemaError, SchemaErrors

pyspark = pytest.importorskip("pyspark")

# Configure Spark for local testing before importing SparkSession-related modules
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
os.environ.setdefault("PYARROW_IGNORE_TIMEZONE", "1")

from pyspark.sql import SparkSession  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    """Create a local SparkSession; skip if Java/Spark cannot start."""
    try:
        builder = (
            SparkSession.builder.master("local[1]")
            .appName("pandera-narwhals-pyspark")
            .config("spark.sql.ansi.enabled", False)
            .config("spark.hadoop.fs.defaultFS", "file:///")
            .config("spark.sql.warehouse.dir", "file:///tmp/spark-warehouse")
        )
        session = builder.getOrCreate()
    except Exception as exc:  # pragma: no cover - environmental
        pytest.skip(f"SparkSession could not be created: {exc}")
    yield session
    session.stop()


@pytest.fixture
def sample_df(spark):
    return spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["a", "b"])


# ---------------------------------------------------------------------------
# Polars schemas validating PySpark DataFrames
# ---------------------------------------------------------------------------


def test_polars_schema_validates_pyspark_dataframe(sample_df):
    """Schema-only checks pass for a well-formed pyspark.sql.DataFrame."""
    import pandera.polars as pa

    schema = pa.DataFrameSchema(
        {
            "a": pa.Column(int),
            "b": pa.Column(str),
        }
    )

    out = schema.validate(sample_df)
    # input type is preserved on success
    assert type(out).__name__ == "DataFrame"
    assert out.schema == sample_df.schema


def test_polars_schema_dtype_mismatch_raises(spark):
    """Wrong dtype is detected at SCHEMA scope."""
    import pandera.polars as pa

    df = spark.createDataFrame([(1.0,), (2.0,)], ["x"])
    schema = pa.DataFrameSchema({"x": pa.Column(int)})

    with pytest.raises(SchemaError, match="expected column 'x' to have type"):
        schema.validate(df)


def test_polars_schema_strict_extra_column(spark):
    """``strict=True`` rejects unexpected columns."""
    import pandera.polars as pa

    df = spark.createDataFrame([(1, "a", True)], ["a", "b", "extra"])
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int), "b": pa.Column(str)}, strict=True
    )

    with pytest.raises(SchemaError, match="not in DataFrameSchema"):
        schema.validate(df)


def test_polars_schema_strict_filter_drops_extra_column(spark):
    """``strict='filter'`` returns the schema columns only."""
    import pandera.polars as pa

    df = spark.createDataFrame(
        [(1, "a", True), (2, "b", False)], ["a", "b", "extra"]
    )
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int), "b": pa.Column(str)}, strict="filter"
    )

    out = schema.validate(df)
    assert out.columns == ["a", "b"]


def test_polars_schema_data_check_skipped_by_default(spark):
    """Data-level checks default to SCHEMA_ONLY for lazy/SQL-lazy frames."""
    import pandera.polars as pa

    # Failing data: 0 < 1 violates ge(1)
    df = spark.createDataFrame([(0,), (1,), (2,)], ["a"])
    schema = pa.DataFrameSchema({"a": pa.Column(int, pa.Check.ge(1))})

    # Default validation depth for pyspark (SQL-lazy) is SCHEMA_ONLY,
    # so the data-level check ``ge(1)`` is skipped.
    out = schema.validate(df)
    assert type(out).__name__ == "DataFrame"


def test_polars_schema_data_check_runs_when_forced(spark):
    """Forcing SCHEMA_AND_DATA runs data checks on pyspark frames."""
    import pandera.polars as pa

    df = spark.createDataFrame([(0,), (1,), (2,)], ["a"])
    schema = pa.DataFrameSchema({"a": pa.Column(int, pa.Check.ge(1))})

    with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
        with pytest.raises(SchemaError):
            schema.validate(df)


def test_polars_schema_unique_check_runs_when_forced(spark):
    """``unique=True`` is a data-level check; runs when depth is forced."""
    import pandera.polars as pa

    df = spark.createDataFrame([(1,), (1,), (2,)], ["a"])
    schema = pa.DataFrameSchema({"a": pa.Column(int, unique=True)})

    with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
        with pytest.raises(SchemaError, match="not unique"):
            schema.validate(df)


def test_polars_schema_lazy_collects_multiple_errors(spark):
    """``lazy=True`` returns all errors via SchemaErrors."""
    import pandera.polars as pa

    df = spark.createDataFrame([(0,), (1,)], ["a"])
    schema = pa.DataFrameSchema(
        {
            "a": pa.Column(int, pa.Check.ge(5)),
            "b": pa.Column(str),  # missing
        }
    )

    with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
        with pytest.raises(SchemaErrors):
            schema.validate(df, lazy=True)


def test_polars_schema_head_subsample(spark):
    """``head=`` is supported on SQL-lazy backends."""
    import pandera.polars as pa

    df = spark.createDataFrame([(i,) for i in range(10)], ["a"])
    schema = pa.DataFrameSchema({"a": pa.Column(int)})

    out = schema.validate(df, head=3)
    assert type(out).__name__ == "DataFrame"


def test_polars_schema_tail_unsupported_for_pyspark(spark):
    """``tail=`` raises NotImplementedError on SQL-lazy backends."""
    import pandera.polars as pa

    df = spark.createDataFrame([(i,) for i in range(5)], ["a"])
    schema = pa.DataFrameSchema({"a": pa.Column(int)})

    with pytest.raises(NotImplementedError, match="tail="):
        schema.validate(df, tail=2)


def test_polars_schema_sample_unsupported(spark):
    """``sample=`` is not supported in the narwhals backend."""
    import pandera.polars as pa

    df = spark.createDataFrame([(i,) for i in range(5)], ["a"])
    schema = pa.DataFrameSchema({"a": pa.Column(int)})

    with pytest.raises(NotImplementedError, match="sample="):
        schema.validate(df, sample=2)


# ---------------------------------------------------------------------------
# Ibis schemas validating PySpark DataFrames
# ---------------------------------------------------------------------------


def test_ibis_schema_validates_pyspark_dataframe(sample_df):
    """An ibis schema can validate a pyspark.sql.DataFrame."""
    pytest.importorskip("ibis")
    import pandera.ibis as pa

    schema = pa.DataFrameSchema(
        {
            "a": pa.Column(int),
            "b": pa.Column(str),
        }
    )

    out = schema.validate(sample_df)
    assert type(out).__name__ == "DataFrame"


def test_ibis_schema_dtype_mismatch_raises(spark):
    """Wrong dtype is detected via ibis schema as well."""
    pytest.importorskip("ibis")
    import pandera.ibis as pa

    df = spark.createDataFrame([(1.0,), (2.0,)], ["x"])
    schema = pa.DataFrameSchema({"x": pa.Column(int)})

    with pytest.raises(SchemaError, match="expected column 'x' to have type"):
        schema.validate(df)


def test_ibis_schema_strict_filter_drops_extra_column(spark):
    """``strict='filter'`` works through the ibis -> narwhals backend."""
    pytest.importorskip("ibis")
    import pandera.ibis as pa

    df = spark.createDataFrame(
        [(1, "a", True), (2, "b", False)], ["a", "b", "extra"]
    )
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int), "b": pa.Column(str)}, strict="filter"
    )

    out = schema.validate(df)
    assert out.columns == ["a", "b"]
