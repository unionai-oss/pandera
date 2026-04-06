"""Tests for PySpark SQL schema inference."""

import pytest

import pandera.pyspark as pa
from pandera.engines.pyspark_engine import BigInt
from pandera.schema_inference.pyspark import (
    infer_dataframe_schema,
    infer_schema,
)


class _MockSparkDataFrame:
    """Stand-in for ``pyspark.sql.DataFrame`` (no JVM)."""


@pytest.fixture
def mock_spark_dataframe(monkeypatch):
    """Patch ``pyspark.sql.DataFrame`` so isinstance checks accept our mock."""
    monkeypatch.setattr("pyspark.sql.DataFrame", _MockSparkDataFrame)


def test_infer_dataframe_schema(mock_spark_dataframe, monkeypatch) -> None:
    """Build schema from statistics without a live Spark session."""
    df = _MockSparkDataFrame()

    def fake_infer(_df):
        return {
            "columns": {
                "a": {
                    "dtype": BigInt,
                    "nullable": False,
                    "checks": None,
                }
            },
            "index": None,
            "checks": None,
            "coerce": True,
        }

    monkeypatch.setattr(
        "pandera.schema_inference.pyspark.infer_pyspark_dataframe_statistics",
        fake_infer,
    )
    schema = infer_dataframe_schema(df)
    assert isinstance(schema, pa.DataFrameSchema)
    assert schema.coerce is True
    assert set(schema.columns) == {"a"}


def test_infer_schema_alias(mock_spark_dataframe, monkeypatch) -> None:
    """infer_schema delegates to infer_dataframe_schema."""
    df = _MockSparkDataFrame()

    def fake_infer(_df):
        return {
            "columns": {
                "x": {
                    "dtype": BigInt,
                    "nullable": True,
                    "checks": None,
                }
            },
            "index": None,
            "checks": None,
            "coerce": True,
        }

    monkeypatch.setattr(
        "pandera.schema_inference.pyspark.infer_pyspark_dataframe_statistics",
        fake_infer,
    )
    assert (
        infer_schema(df).columns.keys()
        == infer_dataframe_schema(df).columns.keys()
    )


def test_infer_dataframe_schema_wrong_type() -> None:
    """infer_dataframe_schema rejects non-Spark DataFrames."""
    with pytest.raises(TypeError, match="Expected pyspark.sql.DataFrame"):
        infer_dataframe_schema([])  # type: ignore[arg-type]
