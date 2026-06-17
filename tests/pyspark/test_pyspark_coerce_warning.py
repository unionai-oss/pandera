"""Tests for the narwhals-backend SchemaWarning emitted for coerce=True on PySpark.

Column-level coerce=True is a no-op across all narwhals backends (Polars, Ibis,
PySpark SQL).  These tests confirm the warning fires for PySpark under the
narwhals backend, matching the cross-backend contract from test_polars_coerce_warning.py.
"""

from __future__ import annotations

import warnings

import pyspark.sql.types as T
import pytest

from pandera.config import CONFIG
from pandera.errors import SchemaErrors, SchemaWarning
from pandera.pyspark import Column, DataFrameSchema
from tests.pyspark.conftest import spark_df

pytestmark = pytest.mark.parametrize(
    "spark_session", ["spark", "spark_connect"]
)

SKIP_NON_NARWHALS = pytest.mark.skipif(
    not CONFIG.use_narwhals_backend,
    reason="SchemaWarning for coerce=True is narwhals-backend-only",
)


@SKIP_NON_NARWHALS
def test_pyspark_coerce_true_dtype_mismatch_emits_schema_warning(
    spark_session, request
):
    """coerce=True on a mismatched PySpark column emits a SchemaWarning."""
    spark = request.getfixturevalue(spark_session)

    schema = DataFrameSchema(
        {"value": Column(T.LongType(), coerce=True)},
    )
    # IntegerType column vs LongType schema: dtype mismatch → warn + error
    data = [(1,), (2,)]
    spark_schema = T.StructType(
        [T.StructField("value", T.IntegerType(), False)]
    )
    df = spark_df(spark, data, spark_schema)

    with pytest.warns(SchemaWarning, match="coerce=True is not applied"):
        with pytest.raises(SchemaErrors):
            schema.validate(df, lazy=True)


@SKIP_NON_NARWHALS
def test_pyspark_coerce_false_no_schema_warning(spark_session, request):
    """No SchemaWarning when coerce is not set on a PySpark column."""
    spark = request.getfixturevalue(spark_session)

    schema = DataFrameSchema(
        {"value": Column(T.LongType())},
    )
    data = [(1,), (2,)]
    spark_schema = T.StructType(
        [T.StructField("value", T.IntegerType(), False)]
    )
    df = spark_df(spark, data, spark_schema)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            schema.validate(df, lazy=True)
        except Exception:
            pass

    schema_warnings = [
        w for w in caught if issubclass(w.category, SchemaWarning)
    ]
    assert len(schema_warnings) == 0
