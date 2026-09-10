"""Tests for _concat_failure_cases PySpark dispatch paths.

Covers the PySpark unwrap-and-union branch and the SchemaWarning emitted when
scalar pl.DataFrame items (from _build_scalar_failure_case) cannot be merged
into a PySpark result.

These tests run in the tests_narwhals_backend(extra='pyspark') nox session.
"""

import warnings

import pytest

pytest.importorskip("narwhals")
pytest.importorskip("polars")
pytest.importorskip("pyspark")

import narwhals.stable.v1 as nw  # noqa: E402
import polars as pl  # noqa: E402
import pyspark.sql as pyspark_sql  # noqa: E402
import pyspark.sql.types as T  # noqa: E402

from pandera.backends.narwhals.base import _concat_failure_cases  # noqa: E402
from pandera.errors import SchemaWarning  # noqa: E402


def test_concat_failure_cases_pyspark_unions_nw_items(spark):
    """PySpark branch unwraps nw_items to native PySpark DataFrames and unions them."""
    schema = T.StructType([T.StructField("val", T.LongType(), True)])
    df_a = spark.createDataFrame([(1,), (2,)], schema=schema)
    df_b = spark.createDataFrame([(3,)], schema=schema)

    nw_a = nw.from_native(df_a, eager_or_interchange_only=False)
    nw_b = nw.from_native(df_b, eager_or_interchange_only=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _concat_failure_cases([nw_a, nw_b])

    assert not any(issubclass(w.category, SchemaWarning) for w in caught)
    assert isinstance(result, pyspark_sql.DataFrame)
    assert result.count() == 3
    assert sorted(r.val for r in result.collect()) == [1, 2, 3]


def test_concat_failure_cases_pyspark_warns_and_skips_pl_items(spark):
    """PySpark branch emits SchemaWarning and drops scalar pl.DataFrame items.

    Scalar pl.DataFrame items (from _build_scalar_failure_case) cannot be
    converted to PySpark without a live SparkSession.  The function warns and
    omits them from the returned PySpark DataFrame.
    """
    schema = T.StructType([T.StructField("val", T.LongType(), True)])
    df_pyspark = spark.createDataFrame([(1,), (2,)], schema=schema)
    nw_item = nw.from_native(df_pyspark, eager_or_interchange_only=False)

    # Mimic what _build_scalar_failure_case produces: a pl.DataFrame with a
    # "column" field so the warning message can name the affected column.
    pl_item = pl.DataFrame(
        {"column": ["some_col"], "failure_case": ["bad_val"]}
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _concat_failure_cases([nw_item, pl_item])

    schema_warnings = [
        w for w in caught if issubclass(w.category, SchemaWarning)
    ]
    assert len(schema_warnings) == 1
    assert "some_col" in str(schema_warnings[0].message)

    assert isinstance(result, pyspark_sql.DataFrame)
    assert result.count() == 2
    # stacklevel=6 is calibrated for the production chain (validate →
    # SchemaErrors.__init__ → failure_cases_metadata → _concat_failure_cases →
    # warnings.warn). When called directly here the stack is only 2 frames deep,
    # so the warning overflows to a "sys" pseudo-frame. This assertion is a
    # necessary-but-not-sufficient guard: it catches stacklevel=1 (which would
    # attribute the warning to base.py) but cannot verify the exact level.
    assert not schema_warnings[0].filename.endswith("base.py"), (
        "SchemaWarning points at base.py internals; stacklevel is too low"
    )


def test_concat_failure_cases_no_warning_when_no_column_field(spark):
    """No SchemaWarning when pl_item has no 'column' field.

    Finding 3 (empty-guard): if pl_items are present but none carry a 'column'
    field, dropped_info stays empty and no SchemaWarning should fire.
    """
    schema = T.StructType([T.StructField("val", T.LongType(), True)])
    df_pyspark = spark.createDataFrame([(1,), (2,)], schema=schema)
    nw_item = nw.from_native(df_pyspark, eager_or_interchange_only=False)

    # pl_item without a "column" field — dropped_info will stay empty
    pl_item = pl.DataFrame({"failure_case": ["bad_val"]})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _concat_failure_cases([nw_item, pl_item])

    schema_warnings = [
        w for w in caught if issubclass(w.category, SchemaWarning)
    ]
    assert len(schema_warnings) == 0, (
        "SchemaWarning must not fire when no pl_item has a 'column' field"
    )
    assert isinstance(result, pyspark_sql.DataFrame)
    assert result.count() == 2
