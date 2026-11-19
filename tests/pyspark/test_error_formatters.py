"""Test error messages shared with the pandas backend."""

import pyspark.pandas as ps
import pytest

import pandera.pandas as pa


def test_pyspark_pandas_failure_message(spark):
    """A failed pandas-on-Spark check reports all failing values."""
    amounts = ps.Series([1.0, -5.0, -5.0], name="amount")
    schema = pa.SeriesSchema(float, pa.Check.gt(0), name="amount")

    with pytest.raises(pa.errors.SchemaError) as exc:
        schema.validate(amounts)

    assert str(exc.value) == (
        "SeriesSchema 'amount' failed element-wise validator number 0: "
        "greater_than(0) failure cases: -5.0, -5.0"
    )
