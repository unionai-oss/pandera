"""Unit tests for pyspark_accessor module."""

from typing import Union

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType, LongType

import pandera.pyspark as pa
from pandera.config import PanderaConfig, ValidationDepth
from pandera.pyspark import pyspark_sql_accessor

spark = SparkSession.builder.getOrCreate()


@pytest.mark.parametrize(
    "schema1, schema2, data, invalid_data",
    [
        [
            pa.DataFrameSchema({"col": pa.Column("long")}, coerce=True),
            pa.DataFrameSchema({"col": pa.Column("float")}, coerce=False),
            spark.createDataFrame([{"col": 1}, {"col": 2}, {"col": 3}]),
            spark.createDataFrame([{"col": 1}, {"col": 2}, {"col": 3}]),
        ],
    ],
)
def test_dataframe_add_schema(
    schema1: pa.DataFrameSchema,
    schema2: pa.DataFrameSchema,
    data: Union[DataFrame, col],
    invalid_data: Union[DataFrame, col],
    config_params: PanderaConfig,
) -> None:
    """
    Test that pyspark object contains schema metadata after pandera validation.
    """
    schema1(data)  # type: ignore[arg-type]

    assert data.pandera.schema == schema1
    assert isinstance(schema1.validate(data), DataFrame)
    assert isinstance(schema1(data), DataFrame)
    if config_params.validation_depth != ValidationDepth.DATA_ONLY:
        assert dict(schema2(invalid_data).pandera.errors["SCHEMA"]) == {
            "WRONG_DATATYPE": [
                {
                    "schema": None,
                    "column": "col",
                    "check": f"dtype('{str(FloatType())}')",
                    "error": f"expected column 'col' to have type {str(FloatType())}, got {str(LongType())}",
                }
            ]
        }  # type: ignore[arg-type]


class CustomAccessor:
    """Mock accessor class"""

    def __init__(self, obj):
        self._obj = obj


def test_modin_accessor_warning():
    """Test that modin accessor raises warning when name already exists."""
    pyspark_sql_accessor.register_dataframe_accessor("foo")(CustomAccessor)
    with pytest.warns(UserWarning):
        pyspark_sql_accessor.register_dataframe_accessor("foo")(CustomAccessor)
