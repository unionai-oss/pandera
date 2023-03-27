"""Unit tests for dask_accessor module."""
from typing import Union

from pyspark.sql import DataFrame
from pyspark.sql.functions import col
import pytest
from pyspark.sql import SparkSession
import pandera as pa


spark = SparkSession.builder.getOrCreate()
@pytest.mark.parametrize(
    "schema1, schema2, data, invalid_data",
    [
        [
            pa.DataFrameSchema({"col": pa.Column(int)}, coerce=True),
            pa.DataFrameSchema({"col": pa.Column(float)}, coerce=True),
            spark.createDataFrame([{"col": [1, 2, 3]}]),
            spark.createDataFrame([{"col": [1, 2, 3]}])
        ],
    ],
)
def test_dataframe_series_add_schema(
    schema1: pa.DataFrameSchema,
    schema2: pa.DataFrameSchema,
    data: Union[DataFrame, col],
    invalid_data: Union[DataFrame, col],
) -> None:
    """
    Test that pandas object contains schema metadata after pandera validation.
    """
    validated_data_1 = schema1(data)  # type: ignore[arg-type]
    assert data.pandera.schema == schema1

    assert validated_data_1.pandera.schema == schema1


    with pytest.raises(TypeError):
        schema1(invalid_data)  # type: ignore[arg-type]
