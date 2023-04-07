"""Unit tests for dask_accessor module."""
from typing import Union

from pyspark.sql import DataFrame
from pyspark.sql.functions import col
import pytest
from pyspark.sql import SparkSession
import pandera as pa
from pandera import pyspark_sql_accessor
from pandera.error_handlers import SchemaError

from pyspark.sql.types import StringType,LongType

spark = SparkSession.builder.getOrCreate()
@pytest.mark.parametrize(
    "schema, invalid_data",
    [
        [
            pa.DataFrameSchema(
        {
            "product": pa.Column(StringType()),
            "code": pa.Column(LongType(), pa.Check.not_equal_to(30)),
        }
    ),
            spark.createDataFrame(data=[(23, 31), (34, 30)], schema=["product", "code"]),
        ],
    ],
)
def test_dataframe_add_schema(
    schema: pa.DataFrameSchema,
    invalid_data: Union[DataFrame, col],
) -> None:
    """
    Test that pandas object contains schema metadata after pandera validation.
    """
    #validated_data_1 = schema(data)  # type: ignore[arg-type]
    #print(schema2.validate(invalid_data))
    #print(schema1.validate(invalid_data))



    #with pytest.raises(SchemaError):
    schema(invalid_data, lazy=True)  # type: ignore[arg-type]
