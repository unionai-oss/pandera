"""Unit tests for dask_accessor module."""
from typing import Union

from pyspark.sql import DataFrame
from pyspark.sql.functions import col
import pytest
from pyspark.sql import SparkSession
import pandera as pa
from pandera import pyspark_sql_accessor
from pandera.error_handlers import SchemaError

from pyspark.sql.types import StringType, LongType
from pandera.api.pyspark.container import DataFrameSchema
from pandera.api.pyspark.components import Column
from tests.pyspark.conftest import spark_df

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
            spark.createDataFrame(
                data=[(23, 31), (34, 30)], schema=["product", "code"]
            ),
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
    # validated_data_1 = schema(data)  # type: ignore[arg-type]
    # print(schema2.validate(invalid_data))
    # print(schema1.validate(invalid_data))

    # with pytest.raises(SchemaError):
    schema(invalid_data, lazy=True)  # type: ignore[arg-type]


def test_pyspark_check_eq(spark, sample_spark_schema):
    """
    Test creating a pyspark DataFrameSchema object
    """

    pandera_schema = DataFrameSchema(
        columns={
            "product": Column("str"),
            "price": Column("int", checks=pa.Check.eq(5)),
        },
        name="product_schema",
        description="schema for product info",
        title="ProductSchema",
    )

    data_fail = [("Bread", 5), ("Butter", 15)]
    df_fail = spark_df(spark, data_fail, sample_spark_schema)
    pandera_schema.validate(df_fail)
    breakpoint()
