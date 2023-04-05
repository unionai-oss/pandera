"""Unit tests for pyspark container."""
from pyspark.sql import SparkSession
import pyspark.sql.types as T
import pyspark.sql.functions as F
import pytest
import pandera as pa
from pandera.api.pyspark.container import DataFrameSchema
from pandera.api.pyspark.components import Column
from pandera.error_handlers import SchemaError

spark = SparkSession.builder.getOrCreate()


def test_pyspark_dtype_string():
    """
    Test string dtype column
    """

    pandera_schema = DataFrameSchema(
        columns={
            "product": Column("str"),
        },
        name="product_schema",
        description="schema for product info",
        title="ProductSchema",
    )

    data = [("Bread"), ("Butter")]

    spark_schema = T.StructType(
        [
            T.StructField("product", T.StringType(), False),
        ],
    )

    df = spark.createDataFrame(data=data, schema=spark_schema)

    validate_df = pandera_schema.validate(df)
