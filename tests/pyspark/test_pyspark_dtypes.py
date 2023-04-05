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


def test_pyspark_dtype_int():
    """
    Test int dtype column
    """

    pandera_schema = DataFrameSchema(
        columns={
            "product": Column("str"),
            "price": Column("int"),
        },
        name="product_schema",
        description="schema for product info",
        title="ProductSchema",
    )

    spark_schema = T.StructType(
        [
            T.StructField("product", T.StringType(), False),
            T.StructField("price", T.IntegerType(), False),
        ],
    )

    data = [("Bread", 9), ("Butter", 15)]

    df = spark.createDataFrame(data=data, schema=spark_schema, verifySchema=False)
    pandera_schema.validate(df)

    with pytest.raises(SchemaError):
        spark_schema = T.StructType(
            [
                T.StructField("product", T.StringType(), False),
                T.StructField("price", T.FloatType(), False),
            ],
        )
        data = [("Bread", 9.0), ("Butter", 15)]

        df_fail = spark.createDataFrame(
            data=data, schema=spark_schema, verifySchema=False
        )

        pandera_schema.validate(df_fail)
