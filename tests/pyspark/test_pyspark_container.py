"""Unit tests for pyspark container."""
from typing import Union

from pyspark.sql import SparkSession
import pyspark.sql.types as T
import pyspark.sql.functions as F
import pytest
import pandera as pa
import pandera.errors
from pandera.api.pyspark.container import DataFrameSchema
from pandera.api.pyspark.components import Column
from pandera.errors import SchemaErrors

spark = SparkSession.builder.getOrCreate()


def test_pyspark_dataframeschema():
    """
    Test creating a pyspark DataFrameSchema object
    """

    schema = DataFrameSchema(
        {
            "name": Column(T.StringType()),
            "age": Column(T.IntegerType(), coerce=True),
        }
    )

    data = [("Neeraj", 35), ("Jask", 30)]

    df = spark.createDataFrame(data=data, schema=["name", "age"])
    schema.report_errors(df)

    data = [("Neeraj", "35"), ("Jask", "a")]

    df2 = spark.createDataFrame(data=data, schema=["name", "age"])

    schema.report_errors(df2)  # typecasted and no error thrown


def test_pyspark_dataframeschema_with_alias_types():
    """
    Test creating a pyspark DataFrameSchema object
    """

    schema = DataFrameSchema(
        columns={
            "product": Column("str", checks=pa.Check.str_startswith("B")),
            "price": Column("int", checks=pa.Check.gt(5)),
        },
        name="product_schema",
        description="schema for product info",
        title="ProductSchema",
    )

    data = [("Bread", 9), ("Butter", 15)]

    spark_schema = T.StructType(
        [
            T.StructField("product", T.StringType(), False),
            T.StructField("price", T.IntegerType(), False),
        ],
    )

    df = spark.createDataFrame(data=data, schema=spark_schema)

    validate_df = schema.report_errors(df)

    with pytest.raises(pandera.errors.PysparkSchemaError):
        data_fail = [("Bread", 3), ("Butter", 15)]

        df_fail = spark.createDataFrame(data=data_fail, schema=spark_schema)

        fail_df = schema.report_errors(df_fail)
        if fail_df:
            raise pandera.errors.PysparkSchemaError


def test_pyspark_column_metadata():
    """
    Test creating a pyspark Column object with metadata
    """

    schema = DataFrameSchema(
        columns={
            "product": Column(
                "str",
                checks=pa.Check.str_startswith("B"),
                metadata={"usecase": "product_pricing", "type": ["t1", "t2"]},
            ),
            "price": Column("int", checks=pa.Check.gt(5)),
        },
        name="product_schema",
        description="schema for product info",
        title="ProductSchema",
    )
    breakpoint()

    data = [("Bread", 9), ("Butter", 15)]

    spark_schema = T.StructType(
        [
            T.StructField("product", T.StringType(), False),
            T.StructField("price", T.IntegerType(), False),
        ],
    )

    df = spark.createDataFrame(data=data, schema=spark_schema)

    validate_df = schema.report_errors(df)

    with pytest.raises(pandera.errors.PysparkSchemaError):
        data_fail = [("Bread", 3), ("Butter", 15)]

        df_fail = spark.createDataFrame(data=data_fail, schema=spark_schema)

        fail_df = schema.report_errors(df_fail)
        if fail_df:
            raise pandera.errors.PysparkSchemaError
