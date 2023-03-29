"""Unit tests for pyspark container."""
from typing import Union

from pyspark.sql import SparkSession
import pyspark.sql.types as T
import pyspark.sql.functions as F
import pytest
import pandera as pa
from pandera.api.pyspark.container import DataFrameSchema
from pandera.api.pyspark.components import Column

spark = SparkSession.builder.getOrCreate()


def test_pyspark_dataframeschema():
    """
    Test creating a pyspark DataFrameSchema object
    """

    schema = DataFrameSchema(
        {
            "name": Column(T.StringType()),
            "age": Column(T.IntegerType()),
        }
    )

    data = [("Neeraj", 35), ("Jask", 30)]

    df = spark.createDataFrame(data=data, schema=["name", "age"])
    validate_df = schema.validate(df)

    data = [("Neeraj", "35"), ("Jask", "a")]

    df2 = spark.createDataFrame(data=data, schema=["name", "age"])
    validate_df2 = schema.validate(df2)  # typecasted and no error thrown


def test_pyspark_dataframeschema_with_alias_types():
    """
    Test creating a pyspark DataFrameSchema object
    """

    schema = DataFrameSchema(
        columns={
            "first_name": Column("str"),
            "age": Column("int", pa.Check.gt(31)),
        },
        name="person_schema",
        description="schema for person info",
        title="PersonSchema",
    )

    data = [("Neeraj", 35), ("Jask", 30)]

    spark_schema = T.StructType(
        [
            T.StructField("first_name", T.StringType(), False),
            T.StructField("age", T.IntegerType(), False),
        ],
    )

    df = spark.createDataFrame(data=data, schema=spark_schema)

    validate_df = schema.validate(df)

    validate_df.show()

    breakpoint()
