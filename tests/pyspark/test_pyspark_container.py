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

    import pdb

    pdb.set_trace()


def test_pyspark_dataframeschema_with_alias_types():
    """
    Test creating a pyspark DataFrameSchema object
    """

    schema = DataFrameSchema(
        columns={
            "name": Column("str"),
            "age": Column("int"),
        },
        name="person_schema",
        description="schema for person info",
        title="PersonSchema",
    )

    data = [("Neeraj", 35), ("Jask", 30)]

    df = spark.createDataFrame(data=data, schema=["name", "age"])

    validate_df = schema.validate(df)
    import pdb

    pdb.set_trace()
