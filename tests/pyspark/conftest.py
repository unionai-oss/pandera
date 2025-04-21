"""conftest"""

# pylint:disable=redefined-outer-name
import datetime
import os

import pyspark.sql.types as T
import pytest
from pyspark.sql import SparkSession

from pandera.config import PanderaConfig


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """
    creates spark session
    """
    spark: SparkSession = SparkSession.builder.getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def spark_connect() -> SparkSession:
    """
    creates spark connection session
    """
    # Set location of localhost Spark Connect server
    os.environ["SPARK_LOCAL_REMOTE"] = "sc://localhost"
    spark: SparkSession = SparkSession.builder.getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def sample_data():
    """
    provides sample data
    """
    return [("Bread", 9), ("Butter", 15)]


@pytest.fixture(scope="session")
def sample_spark_schema():
    """
    provides spark schema for sample data
    """
    return T.StructType(
        [
            T.StructField("product", T.StringType(), True),
            T.StructField("price", T.IntegerType(), True),
        ],
    )


def spark_df(spark, data: list, spark_schema: T.StructType):
    """This function creates spark dataframe from given data and schema object"""
    return spark.createDataFrame(
        data=data, schema=spark_schema, verifySchema=False
    )


@pytest.fixture(scope="session")
def sample_date_object(spark):
    """This fundtion contains sample data for datetime types object"""
    sample_data = [
        (
            datetime.date(2022, 10, 1),
            datetime.datetime(2022, 10, 1, 5, 32, 0),
        ),
        (
            datetime.date(2022, 11, 5),
            datetime.datetime(2022, 11, 5, 15, 34, 0),
        ),
    ]
    sample_spark_schema = T.StructType(
        [
            T.StructField("purchase_date", T.DateType(), False),
            T.StructField("purchase_datetime", T.TimestampType(), False),
        ],
    )
    df = spark_df(
        spark=spark, spark_schema=sample_spark_schema, data=sample_data
    )
    return df


@pytest.fixture(scope="session")
def sample_string_binary_object(spark):
    """This function creates the sample data for binary types"""
    sample_data = [
        (
            "test1",
            "Bread",
        ),
        ("test2", "Butter"),
    ]
    sample_spark_schema = T.StructType(
        [
            T.StructField("purchase_info", T.StringType(), False),
            T.StructField("product", T.StringType(), False),
        ],
    )
    df = spark_df(
        spark=spark, spark_schema=sample_spark_schema, data=sample_data
    )
    df = df.withColumn(
        "purchase_info", df["purchase_info"].cast(T.BinaryType())
    )
    return df


@pytest.fixture(scope="session")
def sample_complex_data(spark):
    """This function creates sample data for complex datatypes types"""
    sample_data = [
        (
            datetime.date(2022, 10, 1),
            [["josh"], ["27"]],
            {"product_bought": "bread"},
        ),
        (
            datetime.date(2022, 11, 5),
            [["Adam"], ["22"]],
            {"product_bought": "bread"},
        ),
    ]

    sample_spark_schema = T.StructType(
        [
            T.StructField("purchase_date", T.DateType(), False),
            T.StructField(
                "customer_details",
                T.ArrayType(
                    T.ArrayType(T.StringType()),
                ),
                False,
            ),
            T.StructField(
                "product_details",
                T.MapType(T.StringType(), T.StringType()),
                False,
            ),
        ],
    )
    return spark_df(spark, sample_data, sample_spark_schema)


@pytest.fixture(scope="session")
def sample_check_data():
    """This creates data for check type"""
    return {
        "test_pass_data": [("foo", 30), ("bar", 30)],
        "test_fail_data": [("foo", 30), ("bar", 31)],
        "test_expression": 30,
    }


@pytest.fixture(scope="session")
def config_params():
    """This function creates config parameters"""
    return PanderaConfig()
