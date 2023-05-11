""" conftest """
import os
import pytest
from pyspark.sql import SparkSession
import pyspark.sql.types as T
import datetime
from pandera.backends.pyspark.utils import ConfigParams
from pandera.pyspark import DataFrameSchema, Column, DataFrameModel, Field
import pandera

@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """
    creates spark session
    """
    return SparkSession.builder.getOrCreate()


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
    return spark.createDataFrame(data=data, schema=spark_schema, verifySchema=False)


@pytest.fixture(scope='session')
def sample_date_object(spark):
    sample_data = [
        (
            datetime.date(2022, 10, 1),
            datetime.datetime(2022, 10, 1, 5, 32, 0),
            datetime.timedelta(45),
            datetime.timedelta(45),
        ),
        (
            datetime.date(2022, 11, 5),
            datetime.datetime(2022, 11, 5, 15, 34, 0),
            datetime.timedelta(30),
            datetime.timedelta(45),
        ),
    ]
    sample_spark_schema = T.StructType(
        [
            T.StructField("purchase_date", T.DateType(), False),
            T.StructField("purchase_datetime", T.TimestampType(), False),
            T.StructField("expiry_time", T.DayTimeIntervalType(), False),
            T.StructField("expected_time", T.DayTimeIntervalType(2, 3), False),
        ],
    )
    df = spark_df(spark=spark, spark_schema=sample_spark_schema, data=sample_data)
    return df

@pytest.fixture(scope='session')
def sample_string_binary_object(spark):
    sample_data = [
        (
            'test1',
            'Bread',
        ),
        (
            'test2',
            "Butter"
        ),
    ]
    sample_spark_schema = T.StructType(
        [
            T.StructField("purchase_info", T.StringType(), False),
            T.StructField("product", T.StringType(), False),
        ],
    )
    df = spark_df(spark=spark, spark_schema=sample_spark_schema, data=sample_data)
    df = df.withColumn('purchase_info', df['purchase_info'].cast(T.BinaryType()))
    return df

@pytest.fixture(scope='session')
def sample_complex_data(spark):
    sample_data = [
        (datetime.date(2022, 10, 1), [["josh"], ["27"]], {"product_bought": "bread"}),
        (datetime.date(2022, 11, 5), [["Adam"], ["22"]], {"product_bought": "bread"}),
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
                T.MapType(
                    T.StringType(),
                    T.StringType()
                ),
                False,
            ),
        ],
    )
    return spark_df(spark, sample_data, sample_spark_schema)

@pytest.fixture(scope='session')
def sample_check_data(spark):

    return {"test_pass_data": [("foo", 30), ("bar", 30)],
     "test_fail_data": [("foo", 30), ("bar", 31)],
     "test_expression": 30}

@pytest.fixture(scope='session')
def config_params():
    return ConfigParams()

def test_config_params(spark, sample_spark_schema, monkeypatch):

    monkeypatch.setenv('VALIDATION', 'DISABLE')
    monkeypatch.setenv('DEPTH', 'SCHEMA_AND_DATA')
    class TestDataFrameSchema(DataFrameSchema):
        pass
    sample_data = [("Bread", 9), ("Cutter", 15)]

    pandra_schema = TestDataFrameSchema(
            {
                "product": Column(T.StringType(), pandera.Check.str_startswith("B")),
                "price_val": Column(T.IntegerType()),
            }
        )


    class TestSchema(DataFrameModel):
        product: T.StringType() = Field(str_startswith='B')
        price_val: T.StringType() = Field()

    params = ConfigParams()
    expected = {'VALIDATION': 'DISABLE', 'DEPTH': 'SCHEMA_AND_DATA'}
    assert dict(params) == expected
    input_df = spark_df(spark, sample_data, sample_spark_schema)

    assert pandra_schema.validate(input_df) is None
    assert TestSchema.validate(input_df) is None
    monkeypatch.setenv('VALIDATION', 'ENABLE')
    monkeypatch.setenv('DEPTH', 'SCHEMA_ONLY')
    params = ConfigParams()
    pandra_schema = DataFrameSchema(
            {
                "product": Column(T.StringType(), pandera.Check.str_startswith("B")),
                "price_val": Column(T.IntegerType()),
            }
        )

    expected = {'VALIDATION': 'ENABLE', 'DEPTH': 'SCHEMA_ONLY'}
    assert dict(params) == expected
    breakpoint()

