"""Unit tests for pyspark container."""
import datetime

import pyspark.sql.types as T
import pytest
from pandera.api.pyspark.container import DataFrameSchema
from pandera.api.pyspark.components import Column
from pandera.error_handlers import SchemaError
from tests.pyspark.conftest import spark_df
from pandera.errors import SchemaErrors
from pyspark.sql import DataFrame

def validate_datatype(spark, sample_spark_schema, sample_data, pandera_schema):

    df = spark_df(spark, sample_data, sample_spark_schema)

    pandera_schema.validate(df)

    validated_data = pandera_schema(df)

    # negative test
    #with pytest.raises(SchemaErrors):

    assert df.pandera.schema == pandera_schema
    assert isinstance(pandera_schema.validate(df), DataFrame)
    assert validated_data.pandera.schema == pandera_schema


def test_pyspark_dtype_int(spark, sample_data, sample_spark_schema):
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

    validate_datatype(spark, sample_spark_schema, sample_data, pandera_schema)

    # negative test
    with pytest.raises(SchemaErrors):
        spark_schema_fail = T.StructType(
            [
                T.StructField("product", T.StringType(), False),
                T.StructField("price", T.FloatType(), False),
            ],
        )
        data_fail = [("Bread", 9.0), ("Butter", 15)]

        df_fail = spark_df(spark, data_fail, spark_schema_fail)

        pandera_schema.validate(df_fail)


def test_pyspark_all_integer_types(spark):
    """
    Test int dtype column
    """

    pandera_schema = DataFrameSchema(
        columns={
            "productid": Column("long"),
            "price": Column("int"),
            "hash": Column("short"),
            "rating": Column("byte"),
        },
        name="product_schema",
        description="schema for product info",
        title="ProductSchema",
    )
    sample_data = [(123435451123, 40000, 2600, 100), (123435451145, 35000, 2500, 124)]
    sample_spark_schema = T.StructType(
        [
            T.StructField("productid", T.LongType(), False),
            T.StructField("price", T.IntegerType(), False),
            T.StructField("hash", T.ShortType(), False),
            T.StructField("rating", T.ByteType(), False),
        ],
    )
    validate_datatype(spark, sample_spark_schema, sample_data, pandera_schema)

def test_pyspark_all_float_types(spark):
    """
    Test int dtype column
    """

    pandera_schema = DataFrameSchema(
        columns={
            "productid": Column("int"),
            "price": Column("double"),
            "rating": Column("float"),
            "weight": Column("decimal"),
            "height":  Column(T.DecimalType(20, 4))
        },
        name="product_schema",
        description="schema for product info",
        title="ProductSchema",
    )
    sample_data = [(123435451123, 40000.0, 7.5, 4.56, 5.7876), (123435451145, 35000.0, 9.5, 10.23, 768.7643413)]
    sample_spark_schema = T.StructType(
        [
            T.StructField("productid", T.IntegerType(), False),
            T.StructField("price", T.DoubleType(), False),
            T.StructField("rating", T.FloatType(), False),
            T.StructField("weight", T.DecimalType(), False),
            T.StructField("height", T.DecimalType(20, 4))
        ],
    )

    validate_datatype(spark, sample_spark_schema, sample_data, pandera_schema)

    # negative test
    with pytest.raises(SchemaErrors):
        sample_spark_schema_fail = T.StructType(
            [
                T.StructField("productid", T.IntegerType(), False),
                T.StructField("price", T.FloatType(), False),
                T.StructField("rating", T.DoubleType(), False),
                T.StructField("weight", T.DecimalType(), False),
                T.StructField("height", T.DecimalType(20, 6))
            ],
        )
        df_fail = spark_df(spark, sample_data, sample_spark_schema_fail)

        pandera_schema.validate(df_fail)


def test_pyspark_all_datetime_types(spark):
    """
    Test int dtype column
    """

    pandera_schema = DataFrameSchema(
        columns={
            "purchase_date": Column("date"),
            "purchase_datetime": Column("datetime"),
            "expiry_time": Column("timedelta"),
            "expected_time": Column(T.DayTimeIntervalType(2, 3)),
        },
        name="product_schema",
        description="schema for product info",
        title="ProductSchema",
    )
    sample_data = [(datetime.date(2022, 10, 1), datetime.datetime(2022, 10, 1, 5, 32, 0), datetime.timedelta(45),  datetime.timedelta(45)),
                   (datetime.date(2022, 11, 5), datetime.datetime(2022, 11, 5, 15, 34, 0), datetime.timedelta(30),  datetime.timedelta(45))]
    sample_spark_schema = T.StructType(
        [
            T.StructField("purchase_date", T.DateType(), False),
            T.StructField("purchase_datetime", T.TimestampType(), False),
            T.StructField("expiry_time", T.DayTimeIntervalType(), False),
            T.StructField("expected_time", T.DayTimeIntervalType(2,3), False),
        ],
    )

    validate_datatype(spark, sample_spark_schema, sample_data, pandera_schema)
    with pytest.raises(SchemaErrors):
        sample_spark_fail_schema = T.StructType(
            [
                T.StructField("purchase_date", T.DateType(), False),
                T.StructField("purchase_datetime", T.TimestampType(), False),
                T.StructField("expiry_time", T.DayTimeIntervalType(), False),
                T.StructField("expected_time", T.DayTimeIntervalType(1, 3), False),
            ],
        )
        validate_datatype(spark, sample_spark_fail_schema, sample_data, pandera_schema)
def test_pyspark_binary_types(spark):
    """
    Test int dtype column
    """

    pandera_schema = DataFrameSchema(
        columns={
            "binary": Column("binary"),
        },
        name="product_schema",
        description="schema for product info",
        title="ProductSchema",
    )
    sample_data = [(datetime.date(2022, 10, 1), datetime.datetime(2022, 10, 1, 5, 32, 0), datetime.timedelta(45)),
                   (datetime.date(2022, 11, 5), datetime.datetime(2022, 11, 5, 15, 34, 0), datetime.timedelta(30))]
    sample_spark_schema = T.StructType(
        [
            T.StructField("binary", T.BinaryType(), False),
        ],
    )

    validate_datatype(spark, sample_spark_schema, sample_data, pandera_schema)



def test_pyspark_array_type(spark):
    """
    Test int dtype column
    """

    pandera_schema = DataFrameSchema(
        columns={
            "purchase_date": Column("date"),
            "customer_details": Column(T.ArrayType(T.StringType(), T.ShortType())),
        },
        name="product_schema",
        description="schema for product info",
        title="ProductSchema",
    )
    sample_data = [(datetime.date(2022, 10, 1),  ["josh", 27]),
                   (datetime.date(2022, 11, 5),  ["Adam", 22])]

    sample_spark_schema = T.StructType(
        [
            T.StructField("purchase_date", T.DateType(), False),
            T.StructField("customer_details", T.ArrayType(T.StringType(), ), False),
        ],
    )
    schema = T.StructType([
        T.StructField("User", T.IntegerType()),
        T.StructField("My_array", T.ArrayType(
            T.StructType([
                T.StructField("user", T.StringType()),
                T.StructField("product", T.StringType()),
                T.StructField("rating", T.DoubleType())
            ])
        )
    )
    ])
    validate_datatype(spark, sample_spark_schema, sample_data, pandera_schema)
    with pytest.raises(SchemaErrors):
        sample_spark_fail_schema = T.StructType(
            [
                T.StructField("purchase_date", T.DateType(), False),
                T.StructField("purchase_datetime", T.TimestampType(), False),
                T.StructField("customer_details", T.ArrayType(T.StringType, T.LongType), False)
            ],
        )
        validate_datatype(spark, sample_spark_fail_schema, sample_data, pandera_schema)
def test_pyspark_binary_types(spark):
    """
    Test int dtype column
    """

    pandera_schema = DataFrameSchema(
        columns={
            "binary": Column("binary"),
        },
        name="product_schema",
        description="schema for product info",
        title="ProductSchema",
    )
    sample_data = [(datetime.date(2022, 10, 1), datetime.datetime(2022, 10, 1, 5, 32, 0), datetime.timedelta(45)),
                   (datetime.date(2022, 11, 5), datetime.datetime(2022, 11, 5, 15, 34, 0), datetime.timedelta(30))]
    sample_spark_schema = T.StructType(
        [
            T.StructField("binary", T.BinaryType(), False),
        ],
    )

    validate_datatype(spark, sample_spark_schema, sample_data, pandera_schema)
