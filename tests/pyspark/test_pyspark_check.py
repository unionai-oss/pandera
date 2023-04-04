
"""Unit tests for pyspark container."""
from typing import Union

from pyspark.sql import SparkSession
import copy
from pyspark.sql.types import LongType, StringType, StructField, StructType, IntegerType
import pyspark.sql.functions as F
import pytest
import pandera as pa
from pandera.api.pyspark.container import DataFrameSchema
from pandera.api.pyspark.components import Column
from pandera.error_handlers import SchemaError

spark = SparkSession.builder.getOrCreate()

def test_equal_to_check() -> None:
    """Test the Check to see if all the values are equal to defined value"""

    schema = DataFrameSchema(
        {
            "product": Column(StringType()),
            "code": Column(LongType(), pa.Check.equal_to(30)),
        }
    )

    data = [("foo", 30), ("bar", 30)]
    df = spark.createDataFrame(data=data, schema=["product", "code"])
    validate_df = schema.validate(df)

    with pytest.raises(SchemaError):
        data_fail = [("foo", 31), ("bar", 30)]
        df_fail = spark.createDataFrame(data=data_fail, schema=["product", "code"])
        validate_fail_df = schema.validate(df_fail)




def test_pyspark_check_eq():
    """
    Test creating a pyspark DataFrameSchema object
    """

    schema = DataFrameSchema(
        columns={
            "product": Column("str"),
            "price": Column("int", checks=pa.Check.eq(5)),
        },
        name="product_schema",
        description="schema for product info",
        title="ProductSchema",
    )


    spark_schema = StructType(
        [
            StructField("product", StringType(), False),
            StructField("price", IntegerType(), False),
        ],
    )
    with pytest.raises(SchemaError):
        data = [("Bread", 5), ("Butter", 15)]
        df = spark.createDataFrame(data=data, schema=spark_schema)
        validate_df = schema.validate(df)




def test_not_equal_to_check() -> None:
    """Test the Check to see if any value is not in the specified value"""

    schema_not_equal_to = DataFrameSchema(
        {
            "product": Column(StringType()),
            "code": Column(LongType(), pa.Check.not_equal_to(30)),
        }
    )

    data = [("foo", 34), ("bar", 35)]
    df = spark.createDataFrame(data=data, schema=["product", "code"])
    validate_df = schema_not_equal_to.validate(df)

    with pytest.raises(SchemaError):
        data_fail = [("foo", 31), ("bar", 30)]
        df_fail = spark.createDataFrame(data=data_fail, schema=["product", "code"])
        validate_fail_df = schema_not_equal_to.validate(df_fail)

    # Validate the alias name check works
    schema_ne = DataFrameSchema(
        columns={
            "product": Column("str"),
            "price": Column("int", checks=pa.Check.ne(5)),
        },
        name="product_schema",
        description="schema for product info",
        title="ProductSchema",
    )

    data = [("Bread", 7), ("Butter", 15)]

    spark_schema = StructType(
        [
            StructField("product", StringType(), False),
            StructField("price", IntegerType(), False),
        ],
    )

    df = spark.createDataFrame(data=data, schema=spark_schema)

    validate_df = schema_ne.validate(df)

    with pytest.raises(SchemaError):
        data_fail = [("foo", 31), ("bar", 30)]
        df_fail = spark.createDataFrame(data=data_fail, schema=["product", "price"])
        validate_fail_df = schema_ne.validate(df_fail)


def test_greater_than_check() -> None:
    """Test the Check to see if any value is not in the specified value"""

    schema_greater_than = DataFrameSchema(
        {
            "product": Column(StringType()),
            "code": Column(LongType(), pa.Check.greater_than(30)),
        }
    )

    data = [("foo", 34), ("bar", 35)]
    df = spark.createDataFrame(data=data, schema=["product", "code"])
    validate_df = schema_greater_than.validate(df)

    with pytest.raises(SchemaError):
        data_fail = [("foo", 29), ("bar", 30)]
        df_fail = spark.createDataFrame(data=data_fail, schema=["product", "code"])
        validate_fail_df = schema_greater_than.validate(df_fail)

    # Validate the alias name check works
    schema_gt = DataFrameSchema(
        columns={
            "product": Column("str"),
            "price": Column("int", checks=pa.Check.gt(5)),
        },
    )

    data = [("Bread", 7), ("Butter", 15)]

    spark_schema = StructType(
        [
            StructField("product", StringType(), False),
            StructField("price", IntegerType(), False),
        ],
    )

    df = spark.createDataFrame(data=data, schema=spark_schema)

    validate_df = schema_gt.validate(df)

    with pytest.raises(SchemaError):
        data = [("Bread", 3), ("Butter", 15)]
        df_fail = spark.createDataFrame(data=data_fail, schema=["product", "price"])
        validate_fail_df = schema_gt.validate(df_fail)



def test_greater_than_or_equal_to_check() -> None:
    """Test the Check to see if any value is not in the specified value"""

    schema_greater_than_or_equal_t = DataFrameSchema(
        {
            "product": Column(StringType()),
            "code": Column(LongType(), pa.Check.greater_than_or_equal_to(30)),
        }
    )

    data = [("foo", 30), ("bar", 35)]
    df = spark.createDataFrame(data=data, schema=["product", "code"])
    validate_df = schema_greater_than_or_equal_t.validate(df)

    with pytest.raises(SchemaError):
        data_fail = [("foo", 29), ("bar", 30)]
        df_fail = spark.createDataFrame(data=data_fail, schema=["product", "code"])
        validate_fail_df = schema_greater_than_or_equal_t.validate(df_fail)

    # Validate the alias name check works
    schema_ge = DataFrameSchema(
        columns={
            "product": Column("str"),
            "price": Column("int", checks=pa.Check.ge(5)),
        },
    )

    data = [("Bread", 5), ("Butter", 15)]

    spark_schema = StructType(
        [
            StructField("product", StringType(), False),
            StructField("price", IntegerType(), False),
        ],
    )

    df = spark.createDataFrame(data=data, schema=spark_schema)

    validate_df = schema_ge.validate(df)

    with pytest.raises(SchemaError):
        data = [("Bread", 3), ("Butter", 15)]
        df_fail = spark.createDataFrame(data=data_fail, schema=["product", "price"])
        validate_fail_df = schema_ge.validate(df_fail)


def test_less_than_check() -> None:
    """Test the Check to see if any value is not in the specified value"""

    schema_less_than = DataFrameSchema(
        {
            "product": Column(StringType()),
            "code": Column(LongType(), pa.Check.less_than(30)),
        }
    )

    data = [("foo", 20), ("bar", 15)]
    df = spark.createDataFrame(data=data, schema=["product", "code"])
    validate_df = schema_less_than.validate(df)

    with pytest.raises(SchemaError):
        data_fail = [("foo", 29), ("bar", 30)]
        df_fail = spark.createDataFrame(data=data_fail, schema=["product", "code"])
        validate_fail_df = schema_less_than.validate(df_fail)

    # Validate the alias name check works
    schema_lt = DataFrameSchema(
        columns={
            "product": Column("str"),
            "price": Column("int", checks=pa.Check.lt(5)),
        },
    )

    data = [("Bread", 3), ("Butter", 1)]

    spark_schema = StructType(
        [
            StructField("product", StringType(), False),
            StructField("price", IntegerType(), False),
        ],
    )

    df = spark.createDataFrame(data=data, schema=spark_schema)

    validate_df = schema_lt.validate(df)

    with pytest.raises(SchemaError):
        data = [("Bread", 3), ("Butter", 15)]
        df_fail = spark.createDataFrame(data=data_fail, schema=["product", "price"])
        validate_fail_df = schema_lt.validate(df_fail)


def test_less_than_equal_to_check() -> None:
    """Test the Check to see if any value is not in the specified value"""

    schema_less_than = DataFrameSchema(
        {
            "product": Column(StringType()),
            "code": Column(LongType(), pa.Check.less_than_or_equal_to(20)),
        }
    )

    data = [("foo", 20), ("bar", 15)]
    df = spark.createDataFrame(data=data, schema=["product", "code"])
    validate_df = schema_less_than.validate(df)

    with pytest.raises(SchemaError):
        data_fail = [("foo", 31), ("bar", 30)]
        df_fail = spark.createDataFrame(data=data_fail, schema=["product", "code"])
        validate_fail_df = schema_less_than.validate(df_fail)

    # Validate the alias name check works
    schema_lt = DataFrameSchema(
        columns={
            "product": Column("str"),
            "price": Column("int", checks=pa.Check.le(5)),
        },
    )

    data = [("Bread", 3), ("Butter", 1)]

    spark_schema = StructType(
        [
            StructField("product", StringType(), False),
            StructField("price", IntegerType(), False),
        ],
    )

    df = spark.createDataFrame(data=data, schema=spark_schema)

    validate_df = schema_lt.validate(df)

    with pytest.raises(SchemaError):
        data = [("Bread", 3), ("Butter", 15)]
        df_fail = spark.createDataFrame(data=data_fail, schema=["product", "price"])
        validate_fail_df = schema_lt.validate(df_fail)



def test_isin_check() -> None:
    """Test the Check to see if any value is not in the specified value"""

    schema = DataFrameSchema(
        {
            "product": Column(StringType()),
            "code": Column(LongType(), pa.Check.isin([20, 15, 45])),
        }
    )

    data = [("foo", 20), ("bar", 15)]
    df = spark.createDataFrame(data=data, schema=["product", "code"])
    validate_df = schema.validate(df)

    with pytest.raises(SchemaError):
        data_fail = [("foo", 20), ("bar", 30)]
        df_fail = spark.createDataFrame(data=data_fail, schema=["product", "code"])
        validate_fail_df = schema.validate(df_fail)


def test_notin_check() -> None:
    """Test the Check to see if any value is not in the specified value"""

    schema = DataFrameSchema(
        {
            "product": Column(StringType()),
            "code": Column(LongType(), pa.Check.notin([20, 15, 45])),
        }
    )

    data = [("foo", 25), ("bar", 35)]
    df = spark.createDataFrame(data=data, schema=["product", "code"])
    validate_df = schema.validate(df)

    with pytest.raises(SchemaError):
        data_fail = [("foo", 20), ("bar", 30)]
        df_fail = spark.createDataFrame(data=data_fail, schema=["product", "code"])
        validate_fail_df = schema.validate(df_fail)


def test_str_startswith_check() -> None:
    """Test the Check to see if any value is not in the specified value"""

    schema = DataFrameSchema(
        {
            "product": Column(StringType(), pa.Check.str_startswith('B')),
            "code": Column(LongType()),
        }
    )

    data = [("Bread", 25), ("Butter", 35)]
    df = spark.createDataFrame(data=data, schema=["product", "code"])
    validate_df = schema.validate(df)

    with pytest.raises(SchemaError):
        data_fail = [("Bread", 25), ("Jam", 35)]
        df_fail = spark.createDataFrame(data=data_fail, schema=["product", "code"])
        validate_fail_df = schema.validate(df_fail)


def test_str_endswith_check() -> None:
    """Test the Check to see if any value is not in the specified value"""

    schema = DataFrameSchema(
        {
            "product": Column(StringType(), pa.Check.str_endswith('t')),
            "code": Column(LongType()),
        }
    )

    data = [("Bat", 25), ("cat", 35)]
    df = spark.createDataFrame(data=data, schema=["product", "code"])
    validate_df = schema.validate(df)

    with pytest.raises(SchemaError):
        data_fail = [("Bread", 25), ("Jam", 35)]
        df_fail = spark.createDataFrame(data=data_fail, schema=["product", "code"])
        validate_fail_df = schema.validate(df_fail)



def test_str_contains_check() -> None:
    """Test the Check to see if any value is not in the specified value"""

    schema = DataFrameSchema(
        {
            "product": Column(StringType(), pa.Check.str_contains('Ba')),
            "code": Column(LongType()),
        }
    )

    data = [("Bat!", 25), ("Bat78", 35)]
    df = spark.createDataFrame(data=data, schema=["product", "code"])
    validate_df = schema.validate(df)
    with pytest.raises(SchemaError):
        data_fail = [("Cs", 25), ("Jam!", 35)]
        df_fail = spark.createDataFrame(data=data_fail, schema=["product", "code"])
        validate_fail_df = schema.validate(df_fail)
