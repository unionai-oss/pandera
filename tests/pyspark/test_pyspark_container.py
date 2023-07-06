"""Unit tests for pyspark container."""

from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.types as T
import pytest
import pandera.pyspark as pa
import pandera.errors
from pandera.config import PanderaConfig, ValidationDepth
from pandera.pyspark import DataFrameSchema, Column

spark = SparkSession.builder.getOrCreate()


def test_pyspark_dataframeschema():
    """
    Test creating a pyspark DataFrameSchema object
    """

    schema = DataFrameSchema(
        {
            "name": Column(T.StringType()),
            "age": Column(T.IntegerType(), coerce=True, nullable=True),
        }
    )

    data = [("Neeraj", 35), ("Jask", 30)]

    df = spark.createDataFrame(data=data, schema=["name", "age"])
    df_out = schema.validate(df)

    assert df_out.pandera.errors is not None

    data = [("Neeraj", "35"), ("Jask", "a")]

    df2 = spark.createDataFrame(data=data, schema=["name", "age"])

    df_out = schema.validate(df2)

    assert not df_out.pandera.errors


def test_pyspark_dataframeschema_with_alias_types(
    config_params: PanderaConfig,
):
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

    df_out = schema.validate(df)

    assert not df_out.pandera.errors
    if config_params.validation_depth in [
        ValidationDepth.SCHEMA_AND_DATA,
        ValidationDepth.DATA_ONLY,
    ]:
        with pytest.raises(pandera.errors.PysparkSchemaError):
            data_fail = [("Bread", 3), ("Butter", 15)]

            df_fail = spark.createDataFrame(
                data=data_fail, schema=spark_schema
            )

            fail_df = schema.validate(df_fail)
            if fail_df.pandera.errors:
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
        metadata={"category": "product"},
    )

    expected = {
        "product_schema": {
            "columns": {
                "product": {
                    "usecase": "product_pricing",
                    "type": ["t1", "t2"],
                },
                "price": None,
            },
            "dataframe": {"category": "product"},
        }
    }

    assert schema.get_metadata() == expected


def test_pyspark_sample():
    """
    Test the sample functionality of pyspark
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

    data = [
        ("Bread", 9),
        ("Butter", 15),
        ("Ice Cream", 10),
        ("Cola", 12),
        ("Choclate", 7),
    ]

    spark_schema = T.StructType(
        [
            T.StructField("product", T.StringType(), False),
            T.StructField("price", T.IntegerType(), False),
        ],
    )

    df = spark.createDataFrame(data=data, schema=spark_schema)

    df_out = schema.validate(df, sample=0.5)

    assert isinstance(df_out, DataFrame)
