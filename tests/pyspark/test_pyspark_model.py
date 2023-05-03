"""Unit tests for DataFrameModel module."""
from pprint import pprint
from pyspark.sql.functions import col
import pyspark.sql.types as T
import pytest
import pandera as pa
from pandera import SchemaModel
from pandera.api.pyspark.model import DataFrameModel
from pandera.api.pyspark.container import DataFrameSchema
from pandera.api.pyspark.model_components import Field
from tests.pyspark.conftest import spark_df


def test_schema_with_bare_types():
    """
    Test that DataFrameModel can be defined without generics.
    """

    class Model(DataFrameModel):
        a: int
        b: str
        c: float

    expected = pa.DataFrameSchema(
        name="Model",
        columns={
            "a": pa.Column(int),
            "b": pa.Column(str),
            "c": pa.Column(float),
        },
    )

    assert expected == Model.to_schema()


def test_schema_with_bare_types_and_field():
    """
    Test that DataFrameModel can be defined without generics.
    """

    class Model(DataFrameModel):
        a: int = Field()
        b: str = Field()
        c: float = Field()

    expected = DataFrameSchema(
        name="Model",
        columns={
            "a": pa.Column(int),
            "b": pa.Column(str),
            "c": pa.Column(float),
        },
    )

    assert expected == Model.to_schema()


def test_schema_with_bare_types_field_and_checks(spark):
    """
    Test that DataFrameModel can be defined without generics.
    """

    class Model(DataFrameModel):
        a: str = Field(str_startswith="B")
        b: int = Field(gt=6)
        c: float = Field()

    expected = DataFrameSchema(
        name="Model",
        columns={
            "a": pa.Column(str, checks=pa.Check.str_startswith("B")),
            "b": pa.Column(int, checks=pa.Check.gt(6)),
            "c": pa.Column(float),
        },
    )

    assert expected == Model.to_schema()

    data_fail = [("Bread", 5, "Food"), ("Cutter", 15, 99.99)]

    spark_schema = T.StructType(
        [
            T.StructField("a", T.StringType(), False),  # should fail
            T.StructField("b", T.IntegerType(), False),  # should fail
            T.StructField("c", T.FloatType(), False),
        ],
    )

    df_fail = spark_df(spark, data_fail, spark_schema)
    df_out = Model.report_errors(check_obj=df_fail)
    assert df_out.pandera.errors != None


def test_schema_with_bare_types_field_type(spark):
    """
    Test that DataFrameModel can be defined without generics.
    """

    class Model(DataFrameModel):
        a: str = Field(str_startswith="B")
        b: int = Field(gt=6)
        c: float = Field()

    data_fail = [("Bread", 5, "Food"), ("Cutter", 15, 99.99)]

    spark_schema = T.StructType(
        [
            T.StructField("a", T.StringType(), False),  # should fail
            T.StructField("b", T.IntegerType(), False),  # should fail
            T.StructField("c", T.StringType(), False),  # should fail
        ],
    )

    df_fail = spark_df(spark, data_fail, spark_schema)
    df_out = Model.report_errors(check_obj=df_fail)
    assert df_out.pandera.errors != None


def test_pyspark_bare_fields(spark):
    """
    Test schema and data level checks
    """

    class pandera_schema(DataFrameModel):
        id: T.IntegerType() = Field(gt=5)
        product_name: T.StringType() = Field(str_startswith="B")
        price: T.DecimalType(20, 5) = Field()
        description: T.ArrayType(T.StringType()) = Field()
        meta: T.MapType(T.StringType(), T.StringType()) = Field()

    data_fail = [
        (5, "Bread", 44.4, ["description of product"], {"product_category": "dairy"}),
        (15, "Butter", 99.0, ["more details here"], {"product_category": "bakery"}),
    ]

    spark_schema = T.StructType(
        [
            T.StructField("id", T.IntegerType(), False),
            T.StructField("product", T.StringType(), False),
            T.StructField("price", T.DecimalType(20, 5), False),
            T.StructField("description", T.ArrayType(T.StringType(), False), False),
            T.StructField(
                "meta", T.MapType(T.StringType(), T.StringType(), False), False
            ),
        ],
    )
    df_fail = spark_df(spark, data_fail, spark_schema)
    df_out = pandera_schema.report_errors(check_obj=df_fail)
    assert df_out.pandera.errors != None
