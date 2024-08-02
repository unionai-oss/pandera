"""Unit tests for dask_accessor module."""

# pylint:disable=redefined-outer-name,abstract-method

import pyspark.sql.types as T
import pytest
from pyspark.sql.types import StringType

import pandera.pyspark as pa
from pandera.api.base import error_handler
from pandera.errors import SchemaError, SchemaErrorReason
from pandera.pyspark import Column, DataFrameModel, DataFrameSchema, Field
from tests.pyspark.conftest import spark_df

pytestmark = pytest.mark.parametrize(
    "spark_session", ["spark", "spark_connect"]
)


def test_dataframe_add_schema(
    spark_session,
    request,
) -> None:
    """
    Test that pyspark object contains schema metadata after pandera validation.
    """
    spark = request.getfixturevalue(spark_session)
    schema = pa.DataFrameSchema(
        {
            "product": pa.Column(StringType()),
            "code": pa.Column(StringType(), pa.Check.not_equal_to(30)),
        }
    )
    invalid_data = spark.createDataFrame(
        data=[("23", 31), ("34", 35)], schema=["product", "code"]
    )
    schema(invalid_data, lazy=True)  # type: ignore[arg-type]


def test_pyspark_check_eq(spark_session, sample_spark_schema, request):
    """Test creating a pyspark DataFrameSchema object"""
    spark = request.getfixturevalue(spark_session)
    pandera_schema = DataFrameSchema(
        columns={
            "product": Column("str", checks=pa.Check.str_startswith("B")),
            "price": Column("int", checks=pa.Check.gt(5)),
        },
        name="product_schema",
        description="schema for product info",
        title="ProductSchema",
    )

    data_fail = [("Bread", 5), ("Cutter", 15)]
    df_fail = spark_df(spark, data_fail, sample_spark_schema)
    df_out = pandera_schema.validate(check_obj=df_fail)
    expected = {
        "DATAFRAME_CHECK": [
            {
                "check": "str_startswith('B')",
                "column": "product",
                "error": "column "
                "'product' "
                "with type "
                f"{str(StringType())} "
                "failed "
                "validation "
                "str_startswith('B')",
                "schema": "product_schema",
            },
            {
                "check": "greater_than(5)",
                "column": "price",
                "error": "column "
                "'price' with "
                "type "
                f"{str(T.IntegerType())} "
                "failed "
                "validation "
                "greater_than(5)",
                "schema": "product_schema",
            },
        ]
    }
    assert dict(df_out.pandera.errors["DATA"]) == expected


def test_pyspark_check_nullable(spark_session, sample_spark_schema, request):
    """
    Test creating a pyspark DataFrameSchema object to validate the nullability functionality
    """
    spark = request.getfixturevalue(spark_session)
    pandera_schema = DataFrameSchema(
        columns={
            "product": Column("str", checks=pa.Check.str_startswith("B")),
            "price": Column("int", nullable=False),
        }
    )

    data_fail = [("Bread", None), ("Cutter", 15)]
    sample_spark_schema = T.StructType(
        [
            T.StructField("product", T.StringType(), False),
            T.StructField("price", T.IntegerType(), True),
        ],
    )
    df_fail = spark_df(spark, data_fail, sample_spark_schema)
    dataframe_output = pandera_schema.validate(check_obj=df_fail)
    expected = {
        "SERIES_CONTAINS_NULLS": [
            {
                "check": "not_nullable",
                "column": "price",
                "error": "non-nullable "
                "column "
                "'price' "
                "contains "
                "null",
                "schema": None,
            }
        ]
    }
    assert dict(dataframe_output.pandera.errors["SCHEMA"]) == expected


def test_pyspark_schema_data_checks(spark_session, request):
    """
    Test schema and data level checks to check the Complex type data match
    """
    spark = request.getfixturevalue(spark_session)
    pandera_schema = DataFrameSchema(
        columns={
            "product": Column("str", checks=pa.Check.str_startswith("B")),
            "price": Column("int", checks=pa.Check.gt(5)),
            "id": Column(T.ArrayType(StringType())),
        },
        name="product_schema",
        description="schema for product info",
        title="ProductSchema",
    )

    data_fail = [("Bread", 5, ["Food"]), ("Cutter", 15, ["99"])]

    spark_schema = T.StructType(
        [
            T.StructField("product", T.StringType(), False),
            T.StructField("price", T.IntegerType(), False),
            T.StructField("id", T.ArrayType(StringType(), False), False),
        ],
    )

    df_fail = spark_df(spark, data_fail, spark_schema)
    output_data = pandera_schema.validate(check_obj=df_fail)
    expected = {
        "DATA": {
            "DATAFRAME_CHECK": [
                {
                    "check": "str_startswith('B')",
                    "column": "product",
                    "error": "column 'product' with "
                    f"type {str(T.StringType())} failed "
                    "validation "
                    "str_startswith('B')",
                    "schema": "product_schema",
                },
                {
                    "check": "greater_than(5)",
                    "column": "price",
                    "error": "column 'price' with type "
                    f"{str(T.IntegerType())} failed "
                    "validation "
                    "greater_than(5)",
                    "schema": "product_schema",
                },
            ]
        },
        "SCHEMA": {
            "WRONG_DATATYPE": [
                {
                    "check": f"dtype('{str(T.ArrayType(StringType(), True))}')",
                    "column": "id",
                    "error": "expected "
                    "column 'id' "
                    "to have type "
                    f"{str(T.ArrayType(StringType(), True))}, got "
                    f"{str(T.ArrayType(StringType(), False))}",
                    "schema": "product_schema",
                }
            ]
        },
    }

    assert dict(output_data.pandera.errors["DATA"]) == expected["DATA"]
    assert dict(output_data.pandera.errors["SCHEMA"]) == expected["SCHEMA"]


def test_pyspark_fields(spark_session, request):
    """
    Test schema and data level checks for pydantic validation
    """
    spark = request.getfixturevalue(spark_session)

    class PanderaSchema(DataFrameModel):
        """Test case schema class"""

        product: T.StringType = Field(str_startswith="B")
        price: T.IntegerType = Field(gt=5)
        id: T.DecimalType(20, 5) = Field()
        id2: T.ArrayType(StringType()) = Field()
        product_info: T.MapType(StringType(), StringType())

    data_fail = [
        ("Bread", 5, 44.4, ["val"], {"product_category": "dairy"}),
        ("Cutter", 15, 99.0, ["val2"], {"product_category": "bakery"}),
    ]

    spark_schema = T.StructType(
        [
            T.StructField("product", T.StringType(), False),
            T.StructField("price", T.IntegerType(), False),
            T.StructField("id", T.DecimalType(20, 5), False),
            T.StructField("id2", T.ArrayType(T.StringType()), False),
            T.StructField(
                "product_info",
                T.MapType(T.StringType(), T.StringType(), False),
                False,
            ),
        ],
    )
    df_fail = spark_df(spark, data_fail, spark_schema)
    df_out = PanderaSchema.validate(check_obj=df_fail)
    data_errors = dict(df_out.pandera.errors["DATA"])
    schema_errors = dict(df_out.pandera.errors["SCHEMA"])
    expected = {
        "DATA": {
            "DATAFRAME_CHECK": [
                {
                    "check": "str_startswith('B')",
                    "column": "product",
                    "error": "column "
                    "'product' "
                    "with type "
                    f"{str(T.StringType())} "
                    "failed "
                    "validation "
                    "str_startswith('B')",
                    "schema": "PanderaSchema",
                },
                {
                    "check": "greater_than(5)",
                    "column": "price",
                    "error": "column "
                    "'price' with "
                    "type "
                    f"{str(T.IntegerType())} "
                    "failed "
                    "validation "
                    "greater_than(5)",
                    "schema": "PanderaSchema",
                },
            ]
        },
        "SCHEMA": {
            "WRONG_DATATYPE": [
                {
                    "check": f"dtype('{str(T.MapType(StringType(), StringType(), True))}')",
                    "column": "product_info",
                    "error": "expected "
                    "column "
                    "'product_info' "
                    "to have type "
                    f"{str(T.MapType(T.StringType(), T.StringType(), True))}, got "
                    f"{str(T.MapType(T.StringType(), T.StringType(), False))}",
                    "schema": "PanderaSchema",
                }
            ]
        },
    }

    assert data_errors == expected["DATA"]
    assert schema_errors == expected["SCHEMA"]


def test_pyspark__error_handler_lazy_validation(
    spark_session,  # pylint:disable=unused-argument
):
    """This function tests the lazy validation for the error handler class of pyspark"""

    errors_not_lazy = error_handler.ErrorHandler(lazy=False)
    errors_lazy = error_handler.ErrorHandler(lazy=True)

    class BaseSchema(DataFrameModel):  # pylint:disable=abstract-method
        """test class"""

        id: int

    test_error = SchemaError(BaseSchema, [123], "Test")

    assert not errors_not_lazy.lazy
    with pytest.raises(SchemaError):
        errors_not_lazy.collect_error(
            error_handler.ErrorCategory.SCHEMA,
            SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
            test_error,
        )
    assert errors_lazy.lazy

    test_error.schema.name = "Test"
    errors_lazy.collect_error(
        error_handler.ErrorCategory.SCHEMA,
        SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
        test_error,
    )

    assert [
        {
            "type": error_handler.ErrorCategory.SCHEMA,
            "column": "Test",
            "check": None,
            "failure_cases_count": 0,
            "reason_code": SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
            "error": test_error,
        }
    ] == errors_lazy._collected_errors
