"""Unit tests for dask_accessor module."""
from typing import Union

from pyspark.sql import DataFrame
from pyspark.sql.functions import col
import pyspark.sql.types as T
import pytest
from pyspark.sql import SparkSession
import pandera as pa
from pandera import pyspark_sql_accessor, SchemaModel
from pandera.error_handlers import SchemaError

from pyspark.sql.types import StringType, LongType
from pandera.api.pyspark.container import DataFrameSchema
from pandera.api.pyspark.components import Column
from pandera.api.pyspark.model import DataFrameModel
from pandera.api.pyspark.model_components import Field
from tests.pyspark.conftest import spark_df

spark = SparkSession.builder.getOrCreate()


@pytest.mark.parametrize(
    "schema, invalid_data",
    [
        [
            pa.DataFrameSchema(
                {
                    "product": pa.Column(StringType()),
                    "code": pa.Column(LongType(), pa.Check.not_equal_to(30)),
                }
            ),
            spark.createDataFrame(
                data=[("23", 31), ("34", 35)], schema=["product", "code"]
            ),
        ],
    ],
)
def test_dataframe_add_schema(
    schema: pa.DataFrameSchema,
    invalid_data: Union[DataFrame, col],
) -> None:
    """
    Test that pandas object contains schema metadata after pandera validation.
    """
    # validated_data_1 = schema(data)  # type: ignore[arg-type]
    # print(schema2.report_errors(invalid_data))
    # print(schema1.report_errors(invalid_data))

    # with pytest.raises(SchemaError):
    schema(invalid_data, lazy=True)  # type: ignore[arg-type]


def test_pyspark_check_eq(spark, sample_spark_schema):
    """
    Test creating a pyspark DataFrameSchema object
    """

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
    errors = pandera_schema.report_errors(check_obj=df_fail)
    print(errors)


def test_pyspark_schema_data_checks(spark):
    """
    Test schema and data level checks
    """

    pandera_schema = DataFrameSchema(
        columns={
            "product": Column("str", checks=pa.Check.str_startswith("B")),
            "price": Column("int", checks=pa.Check.gt(5)),
            "id": Column("int"),
        },
        name="product_schema",
        description="schema for product info",
        title="ProductSchema",
    )

    data_fail = [("Bread", 5, "Food"), ("Cutter", 15, 99)]

    spark_schema = T.StructType(
        [
            T.StructField("product", T.StringType(), False),
            T.StructField("price", T.IntegerType(), False),
            T.StructField("id", T.StringType(), False),
        ],
    )

    df_fail = spark_df(spark, data_fail, spark_schema)
    errors = pandera_schema.report_errors(check_obj=df_fail)
    print(errors)


def test_pyspark_fields(spark):
    """
    Test schema and data level checks
    """

    class pandera_schema(DataFrameModel):
        product: T.StringType() = Field(str_startswith="B")
        price: T.IntegerType() = Field(gt=5)
        id: T.IntegerType() = Field()

    data_fail = [("Bread", 5, "Food"), ("Cutter", 15, 99)]

    spark_schema = T.StructType(
        [
            T.StructField("product", T.StringType(), False),
            T.StructField("price", T.IntegerType(), False),
            T.StructField("id", T.StringType(), False),
        ],
    )

    df_fail = spark_df(spark, data_fail, spark_schema)
    errors = pandera_schema.report_errors(check_obj=df_fail)
    print(errors)
