"""Unit tests for DataFrameModel module."""
from pyspark.sql.functions import col
import pyspark.sql.types as T
import pytest
import pandera as pa
from pandera import SchemaModel
from pandera.error_handlers import SchemaError
from pandera.api.pyspark.model import DataFrameModel
from pandera.api.pyspark.model_components import Field
from tests.pyspark.conftest import spark_df


def test_pyspark_fields(spark):
    """
    Test schema and data level checks
    """

    class pandera_schema(DataFrameModel):
        product: pa.typing.Column[str] = Field(str_startswith="B")
        price: pa.typing.Column[int] = Field(gt=6)
        id: pa.typing.Column[int] = Field()

    data_fail = [("Bread", 5, "Food"), ("Cutter", 15, 99)]

    spark_schema = T.StructType(
        [
            T.StructField("product", T.StringType(), False),  # should fail
            T.StructField("price", T.IntegerType(), False),  # should fail
            T.StructField("id", T.StringType(), False),  # should fail
        ],
    )

    df_fail = spark_df(spark, data_fail, spark_schema)
    errors = pandera_schema.report_errors(check_obj=df_fail)

    if errors:
        raise SchemaError(
            pandera_schema,
            df_fail,
            f"errors: {errors}",
        )
