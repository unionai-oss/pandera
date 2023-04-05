"""Unit tests for pyspark container."""
import pyspark.sql.types as T
import pytest
from pandera.api.pyspark.container import DataFrameSchema
from pandera.api.pyspark.components import Column
from pandera.error_handlers import SchemaError
from tests.pyspark.conftest import spark_df


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

    df = spark_df(spark, sample_data, sample_spark_schema)

    pandera_schema.validate(df)

    # negative test
    with pytest.raises(SchemaError):
        spark_schema_fail = T.StructType(
            [
                T.StructField("product", T.StringType(), False),
                T.StructField("price", T.FloatType(), False),
            ],
        )
        data_fail = [("Bread", 9.0), ("Butter", 15)]

        df_fail = spark_df(spark, data_fail, spark_schema_fail)

        pandera_schema.validate(df_fail)
