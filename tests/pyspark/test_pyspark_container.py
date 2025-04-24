"""Unit tests for pyspark container."""

import platform
from contextlib import nullcontext as does_not_raise
from datetime import date, datetime
from decimal import Decimal

import pyspark.sql.types as T
import pytest
from pyspark.sql import DataFrame, Row

import pandera.errors
import pandera.pyspark as pa
from pandera.config import PanderaConfig, ValidationDepth
from pandera.pyspark import Column, DataFrameModel, DataFrameSchema

pytestmark = pytest.mark.parametrize(
    "spark_session", ["spark", "spark_connect"]
)


def test_pyspark_dataframeschema(spark_session, request):
    """
    Test creating a pyspark DataFrameSchema object
    """
    spark = request.getfixturevalue(spark_session)
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
    spark_session,
    request,
):
    """
    Test creating a pyspark DataFrameSchema object
    """
    spark = request.getfixturevalue(spark_session)
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


def test_pyspark_column_metadata(
    spark_session,  # pylint:disable=unused-argument
):
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


def test_pyspark_sample(spark_session, request):
    """
    Test the sample functionality of pyspark
    """
    spark = request.getfixturevalue(spark_session)
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
        ("Chocolate", 7),
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


def test_pyspark_regex_column(spark_session, request):
    """
    Test creating a pyspark DataFrameSchema object with regex columns
    """
    spark = request.getfixturevalue(spark_session)
    schema = DataFrameSchema(
        {
            # Columns with all caps names must have string values
            "[A-Z]+": Column(T.StringType(), regex=True),
        }
    )

    data = [("Neeraj", 35), ("Jask", 30)]

    df = spark.createDataFrame(data=data, schema=["NAME", "AGE"])
    df_out = schema.validate(df)

    assert df_out.pandera.errors is not None

    data = [("Neeraj", "35"), ("Jask", "a")]

    df2 = spark.createDataFrame(data=data, schema=["NAME", "AGE"])

    df_out = schema.validate(df2)

    assert not df_out.pandera.errors


def test_pyspark_nullable(spark_session, request):
    """
    Test the nullable functionality of pyspark
    """
    spark = request.getfixturevalue(spark_session)
    data = [
        ("Bread", 9),
        ("Butter", 15),
        ("Ice Cream", None),
        ("Cola", 12),
        ("Chocolate", None),
    ]
    spark_schema = T.StructType(
        [
            T.StructField("product", T.StringType(), False),
            T.StructField("price", T.IntegerType(), True),
        ],
    )
    df = spark.createDataFrame(data=data, schema=spark_schema)

    # Check for `nullable=False`
    schema_nullable_false = DataFrameSchema(
        columns={
            "product": Column("str"),
            "price": Column("int", nullable=False),
        },
    )
    with does_not_raise():
        df_out = schema_nullable_false.validate(df)
    assert isinstance(df_out, DataFrame)
    assert "SERIES_CONTAINS_NULLS" in str(dict(df_out.pandera.errors))

    # Check for `nullable=True`
    schema_nullable_true = DataFrameSchema(
        columns={
            "product": Column("str"),
            "price": Column("int", nullable=True),
        },
    )
    with does_not_raise():
        df_out = schema_nullable_true.validate(df)
    assert isinstance(df_out, DataFrame)
    assert df_out.pandera.errors == {}


def test_pyspark_unique_field(spark_session, request):
    """
    Test that field unique True raise an error.
    """
    spark = request.getfixturevalue(spark_session)
    with pytest.raises(pandera.errors.SchemaInitError):
        # pylint: disable=W0223
        class PanderaSchema(DataFrameModel):
            id: T.StringType() = pa.Field(unique=True)

        data = [
            ("id1"),
        ]
        spark_schema = T.StructType(
            [
                T.StructField("id", T.StringType(), False),
            ],
        )
        df = spark.createDataFrame(data=data, schema=spark_schema)
        df_out = PanderaSchema.validate(df)
        assert len(df_out.pandera.errors) == 0


def test_pyspark_unique_config(spark_session, request):
    """
    Test the sample functionality of pyspark
    """
    spark = request.getfixturevalue(spark_session)

    # pylint: disable=W0223
    class PanderaSchema(DataFrameModel):
        product: T.StringType() = pa.Field()
        price: T.IntegerType() = pa.Field()

        class Config:
            unique = "product"

    data = [
        ("Bread", 9),
        ("Butter", 15),
        ("Ice Cream", 10),
        ("Cola", 12),
        ("Chocolate", 7),
        ("Chocolate", 7),
    ]

    spark_schema = T.StructType(
        [
            T.StructField("product", T.StringType(), False),
            T.StructField("price", T.IntegerType(), False),
        ],
    )

    df = spark.createDataFrame(data=data, schema=spark_schema)

    df_out = PanderaSchema.validate(df)

    assert len(df_out.pandera.errors["DATA"]["DUPLICATES"]) == 1


@pytest.fixture(scope="module")
def schema_with_complex_datatypes():
    """
    Model containing all common datatypes for PySpark namespace, supported by parquet.
    """
    schema = DataFrameSchema(
        {
            "non_nullable": Column(T.IntegerType(), nullable=False),
            "binary": Column(T.BinaryType()),
            "byte": Column(T.ByteType()),
            "text": Column(T.StringType()),
            "integer": Column(T.IntegerType()),
            "long": Column(T.LongType()),
            "float": Column(T.FloatType()),
            "double": Column(T.DoubleType()),
            "boolean": Column(T.BooleanType()),
            "decimal": Column(T.DecimalType()),
            "date": Column(T.DateType()),
            "timestamp": Column(T.TimestampType()),
            "timestamp_ntz": Column(T.TimestampNTZType()),
            "array": Column(T.ArrayType(T.StringType())),
            "map": Column(T.MapType(T.StringType(), T.IntegerType())),
            "nested_structure": Column(
                T.MapType(
                    T.ArrayType(T.StringType()),
                    T.MapType(T.StringType(), T.ArrayType(T.StringType())),
                )
            ),
        }
    )

    return schema


def test_schema_to_structtype(
    schema_with_complex_datatypes,
    spark_session,  # pylint:disable=unused-argument
):
    """
    Test the conversion from a schema to a StructType object through `to_structtype()`.
    """

    assert schema_with_complex_datatypes.to_structtype() == T.StructType(
        [
            T.StructField(
                name="non_nullable", dataType=T.IntegerType(), nullable=True
            ),
            T.StructField(
                name="binary", dataType=T.BinaryType(), nullable=True
            ),
            T.StructField(name="byte", dataType=T.ByteType(), nullable=True),
            T.StructField(name="text", dataType=T.StringType(), nullable=True),
            T.StructField(
                name="integer", dataType=T.IntegerType(), nullable=True
            ),
            T.StructField(name="long", dataType=T.LongType(), nullable=True),
            T.StructField(name="float", dataType=T.FloatType(), nullable=True),
            T.StructField(
                name="double", dataType=T.DoubleType(), nullable=True
            ),
            T.StructField(
                name="boolean", dataType=T.BooleanType(), nullable=True
            ),
            T.StructField(
                name="decimal", dataType=T.DecimalType(), nullable=True
            ),
            T.StructField(name="date", dataType=T.DateType(), nullable=True),
            T.StructField(
                name="timestamp", dataType=T.TimestampType(), nullable=True
            ),
            T.StructField(
                name="timestamp_ntz", dataType=T.TimestampType(), nullable=True
            ),
            T.StructField(
                name="array",
                dataType=T.ArrayType(T.StringType()),
                nullable=True,
            ),
            T.StructField(
                name="map",
                dataType=T.MapType(T.StringType(), T.IntegerType()),
                nullable=True,
            ),
            T.StructField(
                name="nested_structure",
                dataType=T.MapType(
                    T.ArrayType(T.StringType()),
                    T.MapType(T.StringType(), T.ArrayType(T.StringType())),
                ),
                nullable=True,
            ),
        ]
    )
    assert schema_with_complex_datatypes.to_structtype() != T.StructType(
        [
            T.StructField(
                name="non_nullable", dataType=T.IntegerType(), nullable=True
            ),
            T.StructField(
                name="binary", dataType=T.StringType(), nullable=True  # Wrong
            ),
            T.StructField(
                name="byte", dataType=T.StringType(), nullable=True
            ),  # Wrong
            T.StructField(name="text", dataType=T.StringType(), nullable=True),
            T.StructField(
                name="integer", dataType=T.IntegerType(), nullable=True
            ),
            T.StructField(name="long", dataType=T.LongType(), nullable=True),
            T.StructField(name="float", dataType=T.FloatType(), nullable=True),
            T.StructField(
                name="double", dataType=T.DoubleType(), nullable=True
            ),
            T.StructField(
                name="boolean", dataType=T.BooleanType(), nullable=True
            ),
            T.StructField(
                name="decimal", dataType=T.DecimalType(), nullable=True
            ),
            T.StructField(name="date", dataType=T.DateType(), nullable=True),
            T.StructField(
                name="timestamp", dataType=T.TimestampType(), nullable=True
            ),
            T.StructField(
                name="timestamp_ntz", dataType=T.TimestampType(), nullable=True
            ),
            T.StructField(
                name="array",
                dataType=T.ArrayType(T.StringType()),
                nullable=True,
            ),
            T.StructField(
                name="map",
                dataType=T.MapType(T.StringType(), T.IntegerType()),
                nullable=True,
            ),
            T.StructField(
                name="nested_structure",
                dataType=T.MapType(
                    T.ArrayType(T.StringType()),
                    T.MapType(T.StringType(), T.ArrayType(T.StringType())),
                ),
                nullable=True,
            ),
        ]
    )


def test_schema_to_ddl(
    schema_with_complex_datatypes,
    spark_session,  # pylint:disable=unused-argument
):
    """
    Test the conversion from a schema to a DDL string through `to_ddl()`.
    """

    assert schema_with_complex_datatypes.to_ddl() == ",".join(
        [
            "non_nullable INT",
            "binary BINARY",
            "byte TINYINT",
            "text STRING",
            "integer INT",
            "long BIGINT",
            "float FLOAT",
            "double DOUBLE",
            "boolean BOOLEAN",
            "decimal DECIMAL(10,0)",
            "date DATE",
            "timestamp TIMESTAMP",
            "timestamp_ntz TIMESTAMP",
            "array ARRAY<STRING>",
            "map MAP<STRING, INT>",
            "nested_structure MAP<ARRAY<STRING>, MAP<STRING, ARRAY<STRING>>>",
        ]
    )
    assert schema_with_complex_datatypes.to_ddl() != ",".join(
        [
            "non_nullable INT",
            "binary STRING",  # Wrong
            "byte STRING",  # Wrong
            "text STRING",
            "integer INT",
            "long BIGINT",
            "float FLOAT",
            "double DOUBLE",
            "boolean BOOLEAN",
            "decimal DECIMAL(10,0)",
            "date DATE",
            "timestamp TIMESTAMP",
            "timestamp_ntz TIMESTAMP",
            "array ARRAY<STRING>",
            "map MAP<STRING, INT>",
            "nested_structure MAP<ARRAY<STRING>, MAP<STRING, ARRAY<STRING>>>",
        ]
    )


@pytest.fixture(scope="function")
def schema_with_simple_datatypes(
    spark_session,  # pylint:disable=unused-argument
):
    """
    Model containing all common datatypes for PySpark namespace, supported by CSV.
    """
    schema = DataFrameSchema(
        {
            "non_nullable": Column(T.IntegerType(), nullable=False),
            "byte": Column(T.ByteType()),
            "text": Column(T.StringType()),
            "integer": Column(T.IntegerType()),
            "long": Column(T.LongType()),
            "float": Column(T.FloatType()),
            "double": Column(T.DoubleType()),
            "boolean": Column(T.BooleanType()),
            "decimal": Column(T.DecimalType()),
            "date": Column(T.DateType()),
            "timestamp": Column(T.TimestampType()),
            "timestamp_ntz": Column(T.TimestampNTZType()),
        }
    )

    return schema


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="skipping due to issues with opening file names for temp files.",
)
def test_pyspark_read(
    schema_with_simple_datatypes, tmp_path, spark_session, request
):
    """
    Test reading a file using an automatically generated schema object.
    """
    spark = request.getfixturevalue(spark_session)
    original_pyspark_schema = T.StructType(
        [
            T.StructField(
                name="non_nullable", dataType=T.IntegerType(), nullable=True
            ),
            T.StructField(name="byte", dataType=T.ByteType(), nullable=True),
            T.StructField(name="text", dataType=T.StringType(), nullable=True),
            T.StructField(
                name="integer", dataType=T.IntegerType(), nullable=True
            ),
            T.StructField(name="long", dataType=T.LongType(), nullable=True),
            T.StructField(name="float", dataType=T.FloatType(), nullable=True),
            T.StructField(
                name="double", dataType=T.DoubleType(), nullable=True
            ),
            T.StructField(
                name="boolean", dataType=T.BooleanType(), nullable=True
            ),
            T.StructField(
                name="decimal", dataType=T.DecimalType(), nullable=True
            ),
            T.StructField(name="date", dataType=T.DateType(), nullable=True),
            T.StructField(
                name="timestamp", dataType=T.TimestampType(), nullable=True
            ),
            T.StructField(
                name="timestamp_ntz", dataType=T.TimestampType(), nullable=True
            ),
        ]
    )
    sample_data = [
        Row(
            1,
            2,
            "3",
            4,
            5,
            6.0,
            7.0,
            True,
            Decimal(8),
            date(2000, 1, 1),
            datetime(2000, 1, 1, 1, 1, 1),
            datetime(2000, 1, 1, 1, 1, 1),
        )
    ]

    # Writes a csv file to disk
    empty_df = spark.createDataFrame(
        sample_data, schema=original_pyspark_schema
    )
    empty_df.show()
    empty_df.write.csv(f"{tmp_path}/test.csv", header=True)

    # Read the file using automatic schema inference, getting a schema different
    # from the expected
    read_df = (
        spark.read.format("csv")
        .option("inferSchema", True)
        .option("header", True)
        .load(f"{tmp_path}/test.csv")
    )
    # The loaded DF schema shouldn't match the original schema
    print(f"Read CSV schema:\n{read_df.schema}")
    print(f"Expected schema:\n{original_pyspark_schema}")
    assert read_df.schema != original_pyspark_schema, "Schemas shouldn't match"

    # Read again the file without `inferSchema`, by setting our expected schema
    # through the usage of `.to_structtype()`
    read_df = spark.read.format("csv").load(
        f"{tmp_path}/test.csv",
        schema=schema_with_simple_datatypes.to_structtype(),
    )
    # The loaded DF should now match the original expected datatypes
    assert read_df.schema == original_pyspark_schema, "Schemas should match"

    # Read again the file without `inferSchema`, by setting our expected schema
    # through the usage of `.to_ddl()`
    read_df = spark.read.format("csv").load(
        f"{tmp_path}/test.csv", schema=schema_with_simple_datatypes.to_ddl()
    )
    # The loaded DF should now match the original expected datatypes
    assert read_df.schema == original_pyspark_schema, "Schemas should match"
