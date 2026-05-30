"""conftest"""

import datetime
import os

import pyspark
import pyspark.sql.types as T
import pytest
from packaging import version
from pyspark.sql import SparkSession

from pandera.api.base.error_handler import ErrorHandler
from pandera.config import CONFIG, PanderaConfig
from pandera.errors import SchemaErrors

PYSPARK_VERSION = version.parse(pyspark.__version__)


@pytest.fixture(autouse=True)
def spark_env_vars():
    """Sets environment variables for pyspark."""
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """
    creates spark session
    """
    builder = SparkSession.builder
    builder = builder.config("spark.sql.ansi.enabled", False)
    # Workaround for Java 17+ security manager issues with Hadoop file system
    # This is needed for PySpark 4.0+ when using Java 17+
    if PYSPARK_VERSION >= version.parse("4.0.0"):
        builder = builder.config("spark.hadoop.fs.defaultFS", "file:///")
        builder = builder.config(
            "spark.sql.warehouse.dir", "file:///tmp/spark-warehouse"
        )
    spark: SparkSession = builder.getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def spark_connect() -> SparkSession:
    """
    creates spark connection session
    """
    # Set location of localhost Spark Connect server
    os.environ["SPARK_LOCAL_REMOTE"] = "sc://localhost"
    builder = SparkSession.builder
    builder = builder.config("spark.sql.ansi.enabled", False)
    # Workaround for Java 17+ security manager issues with Hadoop file system
    # This is needed for PySpark 4.0+ when using Java 17+
    if PYSPARK_VERSION >= version.parse("4.0.0"):
        builder = builder.config("spark.hadoop.fs.defaultFS", "file:///")
        builder = builder.config(
            "spark.sql.warehouse.dir", "file:///tmp/spark-warehouse"
        )
    spark: SparkSession = builder.getOrCreate()
    yield spark
    spark.stop()


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
    """This function creates spark dataframe from given data and schema object"""
    return spark.createDataFrame(
        data=data, schema=spark_schema, verifySchema=False
    )


@pytest.fixture(scope="session")
def sample_date_object(spark):
    """This fundtion contains sample data for datetime types object"""
    sample_data = [
        (
            datetime.date(2022, 10, 1),
            datetime.datetime(2022, 10, 1, 5, 32, 0),
        ),
        (
            datetime.date(2022, 11, 5),
            datetime.datetime(2022, 11, 5, 15, 34, 0),
        ),
    ]
    sample_spark_schema = T.StructType(
        [
            T.StructField("purchase_date", T.DateType(), False),
            T.StructField("purchase_datetime", T.TimestampType(), False),
        ],
    )
    df = spark_df(
        spark=spark, spark_schema=sample_spark_schema, data=sample_data
    )
    return df


@pytest.fixture(scope="session")
def sample_string_binary_object(spark):
    """This function creates the sample data for binary types"""
    sample_data = [
        (
            "test1",
            "Bread",
        ),
        ("test2", "Butter"),
    ]
    sample_spark_schema = T.StructType(
        [
            T.StructField("purchase_info", T.StringType(), False),
            T.StructField("product", T.StringType(), False),
        ],
    )
    df = spark_df(
        spark=spark, spark_schema=sample_spark_schema, data=sample_data
    )
    df = df.withColumn(
        "purchase_info", df["purchase_info"].cast(T.BinaryType())
    )
    return df


@pytest.fixture(scope="session")
def sample_complex_data(spark):
    """This function creates sample data for complex datatypes types"""
    sample_data = [
        (
            datetime.date(2022, 10, 1),
            [["josh"], ["27"]],
            {"product_bought": "bread"},
        ),
        (
            datetime.date(2022, 11, 5),
            [["Adam"], ["22"]],
            {"product_bought": "bread"},
        ),
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
                T.MapType(T.StringType(), T.StringType()),
                False,
            ),
        ],
    )
    return spark_df(spark, sample_data, sample_spark_schema)


@pytest.fixture(scope="session")
def sample_check_data():
    """This creates data for check type"""
    return {
        "test_pass_data": [("foo", 30), ("bar", 30)],
        "test_fail_data": [("foo", 30), ("bar", 31)],
        "test_expression": 30,
    }


@pytest.fixture(scope="session")
def config_params():
    """This function creates config parameters"""
    return PanderaConfig()


def validate_collecting_errors(schema, df, **validate_kwargs):
    """Backend-aware validation helper that returns ``(out_df, errors_dict)``.

    Abstracts the difference between:
    - Native PySpark backend: validation attaches errors to ``df.pandera.errors``
      and returns the DataFrame.
    - Narwhals backend (``CONFIG.use_narwhals_backend=True``): validation raises
      ``pandera.errors.SchemaErrors`` on failure.

    :param schema: A ``DataFrameSchema`` or ``DataFrameModel`` with a
        ``.validate()`` method.
    :param df: The PySpark DataFrame to validate.
    :param validate_kwargs: Additional keyword arguments forwarded to
        ``schema.validate()``.
    :returns: A ``(out_df, errors_dict)`` tuple where ``errors_dict`` is a
        ``dict`` with ``"SCHEMA"`` and/or ``"DATA"`` keys (same format as
        ``df.pandera.errors`` under the native backend).  On success the dict
        is empty (``{}``).  On narwhals-backend failure ``out_df`` is ``None``;
        on native-backend failure ``out_df`` is the annotated DataFrame.
    """
    try:
        out_df = schema.validate(df, **validate_kwargs)
        errors = out_df.pandera.errors
        return (out_df, dict(errors) if errors is not None else {})
    except SchemaErrors as exc:
        # Narwhals path: rebuild the same nested dict structure from the exception.
        handler = ErrorHandler(lazy=True)
        handler.collect_errors(exc.schema_errors)
        schema_name = getattr(schema, "name", None)
        errors = handler.summarize(schema_name=schema_name)
        return (None, dict(errors))


def _cmp_errors(actual, expected):
    """Compare pandera error dicts ignoring the exact error message text.

    Error message format varies by backend (narwhals vs native PySpark),
    so only structural fields (check, column, schema) are compared.
    """

    def drop_error(entries):
        return [{k: v for k, v in e.items() if k != "error"} for e in entries]

    assert set(actual) == set(expected)
    for key in expected:
        assert drop_error(actual[key]) == drop_error(expected[key])
        assert all(e["error"] for e in actual[key])
