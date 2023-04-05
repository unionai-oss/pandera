""" conftest """
import pytest
from pyspark.sql import SparkSession
import pyspark.sql.types as T


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """
    creates spark session
    """
    return SparkSession.builder.getOrCreate()


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
            T.StructField("product", T.StringType(), False),
            T.StructField("price", T.IntegerType(), False),
        ],
    )
