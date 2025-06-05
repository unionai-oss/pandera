"""Tests for the integration between PySpark and Pydantic."""

import pytest
from pydantic import BaseModel, ValidationError
from pyspark.testing.utils import assertDataFrameEqual
import pyspark.sql.types as T

import pandera.pyspark as pa
from pandera.typing.pyspark_sql import DataFrame as PySparkSQLDataFrame
from pandera.typing.pyspark import DataFrame as PySparkDataFrame
from pandera.pyspark import DataFrameModel


@pytest.fixture
def sample_schema_model():
    class SampleSchema(DataFrameModel):
        """
        Sample schema model with data checks.
        """

        product: T.StringType() = pa.Field()
        price: T.IntegerType() = pa.Field()

    return SampleSchema


@pytest.fixture(
    params=[PySparkDataFrame, PySparkSQLDataFrame],
    ids=["pyspark", "pyspark_sql"],
)
def pydantic_container(request, sample_schema_model):
    TypingClass = request.param

    class PydanticContainer(BaseModel):
        """
        Pydantic container with a DataFrameModel as a field.
        """

        data: TypingClass[sample_schema_model]

    return PydanticContainer


@pytest.fixture
def correct_data(spark, sample_data, sample_spark_schema):
    """
    Correct data that should pass validation.
    """
    return spark.createDataFrame(sample_data, sample_spark_schema)


@pytest.fixture
def incorrect_data(spark):
    """
    Incorrect data that should fail validation.
    """
    data = [
        (1, "Apples"),
        (2, "Bananas"),
    ]
    return spark.createDataFrame(data, ["product", "price"])


def test_pydantic_model_instantiates_with_correct_data(
    correct_data, pydantic_container
):
    """
    Test that a Pydantic model can be instantiated with a DataFrameModel when data is valid.
    """
    my_container = pydantic_container(data=correct_data)
    assertDataFrameEqual(my_container.data, correct_data)


def test_pydantic_model_throws_validation_error_with_incorrect_data(
    incorrect_data, pydantic_container
):
    """
    Test that a Pydantic model throws a ValidationError when data is invalid.
    """
    with pytest.raises(ValidationError):
        pydantic_container(data=incorrect_data)
