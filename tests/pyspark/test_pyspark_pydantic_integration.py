"""Tests for integration of Pyspark DataFrames with Pydantic."""
# pylint:disable=redefined-outer-name,abstract-method

from enum import Enum

import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError
from pyspark.testing.utils import assertDataFrameEqual
import pyspark.sql.types as T

from pandera.typing.pyspark_sql import DataFrame as PySparkSQLDataFrame
from pandera.typing.pyspark import DataFrame as PysparkPandasDataFrame, Series
from pandera.pyspark import DataFrameModel as PysparkSQLDataFrameModel
from pandera import DataFrameModel


class PysparkAPIs(Enum):
    """
    Enum for the different Pyspark APIs.
    """

    PANDAS = "pandas"
    SQL = "SQL"


@pytest.fixture(
    params=[PysparkAPIs.PANDAS, PysparkAPIs.SQL],
    ids=["pyspark_pandas", "pyspark_sql"],
)
def pyspark_api(request):
    """
    Fixture for the different Pyspark APIs.
    """
    return request.param


@pytest.fixture
def sample_data_frame_model_class(pyspark_api):
    """
    Fixture for a sample DataFrameModel class.
    """
    if pyspark_api == PysparkAPIs.SQL:

        class SampleSchema(PysparkSQLDataFrameModel):
            """
            Sample schema model
            """

            product: T.StringType()
            price: T.IntegerType()

    elif pyspark_api == PysparkAPIs.PANDAS:

        class SampleSchema(DataFrameModel):
            """
            Sample schema model
            """

            product: Series[str]
            price: Series[pd.Int32Dtype]

    else:
        raise ValueError(f"Unknown data frame library: {pyspark_api}")

    return SampleSchema


@pytest.fixture
def pydantic_container(pyspark_api, sample_data_frame_model_class):
    """
    Fixture for a Pydantic container with a DataFrameModel as a field.
    """
    if pyspark_api == PysparkAPIs.PANDAS:

        class PydanticContainer(BaseModel):
            """
            Pydantic container with a DataFrameModel as a field.
            """

            data: PysparkPandasDataFrame[sample_data_frame_model_class]

    elif pyspark_api == PysparkAPIs.SQL:

        class PydanticContainer(BaseModel):
            """
            Pydantic container with a DataFrameModel as a field.
            """

            data: PySparkSQLDataFrame[sample_data_frame_model_class]

    else:
        raise ValueError(f"Unknown data frame library: {pyspark_api}")

    return PydanticContainer


@pytest.fixture
def correct_data(spark, sample_data, sample_spark_schema, pyspark_api):
    """
    Correct data that should pass validation.
    """
    df = spark.createDataFrame(sample_data, sample_spark_schema)
    if pyspark_api == PysparkAPIs.PANDAS:
        return df.pandas_api()
    elif pyspark_api == PysparkAPIs.SQL:
        return df
    else:
        raise ValueError(f"Unknown data frame library: {pyspark_api}")


@pytest.fixture
def incorrect_data(spark, pyspark_api):
    """
    Incorrect data that should fail validation.
    """
    data = [
        (1, "Apples"),
        (2, "Bananas"),
    ]
    df = spark.createDataFrame(data, ["product", "price"])
    if pyspark_api == PysparkAPIs.PANDAS:
        return df.pandas_api()
    elif pyspark_api == PysparkAPIs.SQL:
        return df
    else:
        raise ValueError(f"Unknown data frame library: {pyspark_api}")


def test_pydantic_model_instantiates_with_correct_data(
    correct_data, pydantic_container
):
    """
    Test that a Pydantic model can be instantiated with a DataFrameModel when data is valid.
    """
    my_container = pydantic_container(data=correct_data)
    assertDataFrameEqual(my_container.data, correct_data)


def test_pydantic_model_throws_validation_error_with_incorrect_data(
    incorrect_data, pydantic_container, pyspark_api
):
    """
    Test that a Pydantic model throws a ValidationError when data is invalid.
    """
    if pyspark_api == PysparkAPIs.PANDAS:
        expected_error_substring = "expected series 'product' to have type str"
    elif pyspark_api == PysparkAPIs.SQL:
        expected_error_substring = (
            "expected column 'product' to have type StringType()"
        )
    else:
        raise ValueError(f"Unknown data frame library: {pyspark_api}")

    with pytest.raises(ValidationError, match=expected_error_substring):
        pydantic_container(data=incorrect_data)
