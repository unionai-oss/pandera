"""This module is to test the behaviour change based on defined config in pandera"""
# pylint:disable=import-outside-toplevel,abstract-method

from contextlib import nullcontext as does_not_raise
import logging
import pyspark.sql.types as T
from pyspark.sql import DataFrame
import pytest

from pandera.backends.pyspark.decorators import cache_check_obj
from pandera.config import CONFIG
from pandera.pyspark import (
    Check,
    DataFrameSchema,
    Column,
)
from tests.pyspark.conftest import spark_df


class TestPanderaDecorators:
    """Class to test all the different configs types"""

    sample_data = [("Bread", 9), ("Cutter", 15)]

    def test_pyspark_cache_requirements(self, spark, sample_spark_schema):
        """Validates if decorator can only be applied in a proper function."""
        # Set expected properties in Config object
        CONFIG.pyspark_cache = True
        input_df = spark_df(spark, self.sample_data, sample_spark_schema)

        class FakeDataFrameSchemaBackend:
            """Class that simulates DataFrameSchemaBackend class."""

            @cache_check_obj()
            def func_w_check_obj(self, check_obj: DataFrame):
                """Right function to use this decorator."""
                return check_obj.columns

            @cache_check_obj()
            def func_wo_check_obj(self, message: str):
                """Wrong function to use this decorator."""
                return message

        # Check for a function that does have a `check_obj`
        with does_not_raise():
            instance = FakeDataFrameSchemaBackend()
            _ = instance.func_w_check_obj(check_obj=input_df)

        # Check for a wrong function, that does not have a `check_obj`
        with pytest.raises(KeyError):
            instance = FakeDataFrameSchemaBackend()
            _ = instance.func_wo_check_obj("wrong")

    @pytest.mark.parametrize(
        "cache_enabled,unpersist_enabled,"
        "expected_caching_message,expected_unpersisting_message",
        [
            (True, True, True, True),
            (True, False, True, None),
            (False, True, None, None),
            (False, False, None, None),
        ],
        scope="function",
    )

    # pylint:disable=too-many-locals
    def test_pyspark_cache_settings(
        self,
        spark,
        sample_spark_schema,
        cache_enabled,
        unpersist_enabled,
        expected_caching_message,
        expected_unpersisting_message,
        caplog,
    ):
        """This function validates that caching/unpersisting works as expected."""
        # Set expected properties in Config object
        CONFIG.pyspark_cache = cache_enabled
        CONFIG.pyspark_unpersist = unpersist_enabled

        # Prepare test data
        input_df = spark_df(spark, self.sample_data, sample_spark_schema)
        pandera_schema = DataFrameSchema(
            {
                "product": Column(T.StringType(), Check.str_startswith("B")),
                "price_val": Column(T.IntegerType()),
            }
        )

        # Capture log message
        with caplog.at_level(logging.DEBUG, logger="pandera"):
            df_out = pandera_schema.validate(input_df)

        # Assertions
        assert isinstance(df_out, DataFrame)
        if expected_caching_message:
            assert (
                "Caching dataframe..." in caplog.text
            ), "Debugging info has no information about caching the dataframe."
        else:
            assert (
                "Caching dataframe..." not in caplog.text
            ), "Debugging info has information about caching. It shouldn't."

        if expected_unpersisting_message:
            assert (
                "Unpersisting dataframe..." in caplog.text
            ), "Debugging info has no information about unpersisting the dataframe."
        else:
            assert (
                "Unpersisting dataframe..." not in caplog.text
            ), "Debugging info has information about unpersisting. It shouldn't."
