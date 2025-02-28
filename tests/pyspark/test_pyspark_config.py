"""This module is to test the behaviour change based on defined config in pandera"""

# pylint:disable=import-outside-toplevel,abstract-method

from dataclasses import asdict

import pyspark.sql.types as T
import pytest

from pandera.config import ValidationDepth, config_context, get_config_context
from pandera.pyspark import (
    Check,
    Column,
    DataFrameModel,
    DataFrameSchema,
    Field,
)
from tests.pyspark.conftest import spark_df

pytestmark = pytest.mark.parametrize(
    "spark_session", ["spark", "spark_connect"]
)


class TestPanderaConfig:
    """Class to test all the different configs types"""

    sample_data = [("Bread", 9), ("Cutter", 15)]

    def test_disable_validation(
        self, spark_session, sample_spark_schema, request
    ):
        """This function validates that a none object is loaded if validation is disabled"""
        spark = request.getfixturevalue(spark_session)
        pandera_schema = DataFrameSchema(
            {
                "product": Column(T.StringType(), Check.str_startswith("B")),
                "price_val": Column(T.IntegerType()),
            }
        )

        class TestSchema(DataFrameModel):
            """Test Schema class"""

            product: T.StringType = Field(str_startswith="B")
            price_val: T.StringType = Field()

        input_df = spark_df(spark, self.sample_data, sample_spark_schema)
        expected = {
            "validation_enabled": False,
            "validation_depth": ValidationDepth.SCHEMA_AND_DATA,
            "cache_dataframe": False,
            "keep_cached_dataframe": False,
        }

        with config_context(validation_enabled=False):
            assert asdict(get_config_context()) == expected
            assert pandera_schema.validate(input_df) == input_df
            assert TestSchema.validate(input_df) == input_df

    # pylint:disable=too-many-locals
    def test_schema_only(self, spark_session, sample_spark_schema, request):
        """This function validates that only schema related checks are run not data level"""
        spark = request.getfixturevalue(spark_session)
        pandera_schema = DataFrameSchema(
            {
                "product": Column(T.StringType(), Check.str_startswith("B")),
                "price_val": Column(T.IntegerType()),
            }
        )

        expected = {
            "validation_enabled": True,
            "validation_depth": ValidationDepth.SCHEMA_ONLY,
            "cache_dataframe": False,
            "keep_cached_dataframe": False,
        }
        input_df = spark_df(spark, self.sample_data, sample_spark_schema)

        with config_context(
            validation_enabled=True,
            validation_depth=ValidationDepth.SCHEMA_ONLY,
        ):
            assert asdict(get_config_context()) == expected
            output_dataframeschema_df = pandera_schema.validate(input_df)

        expected_dataframeschema = {
            "SCHEMA": {
                "COLUMN_NOT_IN_DATAFRAME": [
                    {
                        "check": "column_in_dataframe",
                        "column": None,
                        "error": "column "
                        "'price_val' not "
                        "in dataframe "
                        "Row(product='Bread', "
                        "price=9)",
                        "schema": None,
                    }
                ]
            }
        }

        assert (
            "DATA" not in dict(output_dataframeschema_df.pandera.errors).keys()
        )
        assert (
            dict(output_dataframeschema_df.pandera.errors["SCHEMA"])
            == expected_dataframeschema["SCHEMA"]
        )

        class TestSchema(DataFrameModel):
            """Test Schema"""

            product: T.StringType = Field(str_startswith="B")
            price_val: T.StringType = Field()

        with config_context(
            validation_enabled=True,
            validation_depth=ValidationDepth.SCHEMA_ONLY,
        ):
            output_dataframemodel_df = TestSchema.validate(input_df)

        expected_dataframemodel = {
            "SCHEMA": {
                "COLUMN_NOT_IN_DATAFRAME": [
                    {
                        "check": "column_in_dataframe",
                        "column": "TestSchema",
                        "error": "column "
                        "'price_val' not "
                        "in dataframe "
                        "Row(product='Bread', "
                        "price=9)",
                        "schema": "TestSchema",
                    }
                ]
            }
        }

        assert (
            "DATA" not in dict(output_dataframemodel_df.pandera.errors).keys()
        )
        assert (
            dict(output_dataframemodel_df.pandera.errors["SCHEMA"])
            == expected_dataframemodel["SCHEMA"]
        )

    # pylint:disable=too-many-locals
    def test_data_only(self, spark_session, sample_spark_schema, request):
        """This function validates that only data related checks are run not schema level"""
        spark = request.getfixturevalue(spark_session)
        pandera_schema = DataFrameSchema(
            {
                "product": Column(T.StringType(), Check.str_startswith("B")),
                "price_val": Column(T.IntegerType()),
            }
        )
        expected = {
            "validation_enabled": True,
            "validation_depth": ValidationDepth.DATA_ONLY,
            "cache_dataframe": False,
            "keep_cached_dataframe": False,
        }

        input_df = spark_df(spark, self.sample_data, sample_spark_schema)
        with config_context(
            validation_enabled=True,
            validation_depth=ValidationDepth.DATA_ONLY,
        ):
            assert asdict(get_config_context()) == expected
            output_dataframeschema_df = pandera_schema.validate(input_df)

        expected_dataframeschema = {
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
                        "schema": None,
                    }
                ]
            }
        }

        assert (
            "SCHEMA"
            not in dict(output_dataframeschema_df.pandera.errors).keys()
        )
        assert (
            dict(output_dataframeschema_df.pandera.errors["DATA"])
            == expected_dataframeschema["DATA"]
        )

        class TestSchema(DataFrameModel):
            """Test Schema"""

            product: T.StringType = Field(str_startswith="B")
            price_val: T.StringType = Field()

        with config_context(
            validation_enabled=True,
            validation_depth=ValidationDepth.DATA_ONLY,
        ):
            output_dataframemodel_df = TestSchema.validate(input_df)

        expected_dataframemodel = {
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
                        "schema": "TestSchema",
                    }
                ]
            }
        }

        assert (
            "SCHEMA"
            not in dict(output_dataframemodel_df.pandera.errors).keys()
        )
        assert (
            dict(output_dataframemodel_df.pandera.errors["DATA"])
            == expected_dataframemodel["DATA"]
        )

    # pylint:disable=too-many-locals
    def test_schema_and_data(
        self, spark_session, sample_spark_schema, request
    ):
        """This function validates that both data and schema level checks are validated"""
        spark = request.getfixturevalue(spark_session)
        # self.remove_python_module_cache()
        pandera_schema = DataFrameSchema(
            {
                "product": Column(T.StringType(), Check.str_startswith("B")),
                "price_val": Column(T.IntegerType()),
            }
        )
        expected = {
            "validation_enabled": True,
            "validation_depth": ValidationDepth.SCHEMA_AND_DATA,
            "cache_dataframe": False,
            "keep_cached_dataframe": False,
        }

        input_df = spark_df(spark, self.sample_data, sample_spark_schema)

        with config_context(
            validation_enabled=True,
            validation_depth=ValidationDepth.SCHEMA_AND_DATA,
        ):
            assert asdict(get_config_context()) == expected
            output_dataframeschema_df = pandera_schema.validate(input_df)

        expected_dataframeschema = {
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
                        "schema": None,
                    }
                ]
            },
            "SCHEMA": {
                "COLUMN_NOT_IN_DATAFRAME": [
                    {
                        "check": "column_in_dataframe",
                        "column": None,
                        "error": "column "
                        "'price_val' "
                        "not "
                        "in "
                        "dataframe "
                        "Row(product='Bread', "
                        "price=9)",
                        "schema": None,
                    }
                ]
            },
        }

        assert (
            dict(output_dataframeschema_df.pandera.errors["DATA"])
            == expected_dataframeschema["DATA"]
        )
        assert (
            dict(output_dataframeschema_df.pandera.errors["SCHEMA"])
            == expected_dataframeschema["SCHEMA"]
        )

        class TestSchema(DataFrameModel):
            """Test Schema"""

            product: T.StringType = Field(str_startswith="B")
            price_val: T.StringType = Field()

        with config_context(
            validation_enabled=True,
            validation_depth=ValidationDepth.SCHEMA_AND_DATA,
        ):
            output_dataframemodel_df = TestSchema.validate(input_df)

        expected_dataframemodel = {
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
                        "schema": "TestSchema",
                    }
                ]
            },
            "SCHEMA": {
                "COLUMN_NOT_IN_DATAFRAME": [
                    {
                        "check": "column_in_dataframe",
                        "column": "TestSchema",
                        "error": "column "
                        "'price_val' "
                        "not "
                        "in "
                        "dataframe "
                        "Row(product='Bread', "
                        "price=9)",
                        "schema": "TestSchema",
                    }
                ]
            },
        }

        assert (
            dict(output_dataframemodel_df.pandera.errors["DATA"])
            == expected_dataframemodel["DATA"]
        )
        assert (
            dict(output_dataframemodel_df.pandera.errors["SCHEMA"])
            == expected_dataframemodel["SCHEMA"]
        )

    @pytest.mark.parametrize("cache_dataframe", [True, False])
    @pytest.mark.parametrize("keep_cached_dataframe", [True, False])
    # pylint:disable=too-many-locals
    def test_cache_dataframe_settings(
        self,
        cache_dataframe,
        keep_cached_dataframe,
        spark_session,  # pylint:disable=unused-argument
    ):
        """This function validates setters and getters for cache/keep_cache options."""
        # Set expected properties in Config object
        # Evaluate expected Config
        expected = {
            "validation_enabled": True,
            "validation_depth": ValidationDepth.SCHEMA_AND_DATA,
            "cache_dataframe": cache_dataframe,
            "keep_cached_dataframe": keep_cached_dataframe,
        }
        with config_context(
            cache_dataframe=cache_dataframe,
            keep_cached_dataframe=keep_cached_dataframe,
        ):
            assert asdict(get_config_context()) == expected
