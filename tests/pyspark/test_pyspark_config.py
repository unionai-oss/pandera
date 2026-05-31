"""This module is to test the behaviour change based on defined config in pandera"""

from dataclasses import asdict

import pyspark.sql.types as T
import pytest

from pandera.config import (
    CONFIG,
    ValidationDepth,
    config_context,
    get_config_context,
)
from pandera.pyspark import (
    Check,
    Column,
    DataFrameModel,
    DataFrameSchema,
    Field,
)
from tests.pyspark.conftest import (
    _cmp_errors,
    spark_df,
    validate_collecting_errors,
)

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
            "use_narwhals_backend": CONFIG.use_narwhals_backend,
            "silenced_warnings": [],
        }

        with config_context(validation_enabled=False):
            assert asdict(get_config_context()) == expected
            assert pandera_schema.validate(input_df) == input_df
            assert TestSchema.validate(input_df) == input_df

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
            "use_narwhals_backend": CONFIG.use_narwhals_backend,
            "silenced_warnings": [],
        }
        input_df = spark_df(spark, self.sample_data, sample_spark_schema)

        with config_context(
            validation_enabled=True,
            validation_depth=ValidationDepth.SCHEMA_ONLY,
        ):
            assert asdict(get_config_context()) == expected
            _, schema_errors = validate_collecting_errors(
                pandera_schema, input_df
            )

        expected_dataframeschema = {
            "SCHEMA": {
                "COLUMN_NOT_IN_DATAFRAME": [
                    {
                        "check": "column_in_dataframe",
                        "column": "price_val",
                        "schema": None,
                    }
                ]
            }
        }

        assert "DATA" not in dict(schema_errors).keys()
        _cmp_errors(
            dict(schema_errors["SCHEMA"]),
            expected_dataframeschema["SCHEMA"],
        )

        class TestSchema(DataFrameModel):
            """Test Schema"""

            product: T.StringType = Field(str_startswith="B")
            price_val: T.StringType = Field()

        with config_context(
            validation_enabled=True,
            validation_depth=ValidationDepth.SCHEMA_ONLY,
        ):
            _, model_errors = validate_collecting_errors(TestSchema, input_df)

        expected_dataframemodel = {
            "SCHEMA": {
                "COLUMN_NOT_IN_DATAFRAME": [
                    {
                        "check": "column_in_dataframe",
                        "column": "price_val",
                        "schema": "TestSchema",
                    }
                ]
            }
        }

        assert "DATA" not in dict(model_errors).keys()
        _cmp_errors(
            dict(model_errors["SCHEMA"]),
            expected_dataframemodel["SCHEMA"],
        )

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
            "use_narwhals_backend": CONFIG.use_narwhals_backend,
            "silenced_warnings": [],
        }

        input_df = spark_df(spark, self.sample_data, sample_spark_schema)
        with config_context(
            validation_enabled=True,
            validation_depth=ValidationDepth.DATA_ONLY,
        ):
            assert asdict(get_config_context()) == expected
            _, schema_errors = validate_collecting_errors(
                pandera_schema, input_df
            )

        expected_dataframeschema = {
            "DATA": {
                "DATAFRAME_CHECK": [
                    {
                        "check": "str_startswith('B')",
                        "column": "product",
                        "schema": None,
                    }
                ]
            }
        }

        assert "SCHEMA" not in dict(schema_errors).keys()
        _cmp_errors(
            dict(schema_errors["DATA"]),
            expected_dataframeschema["DATA"],
        )

        class TestSchema(DataFrameModel):
            """Test Schema"""

            product: T.StringType = Field(str_startswith="B")
            price_val: T.StringType = Field()

        with config_context(
            validation_enabled=True,
            validation_depth=ValidationDepth.DATA_ONLY,
        ):
            _, model_errors = validate_collecting_errors(TestSchema, input_df)

        expected_dataframemodel = {
            "DATA": {
                "DATAFRAME_CHECK": [
                    {
                        "check": "str_startswith('B')",
                        "column": "product",
                        "schema": "TestSchema",
                    }
                ]
            }
        }

        assert "SCHEMA" not in dict(model_errors).keys()
        _cmp_errors(
            dict(model_errors["DATA"]),
            expected_dataframemodel["DATA"],
        )

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
            "use_narwhals_backend": CONFIG.use_narwhals_backend,
            "silenced_warnings": [],
        }

        input_df = spark_df(spark, self.sample_data, sample_spark_schema)

        with config_context(
            validation_enabled=True,
            validation_depth=ValidationDepth.SCHEMA_AND_DATA,
        ):
            assert asdict(get_config_context()) == expected
            _, schema_errors = validate_collecting_errors(
                pandera_schema, input_df
            )

        expected_dataframeschema = {
            "DATA": {
                "DATAFRAME_CHECK": [
                    {
                        "check": "str_startswith('B')",
                        "column": "product",
                        "schema": None,
                    }
                ]
            },
            "SCHEMA": {
                "COLUMN_NOT_IN_DATAFRAME": [
                    {
                        "check": "column_in_dataframe",
                        "column": "price_val",
                        "schema": None,
                    }
                ]
            },
        }

        _cmp_errors(
            dict(schema_errors["DATA"]),
            expected_dataframeschema["DATA"],
        )
        _cmp_errors(
            dict(schema_errors["SCHEMA"]),
            expected_dataframeschema["SCHEMA"],
        )

        class TestSchema(DataFrameModel):
            """Test Schema"""

            product: T.StringType = Field(str_startswith="B")
            price_val: T.StringType = Field()

        with config_context(
            validation_enabled=True,
            validation_depth=ValidationDepth.SCHEMA_AND_DATA,
        ):
            _, model_errors = validate_collecting_errors(TestSchema, input_df)

        expected_dataframemodel = {
            "DATA": {
                "DATAFRAME_CHECK": [
                    {
                        "check": "str_startswith('B')",
                        "column": "product",
                        "schema": "TestSchema",
                    }
                ]
            },
            "SCHEMA": {
                "COLUMN_NOT_IN_DATAFRAME": [
                    {
                        "check": "column_in_dataframe",
                        "column": "price_val",
                        "schema": "TestSchema",
                    }
                ]
            },
        }

        _cmp_errors(
            dict(model_errors["DATA"]),
            expected_dataframemodel["DATA"],
        )
        _cmp_errors(
            dict(model_errors["SCHEMA"]),
            expected_dataframemodel["SCHEMA"],
        )

    @pytest.mark.parametrize("cache_dataframe", [True, False])
    @pytest.mark.parametrize("keep_cached_dataframe", [True, False])
    def test_cache_dataframe_settings(
        self,
        cache_dataframe,
        keep_cached_dataframe,
        spark_session,
    ):
        """This function validates setters and getters for cache/keep_cache options."""
        # Set expected properties in Config object
        # Evaluate expected Config
        expected = {
            "validation_enabled": True,
            "validation_depth": ValidationDepth.SCHEMA_AND_DATA,
            "cache_dataframe": cache_dataframe,
            "keep_cached_dataframe": keep_cached_dataframe,
            "use_narwhals_backend": CONFIG.use_narwhals_backend,
            "silenced_warnings": [],
        }
        with config_context(
            cache_dataframe=cache_dataframe,
            keep_cached_dataframe=keep_cached_dataframe,
        ):
            assert asdict(get_config_context()) == expected
