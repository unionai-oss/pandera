"""This module is to test the behaviour change based on defined config in pandera"""
# pylint:disable=import-outside-toplevel,abstract-method

import pyspark.sql.types as T
import pytest

from pandera.config import CONFIG, ValidationDepth
from pandera.pyspark import (
    Check,
    DataFrameSchema,
    Column,
    DataFrameModel,
    Field,
)
from tests.pyspark.conftest import spark_df


class TestPanderaConfig:
    """Class to test all the different configs types"""

    sample_data = [("Bread", 9), ("Cutter", 15)]

    def test_disable_validation(self, spark, sample_spark_schema):
        """This function validates that a none object is loaded if validation is disabled"""

        CONFIG.validation_enabled = False

        pandera_schema = DataFrameSchema(
            {
                "product": Column(T.StringType(), Check.str_startswith("B")),
                "price_val": Column(T.IntegerType()),
            }
        )

        class TestSchema(DataFrameModel):
            """Test Schema class"""

            product: T.StringType() = Field(str_startswith="B")
            price_val: T.StringType() = Field()

        input_df = spark_df(spark, self.sample_data, sample_spark_schema)
        expected = {
            "validation_enabled": False,
            "validation_depth": ValidationDepth.SCHEMA_AND_DATA,
            "pyspark_cache": False,
            "pyspark_unpersist": True,
        }

        assert CONFIG.dict() == expected
        assert pandera_schema.validate(input_df) is not None
        assert TestSchema.validate(input_df) is not None

    # pylint:disable=too-many-locals
    def test_schema_only(self, spark, sample_spark_schema):
        """This function validates that only schema related checks are run not data level"""
        CONFIG.validation_enabled = True
        CONFIG.validation_depth = ValidationDepth.SCHEMA_ONLY

        pandera_schema = DataFrameSchema(
            {
                "product": Column(T.StringType(), Check.str_startswith("B")),
                "price_val": Column(T.IntegerType()),
            }
        )

        expected = {
            "validation_enabled": True,
            "validation_depth": ValidationDepth.SCHEMA_ONLY,
            "pyspark_cache": False,
            "pyspark_unpersist": True,
        }
        assert CONFIG.dict() == expected

        input_df = spark_df(spark, self.sample_data, sample_spark_schema)
        output_dataframeschema_df = pandera_schema.validate(input_df)
        expected_dataframeschema = {
            "SCHEMA": {
                "COLUMN_NOT_IN_DATAFRAME": [
                    {
                        "check": "column_in_dataframe",
                        "column": None,
                        "error": "column "
                        "'price_val' not "
                        "in dataframe\n"
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

            product: T.StringType() = Field(str_startswith="B")
            price_val: T.StringType() = Field()

        output_dataframemodel_df = TestSchema.validate(input_df)

        expected_dataframemodel = {
            "SCHEMA": {
                "COLUMN_NOT_IN_DATAFRAME": [
                    {
                        "check": "column_in_dataframe",
                        "column": "TestSchema",
                        "error": "column "
                        "'price_val' not "
                        "in dataframe\n"
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
    def test_data_only(self, spark, sample_spark_schema):
        """This function validates that only data related checks are run not schema level"""
        CONFIG.validation_enabled = True
        CONFIG.validation_depth = ValidationDepth.DATA_ONLY

        pandera_schema = DataFrameSchema(
            {
                "product": Column(T.StringType(), Check.str_startswith("B")),
                "price_val": Column(T.IntegerType()),
            }
        )
        expected = {
            "validation_enabled": True,
            "validation_depth": ValidationDepth.DATA_ONLY,
            "pyspark_cache": False,
            "pyspark_unpersist": True,
        }
        assert CONFIG.dict() == expected

        input_df = spark_df(spark, self.sample_data, sample_spark_schema)
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

            product: T.StringType() = Field(str_startswith="B")
            price_val: T.StringType() = Field()

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
    def test_schema_and_data(self, spark, sample_spark_schema):
        """This function validates that both data and schema level checks are validated"""
        # self.remove_python_module_cache()
        CONFIG.validation_enabled = True
        CONFIG.validation_depth = ValidationDepth.SCHEMA_AND_DATA

        pandera_schema = DataFrameSchema(
            {
                "product": Column(T.StringType(), Check.str_startswith("B")),
                "price_val": Column(T.IntegerType()),
            }
        )
        expected = {
            "validation_enabled": True,
            "validation_depth": ValidationDepth.SCHEMA_AND_DATA,
            "pyspark_cache": False,
            "pyspark_unpersist": True,
        }
        assert CONFIG.dict() == expected

        input_df = spark_df(spark, self.sample_data, sample_spark_schema)
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
                        "dataframe\n"
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

            product: T.StringType() = Field(str_startswith="B")
            price_val: T.StringType() = Field()

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
                        "dataframe\n"
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

    @pytest.mark.parametrize("cache_enabled", [True, False])
    @pytest.mark.parametrize("unpersist_enabled", [True, False])
    # pylint:disable=too-many-locals
    def test_pyspark_cache_settings(
        self,
        cache_enabled,
        unpersist_enabled,
    ):
        """This function validates setter and getters of caching/unpersisting options."""
        # Set expected properties in Config object
        CONFIG.pyspark_cache = cache_enabled
        CONFIG.pyspark_unpersist = unpersist_enabled

        # Evaluate expected Config
        expected = {
            "validation_enabled": True,
            "validation_depth": ValidationDepth.SCHEMA_AND_DATA,
            "pyspark_cache": cache_enabled,
            "pyspark_unpersist": unpersist_enabled,
        }
        assert CONFIG.dict() == expected
