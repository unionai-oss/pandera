"""This module is to test the behaviour change based on defined config in pandera"""
# pylint:disable=import-outside-toplevel,abstract-method

import pyspark.sql.types as T
from tests.pyspark.conftest import spark_df


class TestConfigParams:
    """Class to test all the different configs types"""

    sample_data = [("Bread", 9), ("Cutter", 15)]

    @staticmethod
    def remove_python_module_cache():
        """
        This function removes the imports from the python module cache to re-instantiate the object
        refer: https://docs.python.org/3/reference/import.html#the-module-cache
        """

        import sys

        values_to_del = []
        # Pyspark module cache loads the modules need to rerun the find all instances of case
        for key in sys.modules:
            if key.startswith("pandera"):
                values_to_del.append(key)
        # Separate loop for delete since removing the module from cache is
        # a runtime if done in same loop causes runtime error
        for value in values_to_del:
            del sys.modules[value]

    def test_disable_validation(self, spark, sample_spark_schema, monkeypatch):
        """This function validates that a none object is loaded if validation is disabled"""

        # Need to do imports and schema definition in code to ensure the object is instantiated from scratch
        self.remove_python_module_cache()
        monkeypatch.setenv("PANDERA_VALIDATION", "DISABLE")

        import pandera
        from pandera.pyspark import (
            DataFrameSchema,
            Column,
            DataFrameModel,
            Field,
        )
        from pandera.backends.pyspark.utils import ConfigParams

        pandra_schema = DataFrameSchema(
            {
                "product": Column(
                    T.StringType(), pandera.Check.str_startswith("B")
                ),
                "price_val": Column(T.IntegerType()),
            }
        )

        class TestSchema(DataFrameModel):
            """Test Schema class"""

            product: T.StringType() = Field(str_startswith="B")
            price_val: T.StringType() = Field()

        params = ConfigParams()
        input_df = spark_df(spark, self.sample_data, sample_spark_schema)
        expected = {"PANDERA_VALIDATION": "DISABLE", "PANDERA_DEPTH": "SCHEMA_AND_DATA"}

        assert dict(params) == expected
        assert pandra_schema.validate(input_df) is None
        assert TestSchema.validate(input_df) is None

    def test_schema_only(
        self, spark, sample_spark_schema, monkeypatch
    ):  # pylint:disable=too-many-locals
        """This function validates that only schema related checks are run not data level"""
        self.remove_python_module_cache()
        monkeypatch.setenv("PANDERA_VALIDATION", "ENABLE")
        monkeypatch.setenv("PANDERA_DEPTH", "SCHEMA_ONLY")
        # Need to do imports and schema definition in code to ensure the object is instantiated from scratch

        import pandera
        from pandera.pyspark import (
            DataFrameSchema,
            Column,
            DataFrameModel,
            Field,
        )
        from pandera.backends.pyspark.utils import ConfigParams

        params = ConfigParams()
        pandra_schema = DataFrameSchema(
            {
                "product": Column(
                    T.StringType(), pandera.Check.str_startswith("B")
                ),
                "price_val": Column(T.IntegerType()),
            }
        )

        expected = {"PANDERA_VALIDATION": "ENABLE", "PANDERA_DEPTH": "SCHEMA_ONLY"}
        assert dict(params) == expected

        input_df = spark_df(spark, self.sample_data, sample_spark_schema)
        output_dataframeschema_df = pandra_schema.validate(input_df)
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

    def test_data_only(
        self, spark, sample_spark_schema, monkeypatch
    ):  # pylint:disable=too-many-locals
        """This function validates that only data related checks are run not schema level"""
        self.remove_python_module_cache()
        monkeypatch.setenv("PANDERA_VALIDATION", "ENABLE")
        monkeypatch.setenv("PANDERA_DEPTH", "DATA_ONLY")

        import pandera
        from pandera.pyspark import (
            DataFrameSchema,
            Column,
            DataFrameModel,
            Field,
        )
        from pandera.backends.pyspark.utils import ConfigParams

        params = ConfigParams()
        pandra_schema = DataFrameSchema(
            {
                "product": Column(
                    T.StringType(), pandera.Check.str_startswith("B")
                ),
                "price_val": Column(T.IntegerType()),
            }
        )

        expected = {"PANDERA_VALIDATION": "ENABLE", "PANDERA_DEPTH": "DATA_ONLY"}
        assert dict(params) == expected

        input_df = spark_df(spark, self.sample_data, sample_spark_schema)
        output_dataframeschema_df = pandra_schema.validate(input_df)
        expected_dataframeschema = {
            "DATA": {
                "DATAFRAME_CHECK": [
                    {
                        "check": "str_startswith('B')",
                        "column": "product",
                        "error": "column "
                        "'product' "
                        "with type "
                        "StringType() "
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
                        "StringType() "
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

    def test_schema_and_data(
        self, spark, sample_spark_schema, monkeypatch
    ):  # pylint:disable=too-many-locals
        """This function validates that both data and schema level checks are validated"""
        self.remove_python_module_cache()
        monkeypatch.setenv("PANDERA_VALIDATION", "ENABLE")
        monkeypatch.setenv("PANDERA_DEPTH", "SCHEMA_AND_DATA")

        import pandera
        from pandera.pyspark import (
            DataFrameSchema,
            Column,
            DataFrameModel,
            Field,
        )
        from pandera.backends.pyspark.utils import ConfigParams

        params = ConfigParams()
        pandra_schema = DataFrameSchema(
            {
                "product": Column(
                    T.StringType(), pandera.Check.str_startswith("B")
                ),
                "price_val": Column(T.IntegerType()),
            }
        )

        expected = {"PANDERA_VALIDATION": "ENABLE", "PANDERA_DEPTH": "SCHEMA_AND_DATA"}
        assert dict(params) == expected

        input_df = spark_df(spark, self.sample_data, sample_spark_schema)
        output_dataframeschema_df = pandra_schema.validate(input_df)
        expected_dataframeschema = {
            "DATA": {
                "DATAFRAME_CHECK": [
                    {
                        "check": "str_startswith('B')",
                        "column": "product",
                        "error": "column "
                        "'product' "
                        "with type "
                        "StringType() "
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
                        "StringType() "
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
