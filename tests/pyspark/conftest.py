""" conftest """
import pytest
from pyspark.sql import SparkSession
import pyspark.sql.types as T
import datetime


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
            T.StructField("product", T.StringType(), True),
            T.StructField("price", T.IntegerType(), True),
        ],
    )


def spark_df(spark, data: list, spark_schema: T.StructType):
    return spark.createDataFrame(data=data, schema=spark_schema, verifySchema=False)


@pytest.fixture(scope='session')
def sample_date_object(spark):
    sample_data = [
        (
            datetime.date(2022, 10, 1),
            datetime.datetime(2022, 10, 1, 5, 32, 0),
            datetime.timedelta(45),
            datetime.timedelta(45),
        ),
        (
            datetime.date(2022, 11, 5),
            datetime.datetime(2022, 11, 5, 15, 34, 0),
            datetime.timedelta(30),
            datetime.timedelta(45),
        ),
    ]
    sample_spark_schema = T.StructType(
        [
            T.StructField("purchase_date", T.DateType(), False),
            T.StructField("purchase_datetime", T.TimestampType(), False),
            T.StructField("expiry_time", T.DayTimeIntervalType(), False),
            T.StructField("expected_time", T.DayTimeIntervalType(2, 3), False),
        ],
    )
    df = spark_df(spark=spark, spark_schema=sample_spark_schema, data=sample_data)
    return df

@pytest.fixture(scope='session')
def sample_string_binary_object(spark):
    sample_data = [
        (
            'test1',
            'Bread',
        ),
        (
            'test2',
            "Butter"
        ),
    ]
    sample_spark_schema = T.StructType(
        [
            T.StructField("purchase_info", T.StringType(), False),
            T.StructField("product", T.StringType(), False),
        ],
    )
    df = spark_df(spark=spark, spark_schema=sample_spark_schema, data=sample_data)
    df = df.withColumn('purchase_info', df['purchase_info'].cast(T.BinaryType()))
    return df

@pytest.fixture(scope='session')
def sample_complex_data(spark):
    sample_data = [
        (datetime.date(2022, 10, 1), [["josh"], ["27"]], {"product_bought": "bread"}),
        (datetime.date(2022, 11, 5), [["Adam"], ["22"]], {"product_bought": "bread"}),
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
                T.MapType(
                    T.StringType(),
                    T.StringType()
                ),
                False,
            ),
        ],
    )
    return spark_df(spark, sample_data, sample_spark_schema)

@pytest.fixture(scope='session')
def sample_check_data(spark):

    return {"test_pass_data": [("foo", 30), ("bar", 30)],
     "test_fail_data": [("foo", 30), ("bar", 31)],
     "test_expression": 30}



import pyspark.sql.types as T
from conftest import spark_df

class TestConfigParams:
    sample_data = [("Bread", 9), ("Cutter", 15)]
    def test_disable_validation(self, spark, sample_spark_schema):

        # Need to do imports and schema definition in code to ensure the object is initiated
        import sys
        sys.argv.append('--validation')
        sys.argv.append('DISABLE')
        import pandera
        from pandera.pyspark import DataFrameSchema, Column, DataFrameModel, Field
        from pandera.backends.pyspark.utils import ConfigParams, PANDERA_CONFIG

        pandra_schema = DataFrameSchema(
                {
                    "product": Column(T.StringType(), pandera.Check.str_startswith("B")),
                    "price_val": Column(T.IntegerType()),
                }
            )


        class TestSchema(DataFrameModel):
            product: T.StringType() = Field(str_startswith='B')
            price_val: T.StringType() = Field()

        params = ConfigParams()
        expected = {'VALIDATION': 'DISABLE', 'DEPTH': 'SCHEMA_AND_DATA'}
        assert dict(params) == expected
        input_df = spark_df(spark, self.sample_data, sample_spark_schema)
        assert pandra_schema.validate(input_df) is None
        assert TestSchema.validate(input_df) is None

    def test_schema_only(self, spark, sample_spark_schema):
        import sys

        sys.argv = sys.argv[:2]
        sys.argv.append('--validation')
        sys.argv.append('ENABLE')
        sys.argv.append('--depth')
        sys.argv.append('SCHEMA_ONLY')

        import pandera
        from pandera.pyspark import DataFrameSchema, Column, DataFrameModel, Field
        from pandera.backends.pyspark.utils import ConfigParams

        params = ConfigParams()
        pandra_schema = DataFrameSchema(
            {
                "product": Column(T.StringType(), pandera.Check.str_startswith("B")),
                "price_val": Column(T.IntegerType()),
            }
        )

        expected = {'VALIDATION': 'ENABLE', 'DEPTH': 'SCHEMA_ONLY'}
        assert dict(params) == expected
        input_df = spark_df(spark, self.sample_data, sample_spark_schema)
        output_dataframeschema_df = pandra_schema.validate(input_df)
        expected_dataframeschema = {'SCHEMA': {'COLUMN_NOT_IN_DATAFRAME': [{'check': 'column_in_dataframe',
                                                                            'column': None,
                                                                            'error': 'column '
                                                                                     "'price_val' not "
                                                                                     'in dataframe\n'
                                                                                     "Row(product='Bread', "
                                                                                     'price=9)',
                                                                            'schema': None}]}}
        assert 'DATA' not in dict(output_dataframeschema_df.pandera.errors).keys()

        assert dict(output_dataframeschema_df.pandera.errors['SCHEMA']) == expected_dataframeschema['SCHEMA']

        class TestSchema(DataFrameModel):
            product: T.StringType() = Field(str_startswith='B')
            price_val: T.StringType() = Field()

        output_dataframemodel_df = TestSchema.validate(input_df)

        expected_dataframemodel = {'SCHEMA': {'COLUMN_NOT_IN_DATAFRAME': [{'check': 'column_in_dataframe',
                                                                           'column': 'TestSchema',
                                                                           'error': 'column '
                                                                                    "'price_val' not "
                                                                                    'in dataframe\n'
                                                                                    "Row(product='Bread', "
                                                                                    'price=9)',
                                                                           'schema': 'TestSchema'}]}}

        assert 'DATA' not in dict(output_dataframemodel_df.pandera.errors).keys()

        assert dict(output_dataframemodel_df.pandera.errors['SCHEMA']) == expected_dataframemodel['SCHEMA']

    def test_data_only(self, spark, sample_spark_schema):
        import sys
        sys.argv.append('--validation')
        sys.argv.append('ENABLE')
        sys.argv.append('--depth')
        sys.argv.append('DATA_ONLY')
        import pandera
        from pandera.pyspark import DataFrameSchema, Column, DataFrameModel, Field
        from pandera.backends.pyspark.utils import ConfigParams
        params = ConfigParams()
        pandra_schema = DataFrameSchema(
            {
                "product": Column(T.StringType(), pandera.Check.str_startswith("B")),
                "price_val": Column(T.IntegerType()),
            }
        )

        expected = {'VALIDATION': 'ENABLE', 'DEPTH': 'DATA_ONLY'}
        assert dict(params) == expected
        input_df = spark_df(spark, self.sample_data, sample_spark_schema)
        output_dataframeschema_df = pandra_schema.validate(input_df)
        expected_dataframeschema = {'DATA': {'DATAFRAME_CHECK': [{'check': "str_startswith('B')",
                                                                  'column': 'product',
                                                                  'error': 'column '
                                                                           "'product' "
                                                                           'with type '
                                                                           'StringType() '
                                                                           'failed '
                                                                           'validation '
                                                                           "str_startswith('B')",
                                                                  'schema': None}]}}
        assert 'SCHEMA' not in dict(output_dataframeschema_df.pandera.errors).keys()

        assert dict(output_dataframeschema_df.pandera.errors['DATA']) == expected_dataframeschema['DATA']

        class TestSchema(DataFrameModel):
            product: T.StringType() = Field(str_startswith='B')
            price_val: T.StringType() = Field()

        output_dataframemodel_df = TestSchema.validate(input_df)

        expected_dataframemodel = {'DATA': {'DATAFRAME_CHECK': [{'check': "str_startswith('B')",
                                                                 'column': 'product',
                                                                 'error': 'column '
                                                                          "'product' "
                                                                          'with type '
                                                                          'StringType() '
                                                                          'failed '
                                                                          'validation '
                                                                          "str_startswith('B')",
                                                                 'schema': "TestSchema"}]}}

        assert 'SCHEMA' not in dict(output_dataframemodel_df.pandera.errors).keys()

        assert dict(output_dataframemodel_df.pandera.errors['DATA']) == expected_dataframemodel['DATA']

    def test_schema_and_data(self, spark, sample_spark_schema):
        import sys
        sys.argv.append('--validation')
        sys.argv.append('ENABLE')
        sys.argv.append('--depth')
        sys.argv.append('SCHEMA_AND_DATA')
        import pandera
        from pandera.pyspark import DataFrameSchema, Column, DataFrameModel, Field
        from pandera.backends.pyspark.utils import ConfigParams
        params = ConfigParams()
        pandra_schema = DataFrameSchema(
            {
                "product": Column(T.StringType(), pandera.Check.str_startswith("B")),
                "price_val": Column(T.IntegerType()),
            }
        )

        expected = {'VALIDATION': 'ENABLE', 'DEPTH': 'SCHEMA_AND_DATA'}
        assert dict(params) == expected
        input_df = spark_df(spark, self.sample_data, sample_spark_schema)
        output_dataframeschema_df = pandra_schema.validate(input_df)
        expected_dataframeschema = {
            'DATA': {'DATAFRAME_CHECK': [
                {'check': "str_startswith('B')",
                 'column': 'product',
                 'error': 'column '
                          "'product' "
                          'with type '
                          'StringType() '
                          'failed '
                          'validation '
                          "str_startswith('B')",
                 'schema': None}
            ]
            },
            'SCHEMA': {'COLUMN_NOT_IN_DATAFRAME': [
                {'check': 'column_in_dataframe',
                 'column': None,
                 'error': 'column '
                          "'price_val' "
                          'not '
                          'in '
                          'dataframe\n'
                          "Row(product='Bread', "
                          'price=9)',
                 'schema': None}
            ]
            }
        }

        assert dict(output_dataframeschema_df.pandera.errors['DATA']) == expected_dataframeschema['DATA']
        assert dict(output_dataframeschema_df.pandera.errors['SCHEMA']) == expected_dataframeschema['SCHEMA']

        class TestSchema(DataFrameModel):
            product: T.StringType() = Field(str_startswith='B')
            price_val: T.StringType() = Field()

        output_dataframemodel_df = TestSchema.validate(input_df)

        expected_dataframemodel = {
            'DATA': {'DATAFRAME_CHECK': [
                {'check': "str_startswith('B')",
                 'column': 'product',
                 'error': 'column '
                          "'product' "
                          'with type '
                          'StringType() '
                          'failed '
                          'validation '
                          "str_startswith('B')",
                 'schema': "TestSchema"}
            ]
            },
            'SCHEMA': {'COLUMN_NOT_IN_DATAFRAME': [
                {'check': 'column_in_dataframe',
                 'column': 'TestSchema',
                 'error': 'column '
                          "'price_val' "
                          'not '
                          'in '
                          'dataframe\n'
                          "Row(product='Bread', "
                          'price=9)',
                 'schema': 'TestSchema'}
            ]
            }
        }

        assert dict(output_dataframemodel_df.pandera.errors['DATA']) == expected_dataframemodel['DATA']
        assert dict(output_dataframemodel_df.pandera.errors['SCHEMA']) == expected_dataframemodel['SCHEMA']

@pytest.fixture(scope='session')
def config_params():
    from pandera.backends.pyspark.utils import ConfigParams
    return ConfigParams()
