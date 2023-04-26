"""Unit tests for pyspark container."""
import datetime
from typing import Union

from pyspark.sql import SparkSession
import copy
from pyspark.sql.types import (LongType,
                               StringType,
                               StructField,
                               StructType,
                               IntegerType,
                               ByteType,
                               ShortType,
                               TimestampType,
                               DateType,
                               DecimalType,
                               DoubleType,
                               BooleanType,
                               FloatType)
import decimal
import pyspark.sql.functions as F
import pytest
import pandera as pa
import pandera.errors
from pandera.api.pyspark.container import DataFrameSchema
from pandera.api.pyspark.components import Column
from pandera.errors import SchemaErrors, PysparkSchemaError
from tests.pyspark.conftest import spark_df

def check_function(spark, check_fn, pass_case_data,
                   fail_case_data, data_types, function_args):
    schema = DataFrameSchema(
        {
            "product": Column(StringType()),
            "code": Column(data_types(), check_fn(*function_args)) if isinstance(function_args, tuple) else
            Column(data_types(), check_fn(function_args)),
        }
    )
    spark_schema = StructType(
        [
            StructField("product", StringType(), False),
            StructField("code", data_types(), False),
        ],
    )
    df = spark.createDataFrame(data=pass_case_data, schema=spark_schema)
    validate_fail_error = schema.validate(df)
    if validate_fail_error:
        raise PysparkSchemaError
    with pytest.raises(PysparkSchemaError):
        df_fail = spark.createDataFrame(data=fail_case_data, schema=spark_schema)
        validate_fail_error = schema.validate(df_fail)
        if validate_fail_error:
            raise PysparkSchemaError



def test_datatype_check_decorator(spark):
    schema = DataFrameSchema(
        {
            "product": Column(StringType()),
            "code": Column(StringType(), pa.Check.str_startswith('B')),
        }
    )


    spark_schema = StructType(
        [
            StructField("product", StringType(), False),
            StructField("code", StringType(), False),
        ],
    )
    pass_case_data = [("foo", 'B1'), ("bar", 'B2')],
    df = spark.createDataFrame(data=pass_case_data, schema=spark_schema)
    validate_fail_error = schema.report_errors(df)
    print(validate_fail_error)
    #with pytest.raises(TypeError):
    fail_schema = DataFrameSchema(
        {
            "product": Column(StringType()),
            "code": Column(IntegerType(), pa.Check.str_startswith('B')),
        }
    )

    spark_schema = StructType(
        [
            StructField("product", StringType(), False),
            StructField("code", IntegerType(), False),
        ],
    )
    fail_case_data = [["foo", 1], ["bar", 2]]
    df = spark.createDataFrame(data=fail_case_data, schema=spark_schema)
    validate_fail_error = schema.report_errors(df)
    print(validate_fail_error)
    df = spark.createDataFrame(data=fail_case_data, schema=spark_schema)
    validate_fail_error = fail_schema.report_errors(df)


@pytest.mark.parametrize("data_dictionary",  [{"Datatype": LongType,
                                               "test_pass_data": [("foo", 30), ("bar", 30)],
                                               "test_fail_data": [("foo", 30), ("bar", 31)],
                                               "test_expression": 30},
                                              {"Datatype": IntegerType,
                                               "test_pass_data": [("foo", 30), ("bar", 30)],
                                               "test_fail_data": [("foo", 30), ("bar", 31)],
                                               "test_expression": 30
                                               },
                                              {"Datatype": ByteType,
                                               "test_pass_data": [("foo", 30), ("bar", 30)],
                                               "test_fail_data": [("foo", 30), ("bar", 31)],
                                               "test_expression": 30
                                               },
                                              {"Datatype": ShortType,
                                               "test_pass_data": [("foo", 30), ("bar", 30)],
                                               "test_fail_data": [("foo", 30), ("bar", 31)],
                                               "test_expression": 30
                                               },
                                              {"Datatype": StringType,
                                               "test_pass_data": [("foo", "a"), ("bar", "a")],
                                               "test_fail_data": [("foo", "a"), ("bar", "b")],
                                               "test_expression": "a"
                                               },
                                              {"Datatype": DoubleType,
                                               "test_pass_data": [("foo", 30.0), ("bar", 30.0)],
                                               "test_fail_data": [("foo", 30.0), ("bar", 31.0)],
                                               "test_expression": 30.0},
                                              {"Datatype": TimestampType,
                                               "test_pass_data": [("foo", datetime.datetime(2020, 10, 1, 10, 0)),
                                                                  ("bar",  datetime.datetime(2020, 10, 1,  10, 0))],
                                               "test_fail_data": [("foo",  datetime.datetime(2020, 10, 2,  10, 0)),
                                                                  ("bar",  datetime.datetime(2020, 10, 2,  10, 0))],
                                               "test_expression": datetime.datetime(2020, 10, 1,  10, 0)
                                               },
                                              {"Datatype": DateType,
                                               "test_pass_data": [("foo", datetime.date(2020, 10, 1)),
                                                                  ("bar", datetime.date(2020, 10, 1))],
                                               "test_fail_data": [("foo", datetime.date(2020, 10, 1)),
                                                                  ("bar", datetime.date(2020, 10, 2))],
                                               "test_expression": datetime.date(2020, 10, 1)},
                                              {"Datatype": DecimalType,
                                               "test_pass_data": [("foo", decimal.Decimal(30)), ("bar", decimal.Decimal(30))],
                                               "test_fail_data": [("foo", decimal.Decimal(30)), ("bar", decimal.Decimal(31))],
                                               "test_expression": decimal.Decimal(30)},
                                              {"Datatype": FloatType,
                                               "test_pass_data": [("foo", 30.0), ("bar", 30.0)],
                                               "test_fail_data": [("foo", 30.0), ("bar", 31.0)],
                                               "test_expression": 30.0
                                               },
                                              {"Datatype": BooleanType,
                                               "test_pass_data": [("foo", True), ("bar", True)],
                                               "test_fail_data": [("foo", False), ("bar", False)],
                                              "test_expression": True}
                                              ]
                         )
@pytest.mark.parametrize("check_fn", [pa.Check.equal_to, pa.Check.eq])
def test_equal_to_check(spark, check_fn, data_dictionary) -> None:
    """Test the Check to see if all the values are equal to defined value"""

    check_function(spark, check_fn, data_dictionary['test_pass_data'], data_dictionary['test_fail_data'],
                   data_dictionary['Datatype'], data_dictionary['test_expression'])



@pytest.mark.parametrize("data_dictionary",  [{"Datatype": LongType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 30), ("bar", 31)],
                                               "test_expression": 30},
                                              {"Datatype": IntegerType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 30), ("bar", 31)],
                                               "test_expression": 30
                                               },
                                              {"Datatype": ByteType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 30), ("bar", 31)],
                                               "test_expression": 30
                                               },
                                              {"Datatype": ShortType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 30), ("bar", 31)],
                                               "test_expression": 30
                                               },
                                              {"Datatype": StringType,
                                               "test_pass_data": [("foo", "b"), ("bar", "c")],
                                               "test_fail_data": [("foo", "a"), ("bar", "b")],
                                               "test_expression": "a"
                                               },
                                              {"Datatype": DoubleType,
                                               "test_pass_data": [("foo", 31.0), ("bar", 32.0)],
                                               "test_fail_data": [("foo", 30.0), ("bar", 31.0)],
                                               "test_expression": 30.0},
                                              {"Datatype": TimestampType,
                                               "test_pass_data": [("foo", datetime.datetime(2020, 10, 1, 10, 0)),
                                                                  ("bar",  datetime.datetime(2020, 10, 2,  10, 0))],
                                               "test_fail_data": [("foo",  datetime.datetime(2020, 10, 3,  10, 0)),
                                                                  ("bar",  datetime.datetime(2020, 10, 2,  10, 0))],
                                               "test_expression": datetime.datetime(2020, 10, 3,  10, 0)
                                               },
                                              {"Datatype": DateType,
                                               "test_pass_data": [("foo", datetime.date(2020, 10, 1)),
                                                                  ("bar", datetime.date(2020, 10, 2))],
                                               "test_fail_data": [("foo", datetime.date(2020, 10, 2)),
                                                                  ("bar", datetime.date(2020, 10, 3))],
                                               "test_expression": datetime.date(2020, 10, 3)},
                                              {"Datatype": DecimalType,
                                               "test_pass_data": [("foo", decimal.Decimal(31)),
                                                                  ("bar", decimal.Decimal(32))],
                                               "test_fail_data": [("foo", decimal.Decimal(30)),
                                                                  ("bar", decimal.Decimal(31))],
                                               "test_expression": decimal.Decimal(30)},
                                              {"Datatype": FloatType,
                                               "test_pass_data": [("foo", 31.0), ("bar", 32.0)],
                                               "test_fail_data": [("foo", 30.0), ("bar", 31.0)],
                                               "test_expression": 30.0
                                               },
                                              {"Datatype": BooleanType,
                                               "test_pass_data": [("foo", True), ("bar", True)],
                                               "test_fail_data": [("foo", False), ("bar", True)],
                                              "test_expression": False}
                                              ]
                         )
@pytest.mark.parametrize("check_fn", [pa.Check.not_equal_to, pa.Check.ne])
def test_not_equal_to_check(spark, check_fn, data_dictionary) -> None:
    """Test the Check to see if any value is not in the specified value"""

    check_function(spark, check_fn, data_dictionary['test_pass_data'], data_dictionary['test_fail_data'],
                   data_dictionary['Datatype'], data_dictionary['test_expression'])



@pytest.mark.parametrize("data_dictionary",  [{"Datatype": LongType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 30), ("bar", 31)],
                                               "test_expression": 30},
                                              {"Datatype": IntegerType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 30), ("bar", 31)],
                                               "test_expression": 30
                                               },
                                              {"Datatype": ByteType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 30), ("bar", 31)],
                                               "test_expression": 30
                                               },
                                              {"Datatype": ShortType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 30), ("bar", 31)],
                                               "test_expression": 30
                                               },
                                              {"Datatype": DoubleType,
                                               "test_pass_data": [("foo", 31.0), ("bar", 32.0)],
                                               "test_fail_data": [("foo", 30.0), ("bar", 31.0)],
                                               "test_expression": 30.0},
                                              {"Datatype": TimestampType,
                                               "test_pass_data": [("foo", datetime.datetime(2020, 10, 1, 11, 0)),
                                                                  ("bar",  datetime.datetime(2020, 10, 2,  11, 0))],
                                               "test_fail_data": [("foo",  datetime.datetime(2020, 10, 1,  10, 0)),
                                                                  ("bar",  datetime.datetime(2020, 10, 2,  11, 0))],
                                               "test_expression": datetime.datetime(2020, 10, 1,  10, 0)
                                               },
                                              {"Datatype": DateType,
                                               "test_pass_data": [("foo", datetime.date(2020, 10, 2)),
                                                                  ("bar", datetime.date(2020, 10, 3))],
                                               "test_fail_data": [("foo", datetime.date(2020, 10, 2)),
                                                                  ("bar", datetime.date(2020, 10, 1))],
                                               "test_expression": datetime.date(2020, 10, 1)},
                                              {"Datatype": DecimalType,
                                               "test_pass_data": [("foo", decimal.Decimal(31)),
                                                                  ("bar", decimal.Decimal(32))],
                                               "test_fail_data": [("foo", decimal.Decimal(30)),
                                                                  ("bar", decimal.Decimal(31))],
                                               "test_expression": decimal.Decimal(30)},
                                              {"Datatype": FloatType,
                                               "test_pass_data": [("foo", 31.0), ("bar", 32.0)],
                                               "test_fail_data": [("foo", 30.0), ("bar", 31.0)],
                                               "test_expression": 30.0
                                               }
                                              ]
                         )
@pytest.mark.parametrize("check_fn", [pa.Check.gt, pa.Check.greater_than])
def test_greater_than_check(spark, check_fn, data_dictionary) -> None:
    """Test the Check to see if any value is not in the specified value"""
    check_function(spark, check_fn, data_dictionary['test_pass_data'], data_dictionary['test_fail_data'],
                   data_dictionary['Datatype'], data_dictionary['test_expression'])


@pytest.mark.parametrize("data_dictionary",  [{"Datatype": LongType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 30), ("bar", 31)],
                                               "test_expression": 31},
                                              {"Datatype": IntegerType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 30), ("bar", 31)],
                                               "test_expression": 31
                                               },
                                              {"Datatype": ByteType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 30), ("bar", 31)],
                                               "test_expression": 31
                                               },
                                              {"Datatype": ShortType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 30), ("bar", 31)],
                                               "test_expression": 31
                                               },
                                              {"Datatype": DoubleType,
                                               "test_pass_data": [("foo", 31.0), ("bar", 32.0)],
                                               "test_fail_data": [("foo", 30.0), ("bar", 31.0)],
                                               "test_expression": 31.0},
                                              {"Datatype": TimestampType,
                                               "test_pass_data": [("foo", datetime.datetime(2020, 10, 1, 11, 0)),
                                                                  ("bar",  datetime.datetime(2020, 10, 2,  11, 0))],
                                               "test_fail_data": [("foo",  datetime.datetime(2020, 10, 1,  10, 0)),
                                                                  ("bar",  datetime.datetime(2020, 10, 2,  11, 0))],
                                               "test_expression": datetime.datetime(2020, 10, 1,  11, 0)
                                               },
                                              {"Datatype": DateType,
                                               "test_pass_data": [("foo", datetime.date(2020, 10, 2)),
                                                                  ("bar", datetime.date(2020, 10, 3))],
                                               "test_fail_data": [("foo", datetime.date(2020, 10, 2)),
                                                                  ("bar", datetime.date(2020, 10, 1))],
                                               "test_expression": datetime.date(2020, 10, 2)},
                                              {"Datatype": DecimalType,
                                               "test_pass_data": [("foo", decimal.Decimal(31)),
                                                                  ("bar", decimal.Decimal(32))],
                                               "test_fail_data": [("foo", decimal.Decimal(30)),
                                                                  ("bar", decimal.Decimal(31))],
                                               "test_expression": decimal.Decimal(31)},
                                              {"Datatype": FloatType,
                                               "test_pass_data": [("foo", 31.0), ("bar", 32.0)],
                                               "test_fail_data": [("foo", 30.0), ("bar", 31.0)],
                                               "test_expression": 31.0
                                               }
                                              ]
                         )
@pytest.mark.parametrize("check_fn", [pa.Check.ge, pa.Check.greater_than_or_equal_to])
def test_greater_than_or_equal_to_check(spark, check_fn, data_dictionary) -> None:
    """Test the Check to see if any value is not in the specified value"""
    check_function(spark, check_fn, data_dictionary['test_pass_data'], data_dictionary['test_fail_data'],
                   data_dictionary['Datatype'], data_dictionary['test_expression'])


@pytest.mark.parametrize("data_dictionary",  [{"Datatype": LongType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 33), ("bar", 31)],
                                               "test_expression": 33},
                                              {"Datatype": IntegerType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 33), ("bar", 31)],
                                               "test_expression": 33
                                               },
                                              {"Datatype": ByteType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 33), ("bar", 31)],
                                               "test_expression": 33
                                               },
                                              {"Datatype": ShortType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 33), ("bar", 31)],
                                               "test_expression": 33
                                               },
                                              {"Datatype": DoubleType,
                                               "test_pass_data": [("foo", 31.0), ("bar", 32.0)],
                                               "test_fail_data": [("foo", 33.0), ("bar", 31.0)],
                                               "test_expression": 33.0},
                                              {"Datatype": TimestampType,
                                               "test_pass_data": [("foo", datetime.datetime(2020, 10, 1, 11, 0)),
                                                                  ("bar",  datetime.datetime(2020, 10, 2,  11, 0))],
                                               "test_fail_data": [("foo",  datetime.datetime(2020, 10, 1,  10, 0)),
                                                                  ("bar",  datetime.datetime(2020, 10, 2,  12, 0))],
                                               "test_expression": datetime.datetime(2020, 10, 2,  12, 0)
                                               },
                                              {"Datatype": DateType,
                                               "test_pass_data": [("foo", datetime.date(2020, 10, 2)),
                                                                  ("bar", datetime.date(2020, 10, 1))],
                                               "test_fail_data": [("foo", datetime.date(2020, 10, 2)),
                                                                  ("bar", datetime.date(2020, 10, 3))],
                                               "test_expression": datetime.date(2020, 10, 3)},
                                              {"Datatype": DecimalType,
                                               "test_pass_data": [("foo", decimal.Decimal(31)),
                                                                  ("bar", decimal.Decimal(32))],
                                               "test_fail_data": [("foo", decimal.Decimal(33)),
                                                                  ("bar", decimal.Decimal(31))],
                                               "test_expression": decimal.Decimal(33)},
                                              {"Datatype": FloatType,
                                               "test_pass_data": [("foo", 31.0), ("bar", 32.0)],
                                               "test_fail_data": [("foo", 30.0), ("bar", 33.0)],
                                               "test_expression": 33.0
                                               }
                                              ]
                         )
@pytest.mark.parametrize("check_fn", [pa.Check.lt, pa.Check.less_than])
def test_less_than_check(spark, check_fn, data_dictionary) -> None:
    """Test the Check to see if any value is not in the specified value"""
    check_function(spark, check_fn, data_dictionary['test_pass_data'], data_dictionary['test_fail_data'],
                   data_dictionary['Datatype'], data_dictionary['test_expression'])


@pytest.mark.parametrize("data_dictionary",  [{"Datatype": LongType,
                                               "test_pass_data": [("foo", 31), ("bar", 33)],
                                               "test_fail_data": [("foo", 34), ("bar", 31)],
                                               "test_expression": 33},
                                              {"Datatype": IntegerType,
                                               "test_pass_data": [("foo", 31), ("bar", 33)],
                                               "test_fail_data": [("foo", 34), ("bar", 31)],
                                               "test_expression": 33
                                               },
                                              {"Datatype": ByteType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 33), ("bar", 31)],
                                               "test_expression": 32
                                               },
                                              {"Datatype": ShortType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 33), ("bar", 31)],
                                               "test_expression": 32
                                               },
                                              {"Datatype": DoubleType,
                                               "test_pass_data": [("foo", 31.0), ("bar", 32.0)],
                                               "test_fail_data": [("foo", 33.0), ("bar", 31.0)],
                                               "test_expression": 32.0},
                                              {"Datatype": TimestampType,
                                               "test_pass_data": [("foo", datetime.datetime(2020, 10, 1, 11, 0)),
                                                                  ("bar",  datetime.datetime(2020, 10, 2,  11, 0))],
                                               "test_fail_data": [("foo",  datetime.datetime(2020, 10, 1,  10, 0)),
                                                                  ("bar",  datetime.datetime(2020, 10, 2,  12, 0))],
                                               "test_expression": datetime.datetime(2020, 10, 2,  11, 0)
                                               },
                                              {"Datatype": DateType,
                                               "test_pass_data": [("foo", datetime.date(2020, 10, 2)),
                                                                  ("bar", datetime.date(2020, 10, 1))],
                                               "test_fail_data": [("foo", datetime.date(2020, 10, 2)),
                                                                  ("bar", datetime.date(2020, 10, 3))],
                                               "test_expression": datetime.date(2020, 10, 2)},
                                              {"Datatype": DecimalType,
                                               "test_pass_data": [("foo", decimal.Decimal(31)),
                                                                  ("bar", decimal.Decimal(32))],
                                               "test_fail_data": [("foo", decimal.Decimal(33)),
                                                                  ("bar", decimal.Decimal(31))],
                                               "test_expression": decimal.Decimal(32)},
                                              {"Datatype": FloatType,
                                               "test_pass_data": [("foo", 31.0), ("bar", 32.0)],
                                               "test_fail_data": [("foo", 30.0), ("bar", 33.0)],
                                               "test_expression": 32.0
                                               }
                                              ]
                         )
@pytest.mark.parametrize("check_fn", [pa.Check.le, pa.Check.less_than_or_equal_to])
def test_less_than_equal_to_check(spark, check_fn, data_dictionary) -> None:
    """Test the Check to see if any value is not in the specified value"""
    check_function(spark, check_fn, data_dictionary['test_pass_data'], data_dictionary['test_fail_data'],
                   data_dictionary['Datatype'], data_dictionary['test_expression'])



@pytest.mark.parametrize("data_dictionary",  [{"Datatype": LongType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 30), ("bar", 31)],
                                               "test_expression": [31, 32]},
                                              {"Datatype": IntegerType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 30), ("bar", 31)],
                                               "test_expression": [31, 32]
                                               },
                                              {"Datatype": ByteType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 30), ("bar", 31)],
                                               "test_expression": [31, 32]
                                               },
                                              {"Datatype": ShortType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 30), ("bar", 31)],
                                               "test_expression": [31, 32]
                                               },
                                              {"Datatype": StringType,
                                               "test_pass_data": [("foo", "b"), ("bar", "c")],
                                               "test_fail_data": [("foo", "a"), ("bar", "b")],
                                               "test_expression": ["b", "c"]
                                               },
                                              {"Datatype": DoubleType,
                                               "test_pass_data": [("foo", 31.0), ("bar", 32.0)],
                                               "test_fail_data": [("foo", 30.0), ("bar", 31.0)],
                                               "test_expression": [31.0, 32.0]},
                                              {"Datatype": TimestampType,
                                               "test_pass_data": [("foo", datetime.datetime(2020, 10, 1, 10, 0)),
                                                                  ("bar",  datetime.datetime(2020, 10, 2,  10, 0))],
                                               "test_fail_data": [("foo",  datetime.datetime(2020, 10, 3,  10, 0)),
                                                                  ("bar",  datetime.datetime(2020, 10, 2,  10, 0))],
                                               "test_expression": [datetime.datetime(2020, 10, 1,  10, 0),
                                                                   datetime.datetime(2020, 10, 2,  10, 0)
                                                                   ]
                                               },
                                              {"Datatype": DateType,
                                               "test_pass_data": [("foo", datetime.date(2020, 10, 1)),
                                                                  ("bar", datetime.date(2020, 10, 2))],
                                               "test_fail_data": [("foo", datetime.date(2020, 10, 2)),
                                                                  ("bar", datetime.date(2020, 10, 3))],
                                               "test_expression": [datetime.date(2020, 10, 1),
                                                                   datetime.date(2020, 10, 2)]
                                               },
                                              {"Datatype": DecimalType,
                                               "test_pass_data": [("foo", decimal.Decimal(31)),
                                                                  ("bar", decimal.Decimal(32))],
                                               "test_fail_data": [("foo", decimal.Decimal(30)),
                                                                  ("bar", decimal.Decimal(31))],
                                               "test_expression": [decimal.Decimal(31), decimal.Decimal(32)]},
                                              {"Datatype": FloatType,
                                               "test_pass_data": [("foo", 31.0), ("bar", 32.0)],
                                               "test_fail_data": [("foo", 30.0), ("bar", 31.0)],
                                               "test_expression": [31.0, 32.0]
                                               }
                                              ]
                         )
def test_isin_check(spark, data_dictionary) -> None:
    """Test the Check to see if any value is not in the specified value"""
    check_function(spark, pa.Check.isin, data_dictionary['test_pass_data'], data_dictionary['test_fail_data'],
                   data_dictionary['Datatype'], data_dictionary['test_expression'])



@pytest.mark.parametrize("data_dictionary",  [{"Datatype": LongType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 30), ("bar", 31)],
                                               "test_expression": [30, 33]},
                                              {"Datatype": IntegerType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 30), ("bar", 31)],
                                               "test_expression": [30, 33]
                                               },
                                              {"Datatype": ByteType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 30), ("bar", 31)],
                                               "test_expression": [30, 33]
                                               },
                                              {"Datatype": ShortType,
                                               "test_pass_data": [("foo", 31), ("bar", 32)],
                                               "test_fail_data": [("foo", 30), ("bar", 31)],
                                               "test_expression": [30, 33]
                                               },
                                              {"Datatype": StringType,
                                               "test_pass_data": [("foo", "b"), ("bar", "c")],
                                               "test_fail_data": [("foo", "a"), ("bar", "b")],
                                               "test_expression": ["a", "d"]
                                               },
                                              {"Datatype": DoubleType,
                                               "test_pass_data": [("foo", 31.0), ("bar", 32.0)],
                                               "test_fail_data": [("foo", 30.0), ("bar", 31.0)],
                                               "test_expression": [30.0, 33.0]},
                                              {"Datatype": TimestampType,
                                               "test_pass_data": [("foo", datetime.datetime(2020, 10, 1, 10, 0)),
                                                                  ("bar",  datetime.datetime(2020, 10, 2,  10, 0))],
                                               "test_fail_data": [("foo",  datetime.datetime(2020, 10, 3,  10, 0)),
                                                                  ("bar",  datetime.datetime(2020, 10, 2,  10, 0))],
                                               "test_expression": [datetime.datetime(2020, 10, 3,  10, 0),
                                                                   datetime.datetime(2020, 10, 4,  10, 0)
                                                                   ]
                                               },
                                              {"Datatype": DateType,
                                               "test_pass_data": [("foo", datetime.date(2020, 10, 1)),
                                                                  ("bar", datetime.date(2020, 10, 2))],
                                               "test_fail_data": [("foo", datetime.date(2020, 10, 2)),
                                                                  ("bar", datetime.date(2020, 10, 3))],
                                               "test_expression": [datetime.date(2020, 10, 3),
                                                                   datetime.date(2020, 10, 4)]
                                               },
                                              {"Datatype": DecimalType,
                                               "test_pass_data": [("foo", decimal.Decimal(31)),
                                                                  ("bar", decimal.Decimal(32))],
                                               "test_fail_data": [("foo", decimal.Decimal(30)),
                                                                  ("bar", decimal.Decimal(31))],
                                               "test_expression": [decimal.Decimal(33), decimal.Decimal(30)]},
                                              {"Datatype": FloatType,
                                               "test_pass_data": [("foo", 31.0), ("bar", 32.0)],
                                               "test_fail_data": [("foo", 30.0), ("bar", 31.0)],
                                               "test_expression": [30.0, 33.0]
                                               }
                                              ]
                         )
def test_notin_check(spark, data_dictionary) -> None:
    """Test the Check to see if any value is not in the specified value"""
    check_function(spark, pa.Check.notin, data_dictionary['test_pass_data'], data_dictionary['test_fail_data'],
                   data_dictionary['Datatype'], data_dictionary['test_expression'])


def test_str_startswith_check(spark) -> None:
    """Test the Check to see if any value is not in the specified value"""
    check_func = pa.Check.str_startswith
    check_value = "B"

    pass_data = [("Bal", "Bread"), ("Bal", "Butter")]
    fail_data = [("Bal", "Test"), ("Bal", "Butter")]
    check_function(spark, check_func, pass_data, fail_data,
                   StringType, check_value)


def test_str_endswith_check(spark) -> None:
    """Test the Check to see if any value is not in the specified value"""
    check_func = pa.Check.str_endswith
    check_value = "d"

    pass_data = [("Bal", "Bread"), ("Bal", "Bad")]
    fail_data = [("Bal", "Test"), ("Bal", "Bad")]
    check_function(spark, check_func, pass_data, fail_data,
                   StringType, check_value)


def test_str_contains_check(spark) -> None:
    """Test the Check to see if any value is not in the specified value"""
    check_func = pa.Check.str_contains
    check_value = "Ba"

    pass_data = [("Bal", "Bat!"), ("Bal", "Bat78")]
    fail_data = [("Bal", "Cs"), ("Bal", "Jam!")]
    check_function(spark, check_func, pass_data, fail_data,
                   StringType, check_value)



@pytest.mark.parametrize("data_dictionary",  [{"Datatype": LongType,
                                               "test_pass_data": [("foo", 31), ("bar", 33)],
                                               "test_fail_data": [("foo", 35), ("bar", 31)],
                                               },
                                              {"Datatype": IntegerType,
                                               "test_pass_data": [("foo", 31), ("bar", 33)],
                                               "test_fail_data": [("foo", 35), ("bar", 31)],
                                               },
                                              {"Datatype": ByteType,
                                               "test_pass_data": [("foo", 31), ("bar", 33)],
                                               "test_fail_data": [("foo", 35), ("bar", 31)],
                                               },
                                              {"Datatype": ShortType,
                                               "test_pass_data": [("foo", 31), ("bar", 33)],
                                               "test_fail_data": [("foo", 35), ("bar", 31)],
                                               },
                                              {"Datatype": DoubleType,
                                               "test_pass_data": [("foo", 31.0), ("bar", 33.0)],
                                               "test_fail_data": [("foo", 35.0), ("bar", 31.0)],
                                               },
                                              {"Datatype": TimestampType,
                                               "test_pass_data": [("foo", datetime.datetime(2020, 10, 1, 11, 0)),
                                                                  ("bar",  datetime.datetime(2020, 10, 2,  11, 0))],
                                               "test_fail_data": [("foo",  datetime.datetime(2020, 10, 1,  10, 0)),
                                                                  ("bar",  datetime.datetime(2020, 10, 5,  12, 0))],

                                               },
                                              {"Datatype": DateType,
                                               "test_pass_data": [("foo", datetime.date(2020, 10, 2)),
                                                                  ("bar", datetime.date(2020, 10, 1))],
                                               "test_fail_data": [("foo", datetime.date(2020, 10, 2)),
                                                                  ("bar", datetime.date(2020, 10, 5))],
                                               },
                                              {"Datatype": DecimalType,
                                               "test_pass_data": [("foo", decimal.Decimal(31)),
                                                                  ("bar", decimal.Decimal(33))],
                                               "test_fail_data": [("foo", decimal.Decimal(34)),
                                                                  ("bar", decimal.Decimal(31))],
                                              },
                                              {"Datatype": FloatType,
                                               "test_pass_data": [("foo", 31.0), ("bar", 32.0)],
                                               "test_fail_data": [("foo", 30.0), ("bar", 35.0)],
                                               }
                                              ]
                         )
def test_in_range_check(spark, data_dictionary) -> None:
    """Test the Check to see if any value is not in the specified value"""
    value_dict = [value[1] for value in data_dictionary["test_pass_data"]]
    min_val = min(value_dict)
    max_val = max(value_dict)
    if isinstance(min_val, datetime.datetime):
        add_value = datetime.timedelta(1)
    elif isinstance(min_val, datetime.date):
        add_value = datetime.timedelta(1)
    else:
        add_value = 1
    # dont include of min and max
    check_function(spark, pa.Check.in_range, data_dictionary['test_pass_data'], data_dictionary['test_fail_data'],
                   data_dictionary['Datatype'], (min_val-add_value, max_val+add_value, False, False))
    # include only min value
    check_function(spark, pa.Check.in_range, data_dictionary['test_pass_data'], data_dictionary['test_fail_data'],
                   data_dictionary['Datatype'], (min_val, max_val+add_value, True, False))
    # include only max value
    check_function(spark, pa.Check.in_range, data_dictionary['test_pass_data'], data_dictionary['test_fail_data'],
                   data_dictionary['Datatype'], (min_val-add_value, max_val, False, True))
    # include both min and max
    check_function(spark, pa.Check.in_range, data_dictionary['test_pass_data'], data_dictionary['test_fail_data'],
                   data_dictionary['Datatype'], (min_val, max_val, True, True))

