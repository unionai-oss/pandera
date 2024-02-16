"""Unit tests for pyspark container."""
# pylint:disable=abstract-method
import datetime
import decimal
from operator import methodcaller
import polars as pl


from polars.datatypes import (
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Date,
    Time,
    Duration,
    Datetime,
    Object,
    Unknown,
    Binary,
    Decimal,
    List,
    Struct,
    Boolean,
    Categorical,
    Utf8,
)
import pytest
from pandera.errors import SchemaError


import pandera.polars as pa
from pandera.polars import DataFrameSchema, Column


class BaseClass:
    """This is the base class for the all the test cases class"""

    def __int__(self, params=None):
        pass

    sample_string_data = {
        "test_pass_data": [("foo", "b"), ("bar", "c")],
        "test_expression": "a",
    }

    sample_array_data = {
        "test_pass_data": [("foo", ["a"]), ("bar", ["a"])],
        "test_expression": [["a"]],
    }

    sample_map_data = {
        "test_pass_data": [("foo", {"key": "val"}), ("bar", {"key": "val"})],
        "test_expression": {"foo": "val"},
    }

    sample_boolean_data = {
        "test_pass_data": [("foo", True), ("bar", True)],
        "test_expression": False,
    }

    def pytest_generate(self, metafunc):
        """This function passes the parameter for each function based on parameter form get_data_param function"""
        raise NotImplementedError

    @staticmethod
    def convert_value(sample_data, conversion_datatype):
        """
        Convert the sample data to other formats excluding dates and does not
        support complex datatypes such as array and map as of now
        """

        data_dict = {}
        for key, value in sample_data.items():
            if key == "test_expression":
                if not isinstance(value, list):
                    data_dict[key] = conversion_datatype(value)
                else:
                    data_dict[key] = [conversion_datatype(i) for i in value]

            else:
                if not isinstance(value[0][1], list):
                    data_dict[key] = [
                        (i[0], conversion_datatype(i[1])) for i in value
                    ]
                else:
                    final_val = []
                    for row in value:
                        data_val = []
                        for column in row[1]:
                            data_val.append(conversion_datatype(column))
                        final_val.append((row[0], data_val))
                    data_dict[key] = final_val
        return data_dict

    @staticmethod
    def convert_data(sample_data, convert_type):
        """
        Convert the numeric data to required format
        """
        if convert_type in ("float32", "float64"):
            data_dict = BaseClass.convert_value(sample_data, float)

        if convert_type == "decimal":
            data_dict = BaseClass.convert_value(sample_data, decimal.Decimal)

        if convert_type == "date":
            data_dict = BaseClass.convert_value(
                sample_data, methodcaller("date")
            )

        if convert_type == "time":
            data_dict = BaseClass.convert_value(
                sample_data, methodcaller("time")
            )

        if convert_type == "binary":
            data_dict = BaseClass.convert_value(
                sample_data, methodcaller("encode")
            )

        return data_dict

    @staticmethod
    def check_function(
        check_fn,
        pass_case_data,
        fail_case_data,
        data_types,
        function_args,
        skip_fail_case=False,
    ):
        """
        This function does performs the actual validation
        """

        schema = DataFrameSchema(
            {
                "product": Column(Utf8),
                "code": Column(data_types, check_fn(function_args)),
            }
        )

        polars_schema = {"product": Utf8, "code": data_types}

        # check that check on pass case data passes
        df = pl.LazyFrame(pass_case_data, orient="row", schema=polars_schema)
        schema.validate(df)

        if not skip_fail_case:
            with pytest.raises(SchemaError):
                df = pl.LazyFrame(fail_case_data, orient="row", schema=polars_schema)
                schema.validate(df)


class TestEqualToCheck(BaseClass):
    """This class is used to test the equal to check"""

    sample_numeric_data = {
        "test_pass_data": [("foo", 30), ("bar", 30)],
        "test_fail_data": [("foo", 30), ("bar", 31)],
        "test_expression": 30,
    }

    sample_datetime_data = {
        "test_pass_data": [
            ("foo", datetime.datetime(2020, 10, 1, 10, 0)),
            ("bar", datetime.datetime(2020, 10, 1, 10, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.datetime(2020, 10, 2, 11, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 11, 0)),
        ],
        "test_expression": datetime.datetime(2020, 10, 1, 10, 0),
    }

    sample_string_data = {
        "test_pass_data": [("foo", "a"), ("bar", "a")],
        "test_fail_data": [("foo", "a"), ("bar", "b")],
        "test_expression": "a",
    }

    sample_boolean_data = {
        "test_pass_data": [("foo", True), ("bar", True)],
        "test_fail_data": [("foo", False), ("bar", False)],
        "test_expression": True,
    }

    sample_array_data = {
        "test_pass_data": [("foo", ["a"]), ("bar", ["a"])],
        "test_fail_data": [("foo", ["a"]), ("bar", ["b"])],
        "test_expression": ["a"],
    }

    sample_duration_data = {
        "test_pass_data": [
            ("foo", datetime.timedelta(2020, 10, 1, 10, 0)),
            ("bar", datetime.timedelta(2020, 10, 1, 10, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.timedelta(2020, 10, 2, 11, 0)),
            ("bar", datetime.timedelta(2020, 10, 2, 11, 0)),
        ],
        "test_expression": datetime.timedelta(2020, 10, 1, 10, 0),
    }

    def pytest_generate_tests(self, metafunc):
        """This function passes the parameter for each function based on parameter form get_data_param function"""
        # called once per each test function
        funcarglist = self.get_data_param()[metafunc.function.__name__]
        argnames = sorted(funcarglist[0])
        metafunc.parametrize(
            argnames,
            [
                [funcargs[name] for name in argnames]
                for funcargs in funcarglist
            ],
        )

    def get_data_param(self):
        """Generate the params which will be used to test this function. All the accpetable
        data types would be tested"""
        return {
            "test_equal_to_check": [
                {"datatype": UInt8, "data": self.sample_numeric_data},
                {"datatype": UInt16, "data": self.sample_numeric_data},
                {"datatype": UInt32, "data": self.sample_numeric_data},
                {"datatype": UInt64, "data": self.sample_numeric_data},
                {"datatype": Int8, "data": self.sample_numeric_data},
                {"datatype": Int16, "data": self.sample_numeric_data},
                {"datatype": Int32, "data": self.sample_numeric_data},
                {"datatype": Int64, "data": self.sample_numeric_data},
                {"datatype": Utf8, "data": self.sample_string_data},
                {
                    "datatype": Binary,
                    "data": self.convert_data(
                        self.sample_string_data, "binary"
                    ),
                },
                {"datatype": Categorical(ordering="physical"), "data": self.sample_string_data},
                {
                    "datatype": Float32,
                    "data": self.convert_data(
                        self.sample_numeric_data, "float32"
                    ),
                },
                {
                    "datatype": Float64,
                    "data": self.convert_data(
                        self.sample_numeric_data, "float64"
                    ),
                },
                {
                    "datatype": Date,
                    "data": self.convert_data(
                        self.sample_datetime_data, "date"
                    ),
                },
                {
                    "datatype": Datetime(time_unit="us"),
                    "data": self.sample_datetime_data,
                },
                {
                    "datatype": Time,
                    "data": self.convert_data(
                        self.sample_datetime_data, "time"
                    ),
                },
                {
                    "datatype": Duration(time_unit="us"),
                    "data": self.sample_duration_data,
                },
                {"datatype": Boolean, "data": self.sample_boolean_data},
                {
                    "datatype": List(Utf8),
                    "data": self.sample_array_data,
                },
            ],
            "test_failed_unaccepted_datatypes": [
                {
                    "datatype": Decimal,
                    "data": self.convert_data(
                        self.sample_numeric_data, "decimal"
                    ),
                },
                {
                    "datatype": Object,
                    "data": self.sample_string_data,
                },
                {
                    "datatype": Unknown,
                    "data": self.sample_string_data,
                },
                {
                    "datatype": Struct({"key": pl.Utf8}),
                    "data": self.sample_map_data,
                },
            ],
        }

    @pytest.mark.parametrize("check_fn", [pa.Check.equal_to, pa.Check.eq])
    def test_equal_to_check(self, check_fn, datatype, data) -> None:
        """Test the Check to see if all the values are equal to defined value"""
        self.check_function(
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            data["test_expression"],
        )
