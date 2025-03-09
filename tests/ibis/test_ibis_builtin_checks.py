"""Unit tests for Ibis checks."""

import datetime
import decimal
from operator import methodcaller

import ibis
import ibis.expr.datatypes as dt
import pytest

import pandera.ibis as pa
from pandera.errors import SchemaError
from pandera.ibis import Column, DataFrameSchema


class BaseClass:
    """This is the base class for the all the test cases class"""

    def __int__(self, params=None):
        pass

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

        return data_dict  # pylint:disable=possibly-used-before-assignment

    @staticmethod
    def check_function(
        check_fn,
        pass_case_data,
        fail_case_data,
        data_types,
        function_args,
        fail_on_init=False,
        init_exception_cls=None,
    ):
        """
        This function does performs the actual validation
        """
        if fail_on_init:
            with pytest.raises(init_exception_cls):
                check_fn(*function_args)
            return

        schema = DataFrameSchema(
            {
                "product": Column(dt.String),
                "code": (
                    Column(data_types, check_fn(*function_args))
                    if isinstance(function_args, tuple)
                    else Column(data_types, check_fn(function_args))
                ),
            }
        )

        ibis_schema = ibis.schema({"product": "string", "code": data_types})

        # check that check on pass case data passes
        t = ibis.memtable(pass_case_data, schema=ibis_schema)
        schema.validate(t)

        with pytest.raises(SchemaError):
            t = ibis.memtable(fail_case_data, schema=ibis_schema)
            schema.validate(t)


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

    sample_duration_data = {
        "test_pass_data": [
            ("foo", datetime.timedelta(100, 10, 1)),
            ("bar", datetime.timedelta(100, 10, 1)),
        ],
        "test_fail_data": [
            ("foo", datetime.timedelta(100, 10, 1)),
            ("bar", datetime.timedelta(100, 11, 1)),
        ],
        "test_expression": datetime.timedelta(100, 10, 1),
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
        """Generate the params which will be used to test this function. All the acceptable
        data types would be tested"""
        return {
            "test_equal_to_check": [
                {"datatype": dt.UInt8, "data": self.sample_numeric_data},
                {"datatype": dt.UInt16, "data": self.sample_numeric_data},
                {"datatype": dt.UInt32, "data": self.sample_numeric_data},
                {"datatype": dt.UInt64, "data": self.sample_numeric_data},
                {"datatype": dt.Int8, "data": self.sample_numeric_data},
                {"datatype": dt.Int16, "data": self.sample_numeric_data},
                {"datatype": dt.Int32, "data": self.sample_numeric_data},
                {"datatype": dt.Int64, "data": self.sample_numeric_data},
                {"datatype": dt.String, "data": self.sample_string_data},
                {
                    "datatype": dt.Float32,
                    "data": self.convert_data(
                        self.sample_numeric_data, "float32"
                    ),
                },
                {
                    "datatype": dt.Float64,
                    "data": self.convert_data(
                        self.sample_numeric_data, "float64"
                    ),
                },
                {
                    "datatype": dt.Date,
                    "data": self.convert_data(
                        self.sample_datetime_data, "date"
                    ),
                },
                {
                    "datatype": dt.Timestamp.from_unit("us"),
                    "data": self.sample_datetime_data,
                },
                {
                    "datatype": dt.Time,
                    "data": self.convert_data(
                        self.sample_datetime_data, "time"
                    ),
                },
                {
                    "datatype": dt.Interval(unit="us"),
                    "data": self.sample_duration_data,
                },
            ]
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


class TestNotEqualToCheck(BaseClass):
    """This class is used to test the not equal to check"""

    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 32)],
        "test_fail_data": [("foo", 30), ("bar", 31)],
        "test_expression": 30,
    }

    sample_datetime_data = {
        "test_pass_data": [
            ("foo", datetime.datetime(2020, 10, 1, 11, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 11, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.datetime(2020, 10, 3, 10, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 11, 0)),
        ],
        "test_expression": datetime.datetime(2020, 10, 3, 10, 0),
    }

    sample_string_data = {
        "test_pass_data": [("foo", "b"), ("bar", "c")],
        "test_fail_data": [("foo", "a"), ("bar", "a")],
        "test_expression": "a",
    }

    sample_duration_data = {
        "test_pass_data": [
            (
                "foo",
                datetime.timedelta(
                    100,
                    11,
                    1,
                ),
            ),
            (
                "bar",
                datetime.timedelta(
                    100,
                    11,
                    1,
                ),
            ),
        ],
        "test_fail_data": [
            (
                "foo",
                datetime.timedelta(
                    100,
                    10,
                    1,
                ),
            ),
            (
                "bar",
                datetime.timedelta(
                    100,
                    10,
                    1,
                ),
            ),
        ],
        "test_expression": datetime.timedelta(
            100,
            10,
            1,
        ),
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
        """Generate the params which will be used to test this function. All the acceptable
        data types would be tested"""
        return {
            "test_not_equal_to_check": [
                {"datatype": dt.UInt8, "data": self.sample_numeric_data},
                {"datatype": dt.UInt16, "data": self.sample_numeric_data},
                {"datatype": dt.UInt32, "data": self.sample_numeric_data},
                {"datatype": dt.UInt64, "data": self.sample_numeric_data},
                {"datatype": dt.Int8, "data": self.sample_numeric_data},
                {"datatype": dt.Int16, "data": self.sample_numeric_data},
                {"datatype": dt.Int32, "data": self.sample_numeric_data},
                {"datatype": dt.Int64, "data": self.sample_numeric_data},
                {"datatype": dt.String, "data": self.sample_string_data},
                {
                    "datatype": dt.Float32,
                    "data": self.convert_data(
                        self.sample_numeric_data, "float32"
                    ),
                },
                {
                    "datatype": dt.Float64,
                    "data": self.convert_data(
                        self.sample_numeric_data, "float64"
                    ),
                },
                {
                    "datatype": dt.Date,
                    "data": self.convert_data(
                        self.sample_datetime_data, "date"
                    ),
                },
                {
                    "datatype": dt.Timestamp.from_unit("us"),
                    "data": self.sample_datetime_data,
                },
                {
                    "datatype": dt.Time,
                    "data": self.convert_data(
                        self.sample_datetime_data, "time"
                    ),
                },
                {
                    "datatype": dt.Interval(unit="us"),
                    "data": self.sample_duration_data,
                },
            ]
        }

    @pytest.mark.parametrize("check_fn", [pa.Check.not_equal_to, pa.Check.ne])
    def test_not_equal_to_check(self, check_fn, datatype, data) -> None:
        """Test the Check to see if all the values are equal to defined value"""
        self.check_function(
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            data["test_expression"],
        )
