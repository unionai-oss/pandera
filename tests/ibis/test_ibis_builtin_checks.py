"""Unit tests for Ibis checks."""

import datetime
import decimal
import re
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

        return data_dict

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

    sample_boolean_data = {
        "test_pass_data": [("foo", True), ("bar", True)],
        "test_fail_data": [("foo", False), ("bar", False)],
        "test_expression": True,
    }

    def pytest_generate_tests(self, metafunc):
        """This function passes the parameter for each function based on parameter from get_data_param function"""
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
                    "datatype": dt.Binary,
                    "data": self.convert_data(
                        self.sample_string_data, "binary"
                    ),
                },
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
                {"datatype": dt.Boolean, "data": self.sample_boolean_data},
            ]
        }

    @pytest.mark.parametrize("check_fn", [pa.Check.equal_to, pa.Check.eq])
    def test_equal_to_check(self, check_fn, datatype, data) -> None:
        """Test the Check to see if all the values are equal to the defined value"""
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

    sample_boolean_data = {
        "test_pass_data": [("foo", True), ("bar", True)],
        "test_fail_data": [("foo", False), ("bar", True)],
        "test_expression": False,
    }

    def pytest_generate_tests(self, metafunc):
        """This function passes the parameter for each function based on parameter from get_data_param function"""
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
                {"datatype": dt.Boolean, "data": self.sample_boolean_data},
            ]
        }

    @pytest.mark.parametrize("check_fn", [pa.Check.not_equal_to, pa.Check.ne])
    def test_not_equal_to_check(self, check_fn, datatype, data) -> None:
        """Test the Check to see if all the values are equal to the defined value"""
        self.check_function(
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            data["test_expression"],
        )


class TestGreaterThanCheck(BaseClass):
    """This class is used to test the greater than check"""

    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 32)],
        "test_fail_data": [("foo", 30), ("bar", 31)],
        "test_expression": 30,
    }

    sample_datetime_data = {
        "test_pass_data": [
            ("foo", datetime.datetime(2020, 10, 2, 11, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 11, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.datetime(2020, 10, 1, 10, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 11, 0)),
        ],
        "test_expression": datetime.datetime(2020, 10, 1, 10, 0),
    }

    sample_duration_data = {
        "test_pass_data": [
            ("foo", datetime.timedelta(100, 11, 1)),
            ("bar", datetime.timedelta(100, 12, 1)),
        ],
        "test_fail_data": [
            ("foo", datetime.timedelta(100, 10, 1)),
            ("bar", datetime.timedelta(100, 11, 1)),
        ],
        "test_expression": datetime.timedelta(100, 10, 1),
    }

    def pytest_generate_tests(self, metafunc):
        """This function passes the parameter for each function based on parameter from get_data_param function"""
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
            "test_greater_than_check": [
                {"datatype": dt.UInt8, "data": self.sample_numeric_data},
                {"datatype": dt.UInt16, "data": self.sample_numeric_data},
                {"datatype": dt.UInt32, "data": self.sample_numeric_data},
                {"datatype": dt.UInt64, "data": self.sample_numeric_data},
                {"datatype": dt.Int8, "data": self.sample_numeric_data},
                {"datatype": dt.Int16, "data": self.sample_numeric_data},
                {"datatype": dt.Int32, "data": self.sample_numeric_data},
                {"datatype": dt.Int64, "data": self.sample_numeric_data},
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

    @pytest.mark.parametrize("check_fn", [pa.Check.greater_than, pa.Check.gt])
    def test_greater_than_check(self, check_fn, datatype, data) -> None:
        """Test the Check to see if all the values are equal to the defined value"""
        self.check_function(
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            data["test_expression"],
        )


class TestGreaterThanEqualToCheck(BaseClass):
    """This class is used to test the greater than equal to check"""

    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 32)],
        "test_fail_data": [("foo", 30), ("bar", 31)],
        "test_expression": 31,
    }

    sample_datetime_data = {
        "test_pass_data": [
            ("foo", datetime.datetime(2020, 10, 1, 11, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 11, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.datetime(2020, 10, 1, 11, 0)),
            ("bar", datetime.datetime(2020, 9, 1, 10, 0)),
        ],
        "test_expression": datetime.datetime(2020, 10, 1, 11, 0),
    }

    sample_duration_data = {
        "test_pass_data": [
            ("foo", datetime.timedelta(100, 10, 1)),
            ("bar", datetime.timedelta(100, 11, 1)),
        ],
        "test_fail_data": [
            ("foo", datetime.timedelta(100, 11, 1)),
            ("bar", datetime.timedelta(100, 9, 1)),
        ],
        "test_expression": datetime.timedelta(100, 10, 1),
    }

    def pytest_generate_tests(self, metafunc):
        """This function passes the parameter for each function based on parameter from get_data_param function"""
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
            "test_greater_than_or_equal_to_check": [
                {"datatype": dt.UInt8, "data": self.sample_numeric_data},
                {"datatype": dt.UInt16, "data": self.sample_numeric_data},
                {"datatype": dt.UInt32, "data": self.sample_numeric_data},
                {"datatype": dt.UInt64, "data": self.sample_numeric_data},
                {"datatype": dt.Int8, "data": self.sample_numeric_data},
                {"datatype": dt.Int16, "data": self.sample_numeric_data},
                {"datatype": dt.Int32, "data": self.sample_numeric_data},
                {"datatype": dt.Int64, "data": self.sample_numeric_data},
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

    @pytest.mark.parametrize(
        "check_fn", [pa.Check.greater_than_or_equal_to, pa.Check.ge]
    )
    def test_greater_than_or_equal_to_check(
        self, check_fn, datatype, data
    ) -> None:
        """Test the Check to see if all the values are equal to the defined value"""
        self.check_function(
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            data["test_expression"],
        )


class TestLessThanCheck(BaseClass):
    """This class is used to test the less than check"""

    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 32)],
        "test_fail_data": [("foo", 34), ("bar", 33)],
        "test_expression": 33,
    }

    sample_datetime_data = {
        "test_pass_data": [
            ("foo", datetime.datetime(2020, 10, 1, 10, 0)),
            ("bar", datetime.datetime(2020, 10, 1, 10, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.datetime(2020, 11, 1, 11, 0)),
            ("bar", datetime.datetime(2020, 12, 1, 12, 0)),
        ],
        "test_expression": datetime.datetime(2020, 11, 1, 11, 0),
    }

    sample_duration_data = {
        "test_pass_data": [
            ("foo", datetime.timedelta(100, 10, 1)),
            ("bar", datetime.timedelta(100, 10, 1)),
        ],
        "test_fail_data": [
            ("foo", datetime.timedelta(100, 15, 1)),
            ("bar", datetime.timedelta(100, 10, 1)),
        ],
        "test_expression": datetime.timedelta(100, 15, 1),
    }

    def pytest_generate_tests(self, metafunc):
        """This function passes the parameter for each function based on parameter from get_data_param function"""
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
            "test_less_than_check": [
                {"datatype": dt.UInt8, "data": self.sample_numeric_data},
                {"datatype": dt.UInt16, "data": self.sample_numeric_data},
                {"datatype": dt.UInt32, "data": self.sample_numeric_data},
                {"datatype": dt.UInt64, "data": self.sample_numeric_data},
                {"datatype": dt.Int8, "data": self.sample_numeric_data},
                {"datatype": dt.Int16, "data": self.sample_numeric_data},
                {"datatype": dt.Int32, "data": self.sample_numeric_data},
                {"datatype": dt.Int64, "data": self.sample_numeric_data},
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

    @pytest.mark.parametrize("check_fn", [pa.Check.less_than, pa.Check.lt])
    def test_less_than_check(self, check_fn, datatype, data) -> None:
        """Test the Check to see if all the values are equal to the defined value"""
        self.check_function(
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            data["test_expression"],
        )


class TestLessThanEqualToCheck(BaseClass):
    """This class is used to test the less than equal to check"""

    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 33)],
        "test_fail_data": [("foo", 34), ("bar", 31)],
        "test_expression": 33,
    }

    sample_datetime_data = {
        "test_pass_data": [
            ("foo", datetime.datetime(2020, 11, 1, 11, 0)),
            ("bar", datetime.datetime(2020, 10, 1, 10, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.datetime(2020, 11, 1, 11, 0)),
            ("bar", datetime.datetime(2020, 12, 1, 12, 0)),
        ],
        "test_expression": datetime.datetime(2020, 11, 1, 11, 0),
    }

    sample_duration_data = {
        "test_pass_data": [
            ("foo", datetime.timedelta(100, 15, 1)),
            ("bar", datetime.timedelta(100, 10, 1)),
        ],
        "test_fail_data": [
            ("foo", datetime.timedelta(100, 16, 1)),
            ("bar", datetime.timedelta(100, 16, 1)),
        ],
        "test_expression": datetime.timedelta(100, 15, 1),
    }

    def pytest_generate_tests(self, metafunc):
        """This function passes the parameter for each function based on parameter from get_data_param function"""
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
            "test_less_than_or_equal_to_check": [
                {"datatype": dt.UInt8, "data": self.sample_numeric_data},
                {"datatype": dt.UInt16, "data": self.sample_numeric_data},
                {"datatype": dt.UInt32, "data": self.sample_numeric_data},
                {"datatype": dt.UInt64, "data": self.sample_numeric_data},
                {"datatype": dt.Int8, "data": self.sample_numeric_data},
                {"datatype": dt.Int16, "data": self.sample_numeric_data},
                {"datatype": dt.Int32, "data": self.sample_numeric_data},
                {"datatype": dt.Int64, "data": self.sample_numeric_data},
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

    @pytest.mark.parametrize(
        "check_fn", [pa.Check.less_than_or_equal_to, pa.Check.le]
    )
    def test_less_than_or_equal_to_check(
        self, check_fn, datatype, data
    ) -> None:
        """Test the Check to see if all the values are equal to the defined value"""
        self.check_function(
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            data["test_expression"],
        )


class TestInRangeCheck(BaseClass):
    """This class is used to test the value in range check"""

    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 33)],
        "test_fail_data": [("foo", 35), ("bar", 31)],
    }

    sample_datetime_data = {
        "test_pass_data": [
            ("foo", datetime.datetime(2020, 10, 1, 11, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 11, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.datetime(2020, 10, 1, 10, 0)),
            ("bar", datetime.datetime(2020, 10, 5, 12, 0)),
        ],
    }

    sample_duration_data = {
        "test_pass_data": [
            ("foo", datetime.timedelta(101, 20, 1)),
            ("bar", datetime.timedelta(103, 20, 1)),
        ],
        "test_fail_data": [
            ("foo", datetime.timedelta(105, 15, 1)),
            ("bar", datetime.timedelta(101, 20, 1)),
        ],
    }

    sample_boolean_data = {
        "test_pass_data": [("foo", True), ("bar", False)],
        "test_expression": [False],
    }

    def pytest_generate_tests(self, metafunc):
        """This function passes the parameter for each function based on parameter from get_data_param function"""
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

    def create_min_max(self, data_dictionary):
        """This function create the min and max value from the data dictionary to be used for in range test"""
        value_dict = [value[1] for value in data_dictionary["test_pass_data"]]
        min_val = min(value_dict)
        max_val = max(value_dict)
        if isinstance(
            min_val, (datetime.datetime, datetime.date, datetime.timedelta)
        ):
            add_value = datetime.timedelta(1)
        elif isinstance(min_val, datetime.time):
            add_value = 1
        else:
            add_value = 1
        return min_val, max_val, add_value

    def get_data_param(self):
        """Generate the params which will be used to test this function. All the acceptable
        data types would be tested"""
        param_vals = [
            {"datatype": dt.UInt8, "data": self.sample_numeric_data},
            {"datatype": dt.UInt16, "data": self.sample_numeric_data},
            {"datatype": dt.UInt32, "data": self.sample_numeric_data},
            {"datatype": dt.UInt64, "data": self.sample_numeric_data},
            {"datatype": dt.Int8, "data": self.sample_numeric_data},
            {"datatype": dt.Int16, "data": self.sample_numeric_data},
            {"datatype": dt.Int32, "data": self.sample_numeric_data},
            {"datatype": dt.Int64, "data": self.sample_numeric_data},
            {
                "datatype": dt.Float32,
                "data": self.convert_data(self.sample_numeric_data, "float32"),
            },
            {
                "datatype": dt.Float64,
                "data": self.convert_data(self.sample_numeric_data, "float64"),
            },
            {
                "datatype": dt.Date,
                "data": self.convert_data(self.sample_datetime_data, "date"),
            },
            {
                "datatype": dt.Timestamp.from_unit("us"),
                "data": self.sample_datetime_data,
            },
            {
                "datatype": dt.Time,
                "data": self.convert_data(self.sample_datetime_data, "time"),
            },
            {
                "datatype": dt.Interval(unit="us"),
                "data": self.sample_duration_data,
            },
        ]

        return {
            "test_inrange_exclude_min_max_check": param_vals,
            "test_inrange_exclude_min_only_check": param_vals,
            "test_inrange_exclude_max_only_check": param_vals,
            "test_inrange_include_min_max_check": param_vals,
        }

    def safe_add(self, val1, val2):
        """It's not possible to add to datetime.time object, so wrapping +/- operations to handle this case"""
        if isinstance(val1, datetime.time):
            return datetime.time(val1.hour + val2)
        else:
            return val1 + val2

    def safe_subtract(self, val1, val2):
        """It's not possible to subtract from datetime.time object, so wrapping +/- operations to handle this case"""
        if isinstance(val1, datetime.time):
            return datetime.time(val1.hour - val2)
        else:
            return val1 - val2

    @pytest.mark.parametrize("check_fn", [pa.Check.in_range, pa.Check.between])
    def test_inrange_exclude_min_max_check(
        self, check_fn, datatype, data
    ) -> None:
        """Test the Check to see if all the values are equal to the defined value"""
        min_val, max_val, add_value = self.create_min_max(data)
        self.check_function(
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            (
                self.safe_subtract(min_val, add_value),
                self.safe_add(max_val, add_value),
                False,
                False,
            ),
        )

    @pytest.mark.parametrize("check_fn", [pa.Check.in_range, pa.Check.between])
    def test_inrange_exclude_min_only_check(
        self, check_fn, datatype, data
    ) -> None:
        """Test the Check to see if all the values are equal to the defined value"""
        min_val, max_val, add_value = self.create_min_max(data)
        self.check_function(
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            (min_val, self.safe_add(max_val, add_value), True, False),
        )

    @pytest.mark.parametrize("check_fn", [pa.Check.in_range, pa.Check.between])
    def test_inrange_exclude_max_only_check(
        self, check_fn, datatype, data
    ) -> None:
        """Test the Check to see if all the values are equal to the defined value"""
        min_val, max_val, add_value = self.create_min_max(data)
        self.check_function(
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            (self.safe_subtract(min_val, add_value), max_val, False, True),
        )

    @pytest.mark.parametrize("check_fn", [pa.Check.in_range, pa.Check.between])
    def test_inrange_include_min_max_check(
        self, check_fn, datatype, data
    ) -> None:
        """Test the Check to see if all the values are equal to the defined value"""
        min_val, max_val, _ = self.create_min_max(data)
        self.check_function(
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            (min_val, max_val, True, True),
        )


class TestIsInCheck(BaseClass):
    """This class is used to test the isin check"""

    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 32)],
        "test_fail_data": [("foo", 30), ("bar", 31)],
        "test_expression": [31, 32],
    }

    sample_datetime_data = {
        "test_pass_data": [
            ("foo", datetime.datetime(2020, 10, 1, 10, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 10, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.datetime(2020, 10, 3, 10, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 11, 0)),
        ],
        "test_expression": [
            datetime.datetime(2020, 10, 1, 10, 0),
            datetime.datetime(2020, 10, 2, 10, 0),
        ],
    }

    sample_string_data = {
        "test_pass_data": [("foo", "b"), ("bar", "c")],
        "test_fail_data": [("foo", "a"), ("bar", "b")],
        "test_expression": ["b", "c"],
    }

    sample_duration_data = {
        "test_pass_data": [
            ("foo", datetime.timedelta(100, 15, 1)),
            ("bar", datetime.timedelta(100, 10, 1)),
        ],
        "test_fail_data": [
            ("foo", datetime.timedelta(100, 15, 1)),
            ("bar", datetime.timedelta(100, 20, 1)),
        ],
        "test_expression": [
            datetime.timedelta(100, 15, 1),
            datetime.timedelta(100, 10, 1),
        ],
    }

    def pytest_generate_tests(self, metafunc):
        """This function passes the parameter for each function based on parameter from get_data_param function"""
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
            "test_isin_check": [
                {"datatype": dt.UInt8, "data": self.sample_numeric_data},
                {"datatype": dt.UInt16, "data": self.sample_numeric_data},
                {"datatype": dt.UInt32, "data": self.sample_numeric_data},
                {"datatype": dt.UInt64, "data": self.sample_numeric_data},
                {"datatype": dt.Int8, "data": self.sample_numeric_data},
                {"datatype": dt.Int16, "data": self.sample_numeric_data},
                {"datatype": dt.Int32, "data": self.sample_numeric_data},
                {"datatype": dt.Int64, "data": self.sample_numeric_data},
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

    def test_isin_check(self, datatype, data) -> None:
        """Test the Check to see if all the values are equal to the defined value"""
        self.check_function(
            pa.Check.isin,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            data["test_expression"],
        )


class TestNotInCheck(BaseClass):
    """This class is used to test the notin check"""

    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 32)],
        "test_fail_data": [("foo", 30), ("bar", 31)],
        "test_expression": [30, 33],
    }

    sample_datetime_data = {
        "test_pass_data": [
            ("foo", datetime.datetime(2020, 10, 1, 12, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 12, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.datetime(2020, 10, 3, 10, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 10, 0)),
        ],
        "test_expression": [
            datetime.datetime(2020, 10, 3, 10, 0),
            datetime.datetime(2020, 10, 4, 11, 0),
        ],
    }

    sample_string_data = {
        "test_pass_data": [("foo", "b"), ("bar", "c")],
        "test_fail_data": [("foo", "a"), ("bar", "b")],
        "test_expression": ["a", "d"],
    }

    sample_duration_data = {
        "test_pass_data": [
            ("foo", datetime.timedelta(100, 20, 1)),
            ("bar", datetime.timedelta(100, 20, 1)),
        ],
        "test_fail_data": [
            ("foo", datetime.timedelta(100, 15, 1)),
            ("bar", datetime.timedelta(100, 20, 1)),
        ],
        "test_expression": [
            datetime.timedelta(100, 15, 1),
            datetime.timedelta(100, 10, 1),
        ],
    }

    def pytest_generate_tests(self, metafunc):
        """This function passes the parameter for each function based on parameter from get_data_param function"""
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
            "test_notin_check": [
                {"datatype": dt.UInt8, "data": self.sample_numeric_data},
                {"datatype": dt.UInt16, "data": self.sample_numeric_data},
                {"datatype": dt.UInt32, "data": self.sample_numeric_data},
                {"datatype": dt.UInt64, "data": self.sample_numeric_data},
                {"datatype": dt.Int8, "data": self.sample_numeric_data},
                {"datatype": dt.Int16, "data": self.sample_numeric_data},
                {"datatype": dt.Int32, "data": self.sample_numeric_data},
                {"datatype": dt.Int64, "data": self.sample_numeric_data},
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

    def test_notin_check(self, datatype, data) -> None:
        """Test the Check to see if all the values are equal to the defined value"""
        self.check_function(
            pa.Check.notin,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            data["test_expression"],
        )


class TestStringType(BaseClass):
    """This class is used to test the string type checks"""

    @pytest.mark.parametrize(
        "check_value",
        ["Ba", r"Ba+", re.compile("Ba"), re.compile(r"Ba+")],
    )
    def test_str_matches_check(self, check_value) -> None:
        """Test the Check to see if any value is not in the specified value"""
        check_func = pa.Check.str_matches

        pass_data = [("Bal", "Bat!"), ("Bal", "Bat78")]
        fail_data = [("Bal", "fooBar"), ("Bal", "Bam!")]
        self.check_function(
            check_func, pass_data, fail_data, dt.String(), check_value
        )

    @pytest.mark.parametrize(
        "check_value",
        ["Ba", r"Ba+", re.compile("Ba"), re.compile(r"Ba+")],
    )
    def test_str_contains_check(self, check_value) -> None:
        """Test the Check to see if any value is not in the specified value"""
        check_func = pa.Check.str_contains

        pass_data = [("Bal", "Bat!"), ("Bal", "Bat78")]
        fail_data = [("Bal", "Cs"), ("Bal", "Bam!")]
        self.check_function(
            check_func, pass_data, fail_data, dt.String(), check_value
        )

    def test_str_startswith_check(self) -> None:
        """Test the Check to see if any value is not in the specified value"""
        check_func = pa.Check.str_startswith
        check_value = "B"

        pass_data = [("Bal", "Bread"), ("Bal", "Butter")]
        fail_data = [("Bal", "Test"), ("Bal", "Butter")]
        self.check_function(
            check_func, pass_data, fail_data, dt.String(), check_value
        )

    def test_str_endswith_check(self) -> None:
        """Test the Check to see if any value is not in the specified value"""
        check_func = pa.Check.str_endswith
        check_value = "d"

        pass_data = [("Bal", "Bread"), ("Bal", "Bad")]
        fail_data = [("Bal", "Test"), ("Bal", "Bad")]
        self.check_function(
            check_func, pass_data, fail_data, dt.String(), check_value
        )

    @pytest.mark.parametrize(
        "check_value",
        [(3, None), (None, 4), (3, 7), (1, 4), (3, 4), (None, None)],
    )
    def test_str_length_check(self, check_value) -> None:
        """Test the Check to see if length of strings is within a specified range."""
        check_func = pa.Check.str_length

        pass_data = [("Bal", "Bat"), ("Bal", "Batt")]
        fail_data = [("Bal", "Cs"), ("Bal", "BamBam")]

        if check_value == (None, None):
            fail_on_init = True
            init_exception_cls = ValueError
        else:
            fail_on_init = False
            init_exception_cls = None

        self.check_function(
            check_func,
            pass_data,
            fail_data,
            dt.String(),
            check_value,
            fail_on_init=fail_on_init,
            init_exception_cls=init_exception_cls,
        )


class TestUniqueValuesEqCheck(BaseClass):
    """This class is used to test the unique values eq check"""

    sample_numeric_data = {
        "test_pass_data": [("foo", 32), ("bar", 31)],
        "test_fail_data": [("foo", 31), ("bar", 31)],
        "test_expression": [31, 32],
    }

    sample_datetime_data = {
        "test_pass_data": [
            ("foo", datetime.datetime(2020, 10, 1, 10, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 11, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.datetime(2020, 10, 3, 10, 0)),
            ("bar", datetime.datetime(2020, 10, 3, 10, 0)),
        ],
        "test_expression": [
            datetime.datetime(2020, 10, 1, 10, 0),
            datetime.datetime(2020, 10, 2, 11, 0),
        ],
    }

    sample_string_data = {
        "test_pass_data": [("foo", "b"), ("bar", "c")],
        "test_fail_data": [("foo", "a"), ("bar", "b")],
        "test_expression": ["b", "c"],
    }

    sample_duration_data = {
        "test_pass_data": [
            ("foo", datetime.timedelta(100, 15, 1)),
            ("bar", datetime.timedelta(100, 10, 1)),
        ],
        "test_fail_data": [
            ("foo", datetime.timedelta(100, 15, 1)),
            ("bar", datetime.timedelta(100, 20, 1)),
        ],
        "test_expression": [
            datetime.timedelta(100, 15, 1),
            datetime.timedelta(100, 10, 1),
        ],
    }

    def pytest_generate_tests(self, metafunc):
        """This function passes the parameter for each function based on parameter from get_data_param function"""
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
            "test_unique_values_eq_check": [
                {"datatype": dt.UInt8, "data": self.sample_numeric_data},
                {"datatype": dt.UInt16, "data": self.sample_numeric_data},
                {"datatype": dt.UInt32, "data": self.sample_numeric_data},
                {"datatype": dt.UInt64, "data": self.sample_numeric_data},
                {"datatype": dt.Int8, "data": self.sample_numeric_data},
                {"datatype": dt.Int16, "data": self.sample_numeric_data},
                {"datatype": dt.Int32, "data": self.sample_numeric_data},
                {"datatype": dt.Int64, "data": self.sample_numeric_data},
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
                # TODO(deepyaman): xfail this; raises ArrowNotImplementedError("Unsupported cast from month_day_nano_interval to duration using function cast_duration")
                # {
                #     "datatype": dt.Interval(unit="us"),
                #     "data": self.sample_duration_data,
                # },
            ]
        }

    def test_unique_values_eq_check(self, datatype, data) -> None:
        """Test the Check to see if all the values are equal to the defined value"""
        self.check_function(
            pa.Check.unique_values_eq,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            data["test_expression"],
        )
