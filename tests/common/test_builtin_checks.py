"""Cross-backend builtin check tests.

Parametrized over PolarsBackend and IbisBackend; runs against whichever
pandera backend is active (native or narwhals).  Narwhals-incompatible
dtype × backend combinations are xfailed rather than excluded.
"""

import datetime
import decimal
import inspect
import re

import pytest

from pandera import dtypes
from pandera.api.checks import Check
from pandera.config import CONFIG
from pandera.errors import SchemaError, SchemaErrors
from tests.common.conftest import BACKENDS


class BaseClass:
    def pytest_generate_tests(self, metafunc):
        all_params = self.get_data_param()
        func_params = all_params.get(metafunc.function.__name__)
        if not func_params:
            return

        extra_keys = sorted(k for k in func_params[0] if k != "datatype")
        argnames = ["backend", "dtype"] + extra_keys

        combined = []
        for b_param in BACKENDS:
            backend = b_param.values[0]
            for i, entry in enumerate(func_params):
                datatype = entry["datatype"]
                is_pandera_dtype = inspect.isclass(datatype) and issubclass(
                    datatype, dtypes.DataType
                )
                if is_pandera_dtype:
                    if backend.dtype(datatype) is None:
                        continue
                    dtype = datatype
                    dtype_id = str(datatype())
                else:
                    dtype = backend.dtype(datatype)
                    if dtype is None:
                        continue
                    dtype_id = datatype
                row = [backend, dtype] + [entry[k] for k in extra_keys]
                combined.append(
                    pytest.param(
                        *row,
                        id=f"data{i}-{dtype_id}-{backend.name}",
                        marks=b_param.marks,
                    )
                )

        metafunc.parametrize(argnames, combined)

    @staticmethod
    def check_function(
        backend,
        check_fn,
        pass_case_data,
        fail_case_data,
        dtype,
        function_args,
        fail_on_init=False,
        init_exception_cls=None,
    ):
        if fail_on_init:
            with pytest.raises(init_exception_cls):
                check_fn(*function_args)
            return

        if CONFIG.use_narwhals_backend and backend.is_narwhals_incompatible(
            dtype
        ):
            pytest.xfail(
                f"Narwhals engine does not support this dtype on {backend.name}"
            )

        if (
            CONFIG.use_narwhals_backend
            and backend.name == "ibis"
            and isinstance(dtype, type)
            and issubclass(dtype, dtypes.Timedelta)
        ):
            pytest.xfail(
                "Ibis does not support mixed-unit Python timedelta literals "
                "as check comparison values under the Narwhals backend"
            )

        schema = backend.DataFrameSchema(
            {
                "product": backend.Column(backend.string_dtype),
                "code": (
                    backend.Column(dtype, check_fn(*function_args))
                    if isinstance(function_args, tuple)
                    else backend.Column(dtype, check_fn(function_args))
                ),
            }
        )

        t_pass = backend.make_frame(pass_case_data, dtype)
        schema.validate(t_pass)

        with pytest.raises((SchemaError, SchemaErrors)):
            t_fail = backend.make_frame(fail_case_data, dtype)
            schema.validate(t_fail)


class TestEqualToCheck(BaseClass):
    sample_numeric_data = {
        "test_pass_data": [("foo", 30), ("bar", 30)],
        "test_fail_data": [("foo", 30), ("bar", 31)],
        "test_expression": 30,
    }
    sample_float_data = {
        "test_pass_data": [("foo", 30.0), ("bar", 30.0)],
        "test_fail_data": [("foo", 30.0), ("bar", 31.0)],
        "test_expression": 30.0,
    }
    sample_string_data = {
        "test_pass_data": [("foo", "a"), ("bar", "a")],
        "test_fail_data": [("foo", "a"), ("bar", "b")],
        "test_expression": "a",
    }
    sample_binary_data = {
        "test_pass_data": [("foo", b"a"), ("bar", b"a")],
        "test_fail_data": [("foo", b"a"), ("bar", b"b")],
        "test_expression": b"a",
    }
    sample_decimal_data = {
        "test_pass_data": [
            ("foo", decimal.Decimal("30")),
            ("bar", decimal.Decimal("30")),
        ],
        "test_fail_data": [
            ("foo", decimal.Decimal("30")),
            ("bar", decimal.Decimal("31")),
        ],
        "test_expression": decimal.Decimal("30"),
    }
    sample_date_data = {
        "test_pass_data": [
            ("foo", datetime.date(2020, 10, 1)),
            ("bar", datetime.date(2020, 10, 1)),
        ],
        "test_fail_data": [
            ("foo", datetime.date(2020, 10, 2)),
            ("bar", datetime.date(2020, 10, 2)),
        ],
        "test_expression": datetime.date(2020, 10, 1),
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
    sample_time_data = {
        "test_pass_data": [
            ("foo", datetime.time(10, 0)),
            ("bar", datetime.time(10, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.time(11, 0)),
            ("bar", datetime.time(11, 0)),
        ],
        "test_expression": datetime.time(10, 0),
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
    list_utf8 = {
        "test_pass_data": [("foo", ["a"]), ("bar", ["a"])],
        "test_fail_data": [("foo", ["a"]), ("bar", ["b"])],
        "test_expression": ["a"],
    }

    def get_data_param(self):
        return {
            "test_equal_to_check": [
                {"datatype": dtypes.UInt8, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt16, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt32, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt64, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int8, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int16, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int32, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int64, "data": self.sample_numeric_data},
                {"datatype": dtypes.Float32, "data": self.sample_float_data},
                {"datatype": dtypes.Float64, "data": self.sample_float_data},
                {"datatype": dtypes.String, "data": self.sample_string_data},
                {"datatype": dtypes.Binary, "data": self.sample_binary_data},
                {"datatype": "categorical", "data": self.sample_string_data},
                {"datatype": dtypes.Decimal, "data": self.sample_decimal_data},
                {"datatype": dtypes.Date, "data": self.sample_date_data},
                {
                    "datatype": dtypes.DateTime,
                    "data": self.sample_datetime_data,
                },
                {"datatype": dtypes.Time, "data": self.sample_time_data},
                {
                    "datatype": dtypes.Timedelta,
                    "data": self.sample_duration_data,
                },
                {"datatype": dtypes.Bool, "data": self.sample_boolean_data},
                {"datatype": "list_utf8", "data": self.list_utf8},
            ]
        }

    @pytest.mark.parametrize("check_fn", [Check.equal_to, Check.eq])
    def test_equal_to_check(self, backend, check_fn, dtype, data):
        self.check_function(
            backend,
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            dtype,
            data["test_expression"],
        )


class TestNotEqualToCheck(BaseClass):
    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 32)],
        "test_fail_data": [("foo", 30), ("bar", 31)],
        "test_expression": 30,
    }
    sample_float_data = {
        "test_pass_data": [("foo", 31.0), ("bar", 32.0)],
        "test_fail_data": [("foo", 30.0), ("bar", 31.0)],
        "test_expression": 30.0,
    }
    sample_string_data = {
        "test_pass_data": [("foo", "b"), ("bar", "c")],
        "test_fail_data": [("foo", "a"), ("bar", "a")],
        "test_expression": "a",
    }
    sample_binary_data = {
        "test_pass_data": [("foo", b"b"), ("bar", b"c")],
        "test_fail_data": [("foo", b"a"), ("bar", b"a")],
        "test_expression": b"a",
    }
    sample_decimal_data = {
        "test_pass_data": [
            ("foo", decimal.Decimal("31")),
            ("bar", decimal.Decimal("32")),
        ],
        "test_fail_data": [
            ("foo", decimal.Decimal("30")),
            ("bar", decimal.Decimal("30")),
        ],
        "test_expression": decimal.Decimal("30"),
    }
    sample_date_data = {
        "test_pass_data": [
            ("foo", datetime.date(2020, 10, 1)),
            ("bar", datetime.date(2020, 10, 2)),
        ],
        "test_fail_data": [
            ("foo", datetime.date(2020, 10, 3)),
            ("bar", datetime.date(2020, 10, 3)),
        ],
        "test_expression": datetime.date(2020, 10, 3),
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
    sample_time_data = {
        "test_pass_data": [
            ("foo", datetime.time(11, 0)),
            ("bar", datetime.time(12, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.time(10, 0)),
            ("bar", datetime.time(10, 0)),
        ],
        "test_expression": datetime.time(10, 0),
    }
    sample_duration_data = {
        "test_pass_data": [
            ("foo", datetime.timedelta(100, 11, 1)),
            ("bar", datetime.timedelta(100, 11, 1)),
        ],
        "test_fail_data": [
            ("foo", datetime.timedelta(100, 10, 1)),
            ("bar", datetime.timedelta(100, 10, 1)),
        ],
        "test_expression": datetime.timedelta(100, 10, 1),
    }
    sample_boolean_data = {
        "test_pass_data": [("foo", True), ("bar", True)],
        "test_fail_data": [("foo", False), ("bar", True)],
        "test_expression": False,
    }
    list_utf8 = {
        "test_pass_data": [("foo", ["b"]), ("bar", ["c"])],
        "test_fail_data": [("foo", ["a"]), ("bar", ["b"])],
        "test_expression": ["a"],
    }

    def get_data_param(self):
        return {
            "test_not_equal_to_check": [
                {"datatype": dtypes.UInt8, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt16, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt32, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt64, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int8, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int16, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int32, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int64, "data": self.sample_numeric_data},
                {"datatype": dtypes.Float32, "data": self.sample_float_data},
                {"datatype": dtypes.Float64, "data": self.sample_float_data},
                {"datatype": dtypes.String, "data": self.sample_string_data},
                {"datatype": dtypes.Binary, "data": self.sample_binary_data},
                {"datatype": "categorical", "data": self.sample_string_data},
                {"datatype": dtypes.Decimal, "data": self.sample_decimal_data},
                {"datatype": dtypes.Date, "data": self.sample_date_data},
                {
                    "datatype": dtypes.DateTime,
                    "data": self.sample_datetime_data,
                },
                {"datatype": dtypes.Time, "data": self.sample_time_data},
                {
                    "datatype": dtypes.Timedelta,
                    "data": self.sample_duration_data,
                },
                {"datatype": dtypes.Bool, "data": self.sample_boolean_data},
                {"datatype": "list_utf8", "data": self.list_utf8},
            ]
        }

    @pytest.mark.parametrize("check_fn", [Check.not_equal_to, Check.ne])
    def test_not_equal_to_check(self, backend, check_fn, dtype, data):
        self.check_function(
            backend,
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            dtype,
            data["test_expression"],
        )


class TestGreaterThanCheck(BaseClass):
    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 32)],
        "test_fail_data": [("foo", 30), ("bar", 31)],
        "test_expression": 30,
    }
    sample_float_data = {
        "test_pass_data": [("foo", 31.0), ("bar", 32.0)],
        "test_fail_data": [("foo", 30.0), ("bar", 31.0)],
        "test_expression": 30.0,
    }
    sample_decimal_data = {
        "test_pass_data": [
            ("foo", decimal.Decimal("31")),
            ("bar", decimal.Decimal("32")),
        ],
        "test_fail_data": [
            ("foo", decimal.Decimal("30")),
            ("bar", decimal.Decimal("31")),
        ],
        "test_expression": decimal.Decimal("30"),
    }
    sample_date_data = {
        "test_pass_data": [
            ("foo", datetime.date(2020, 10, 2)),
            ("bar", datetime.date(2020, 10, 2)),
        ],
        "test_fail_data": [
            ("foo", datetime.date(2020, 10, 1)),
            ("bar", datetime.date(2020, 10, 2)),
        ],
        "test_expression": datetime.date(2020, 10, 1),
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
    sample_time_data = {
        "test_pass_data": [
            ("foo", datetime.time(11, 0)),
            ("bar", datetime.time(12, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.time(10, 0)),
            ("bar", datetime.time(11, 0)),
        ],
        "test_expression": datetime.time(10, 0),
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

    def get_data_param(self):
        return {
            "test_greater_than_check": [
                {"datatype": dtypes.UInt8, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt16, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt32, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt64, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int8, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int16, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int32, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int64, "data": self.sample_numeric_data},
                {"datatype": dtypes.Float32, "data": self.sample_float_data},
                {"datatype": dtypes.Float64, "data": self.sample_float_data},
                {"datatype": dtypes.Decimal, "data": self.sample_decimal_data},
                {"datatype": dtypes.Date, "data": self.sample_date_data},
                {
                    "datatype": dtypes.DateTime,
                    "data": self.sample_datetime_data,
                },
                {"datatype": dtypes.Time, "data": self.sample_time_data},
                {
                    "datatype": dtypes.Timedelta,
                    "data": self.sample_duration_data,
                },
            ]
        }

    @pytest.mark.parametrize("check_fn", [Check.greater_than, Check.gt])
    def test_greater_than_check(self, backend, check_fn, dtype, data):
        self.check_function(
            backend,
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            dtype,
            data["test_expression"],
        )


class TestGreaterThanEqualToCheck(BaseClass):
    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 32)],
        "test_fail_data": [("foo", 30), ("bar", 31)],
        "test_expression": 31,
    }
    sample_float_data = {
        "test_pass_data": [("foo", 31.0), ("bar", 32.0)],
        "test_fail_data": [("foo", 30.0), ("bar", 31.0)],
        "test_expression": 31.0,
    }
    sample_decimal_data = {
        "test_pass_data": [
            ("foo", decimal.Decimal("31")),
            ("bar", decimal.Decimal("32")),
        ],
        "test_fail_data": [
            ("foo", decimal.Decimal("30")),
            ("bar", decimal.Decimal("31")),
        ],
        "test_expression": decimal.Decimal("31"),
    }
    sample_date_data = {
        "test_pass_data": [
            ("foo", datetime.date(2020, 10, 1)),
            ("bar", datetime.date(2020, 10, 2)),
        ],
        "test_fail_data": [
            ("foo", datetime.date(2020, 10, 1)),
            ("bar", datetime.date(2020, 9, 1)),
        ],
        "test_expression": datetime.date(2020, 10, 1),
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
    sample_time_data = {
        "test_pass_data": [
            ("foo", datetime.time(10, 0)),
            ("bar", datetime.time(11, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.time(10, 0)),
            ("bar", datetime.time(9, 0)),
        ],
        "test_expression": datetime.time(10, 0),
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

    def get_data_param(self):
        return {
            "test_greater_than_or_equal_to_check": [
                {"datatype": dtypes.UInt8, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt16, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt32, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt64, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int8, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int16, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int32, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int64, "data": self.sample_numeric_data},
                {"datatype": dtypes.Float32, "data": self.sample_float_data},
                {"datatype": dtypes.Float64, "data": self.sample_float_data},
                {"datatype": dtypes.Decimal, "data": self.sample_decimal_data},
                {"datatype": dtypes.Date, "data": self.sample_date_data},
                {
                    "datatype": dtypes.DateTime,
                    "data": self.sample_datetime_data,
                },
                {"datatype": dtypes.Time, "data": self.sample_time_data},
                {
                    "datatype": dtypes.Timedelta,
                    "data": self.sample_duration_data,
                },
            ]
        }

    @pytest.mark.parametrize(
        "check_fn", [Check.greater_than_or_equal_to, Check.ge]
    )
    def test_greater_than_or_equal_to_check(
        self, backend, check_fn, dtype, data
    ):
        self.check_function(
            backend,
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            dtype,
            data["test_expression"],
        )


class TestLessThanCheck(BaseClass):
    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 32)],
        "test_fail_data": [("foo", 34), ("bar", 33)],
        "test_expression": 33,
    }
    sample_float_data = {
        "test_pass_data": [("foo", 31.0), ("bar", 32.0)],
        "test_fail_data": [("foo", 34.0), ("bar", 33.0)],
        "test_expression": 33.0,
    }
    sample_decimal_data = {
        "test_pass_data": [
            ("foo", decimal.Decimal("31")),
            ("bar", decimal.Decimal("32")),
        ],
        "test_fail_data": [
            ("foo", decimal.Decimal("34")),
            ("bar", decimal.Decimal("33")),
        ],
        "test_expression": decimal.Decimal("33"),
    }
    sample_date_data = {
        "test_pass_data": [
            ("foo", datetime.date(2020, 10, 1)),
            ("bar", datetime.date(2020, 10, 1)),
        ],
        "test_fail_data": [
            ("foo", datetime.date(2020, 11, 1)),
            ("bar", datetime.date(2020, 12, 1)),
        ],
        "test_expression": datetime.date(2020, 11, 1),
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
    sample_time_data = {
        "test_pass_data": [
            ("foo", datetime.time(10, 0)),
            ("bar", datetime.time(10, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.time(12, 0)),
            ("bar", datetime.time(13, 0)),
        ],
        "test_expression": datetime.time(11, 0),
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

    def get_data_param(self):
        return {
            "test_less_than_check": [
                {"datatype": dtypes.UInt8, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt16, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt32, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt64, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int8, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int16, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int32, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int64, "data": self.sample_numeric_data},
                {"datatype": dtypes.Float32, "data": self.sample_float_data},
                {"datatype": dtypes.Float64, "data": self.sample_float_data},
                {"datatype": dtypes.Decimal, "data": self.sample_decimal_data},
                {"datatype": dtypes.Date, "data": self.sample_date_data},
                {
                    "datatype": dtypes.DateTime,
                    "data": self.sample_datetime_data,
                },
                {"datatype": dtypes.Time, "data": self.sample_time_data},
                {
                    "datatype": dtypes.Timedelta,
                    "data": self.sample_duration_data,
                },
            ]
        }

    @pytest.mark.parametrize("check_fn", [Check.less_than, Check.lt])
    def test_less_than_check(self, backend, check_fn, dtype, data):
        self.check_function(
            backend,
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            dtype,
            data["test_expression"],
        )


class TestLessThanEqualToCheck(BaseClass):
    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 33)],
        "test_fail_data": [("foo", 34), ("bar", 31)],
        "test_expression": 33,
    }
    sample_float_data = {
        "test_pass_data": [("foo", 31.0), ("bar", 33.0)],
        "test_fail_data": [("foo", 34.0), ("bar", 31.0)],
        "test_expression": 33.0,
    }
    sample_decimal_data = {
        "test_pass_data": [
            ("foo", decimal.Decimal("31")),
            ("bar", decimal.Decimal("33")),
        ],
        "test_fail_data": [
            ("foo", decimal.Decimal("34")),
            ("bar", decimal.Decimal("31")),
        ],
        "test_expression": decimal.Decimal("33"),
    }
    sample_date_data = {
        "test_pass_data": [
            ("foo", datetime.date(2020, 11, 1)),
            ("bar", datetime.date(2020, 10, 1)),
        ],
        "test_fail_data": [
            ("foo", datetime.date(2020, 11, 1)),
            ("bar", datetime.date(2020, 12, 1)),
        ],
        "test_expression": datetime.date(2020, 11, 1),
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
    sample_time_data = {
        "test_pass_data": [
            ("foo", datetime.time(11, 0)),
            ("bar", datetime.time(10, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.time(12, 0)),
            ("bar", datetime.time(12, 0)),
        ],
        "test_expression": datetime.time(11, 0),
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

    def get_data_param(self):
        return {
            "test_less_than_or_equal_to_check": [
                {"datatype": dtypes.UInt8, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt16, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt32, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt64, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int8, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int16, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int32, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int64, "data": self.sample_numeric_data},
                {"datatype": dtypes.Float32, "data": self.sample_float_data},
                {"datatype": dtypes.Float64, "data": self.sample_float_data},
                {"datatype": dtypes.Decimal, "data": self.sample_decimal_data},
                {"datatype": dtypes.Date, "data": self.sample_date_data},
                {
                    "datatype": dtypes.DateTime,
                    "data": self.sample_datetime_data,
                },
                {"datatype": dtypes.Time, "data": self.sample_time_data},
                {
                    "datatype": dtypes.Timedelta,
                    "data": self.sample_duration_data,
                },
            ]
        }

    @pytest.mark.parametrize(
        "check_fn", [Check.less_than_or_equal_to, Check.le]
    )
    def test_less_than_or_equal_to_check(self, backend, check_fn, dtype, data):
        self.check_function(
            backend,
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            dtype,
            data["test_expression"],
        )


class TestIsInCheck(BaseClass):
    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 32)],
        "test_fail_data": [("foo", 30), ("bar", 31)],
        "test_expression": [31, 32],
    }
    sample_float_data = {
        "test_pass_data": [("foo", 31.0), ("bar", 32.0)],
        "test_fail_data": [("foo", 30.0), ("bar", 31.0)],
        "test_expression": [31.0, 32.0],
    }
    sample_string_data = {
        "test_pass_data": [("foo", "b"), ("bar", "c")],
        "test_fail_data": [("foo", "a"), ("bar", "b")],
        "test_expression": ["b", "c"],
    }
    sample_binary_data = {
        "test_pass_data": [("foo", b"b"), ("bar", b"c")],
        "test_fail_data": [("foo", b"a"), ("bar", b"b")],
        "test_expression": [b"b", b"c"],
    }
    sample_date_data = {
        "test_pass_data": [
            ("foo", datetime.date(2020, 10, 1)),
            ("bar", datetime.date(2020, 10, 2)),
        ],
        "test_fail_data": [
            ("foo", datetime.date(2020, 10, 3)),
            ("bar", datetime.date(2020, 10, 2)),
        ],
        "test_expression": [
            datetime.date(2020, 10, 1),
            datetime.date(2020, 10, 2),
        ],
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
    sample_time_data = {
        "test_pass_data": [
            ("foo", datetime.time(10, 0)),
            ("bar", datetime.time(11, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.time(10, 0)),
            ("bar", datetime.time(12, 0)),
        ],
        "test_expression": [datetime.time(10, 0), datetime.time(11, 0)],
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

    def get_data_param(self):
        return {
            "test_isin_check": [
                {"datatype": dtypes.UInt8, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt16, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt32, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt64, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int8, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int16, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int32, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int64, "data": self.sample_numeric_data},
                {"datatype": dtypes.Float32, "data": self.sample_float_data},
                {"datatype": dtypes.Float64, "data": self.sample_float_data},
                # FIXME(deepyaman): polars isin cannot check List(Decimal(38, 0))
                # values in Decimal(38, 10) data; skip decimal for now
                {"datatype": dtypes.Date, "data": self.sample_date_data},
                {
                    "datatype": dtypes.DateTime,
                    "data": self.sample_datetime_data,
                },
                {"datatype": dtypes.Time, "data": self.sample_time_data},
                {
                    "datatype": dtypes.Timedelta,
                    "data": self.sample_duration_data,
                },
                {"datatype": "categorical", "data": self.sample_string_data},
                {"datatype": dtypes.String, "data": self.sample_string_data},
                {"datatype": dtypes.Binary, "data": self.sample_binary_data},
            ]
        }

    def test_isin_check(self, backend, dtype, data):
        self.check_function(
            backend,
            Check.isin,
            data["test_pass_data"],
            data["test_fail_data"],
            dtype,
            data["test_expression"],
        )


class TestNotInCheck(BaseClass):
    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 32)],
        "test_fail_data": [("foo", 30), ("bar", 31)],
        "test_expression": [30, 33],
    }
    sample_float_data = {
        "test_pass_data": [("foo", 31.0), ("bar", 32.0)],
        "test_fail_data": [("foo", 30.0), ("bar", 31.0)],
        "test_expression": [30.0, 33.0],
    }
    sample_string_data = {
        "test_pass_data": [("foo", "b"), ("bar", "c")],
        "test_fail_data": [("foo", "a"), ("bar", "b")],
        "test_expression": ["a", "d"],
    }
    sample_binary_data = {
        "test_pass_data": [("foo", b"b"), ("bar", b"c")],
        "test_fail_data": [("foo", b"a"), ("bar", b"b")],
        "test_expression": [b"a", b"d"],
    }
    sample_date_data = {
        "test_pass_data": [
            ("foo", datetime.date(2020, 10, 1)),
            ("bar", datetime.date(2020, 10, 2)),
        ],
        "test_fail_data": [
            ("foo", datetime.date(2020, 10, 3)),
            ("bar", datetime.date(2020, 10, 2)),
        ],
        "test_expression": [
            datetime.date(2020, 10, 3),
            datetime.date(2020, 10, 4),
        ],
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
    sample_time_data = {
        "test_pass_data": [
            ("foo", datetime.time(10, 0)),
            ("bar", datetime.time(11, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.time(12, 0)),
            ("bar", datetime.time(10, 0)),
        ],
        "test_expression": [datetime.time(12, 0), datetime.time(13, 0)],
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

    def get_data_param(self):
        return {
            "test_notin_check": [
                {"datatype": dtypes.UInt8, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt16, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt32, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt64, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int8, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int16, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int32, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int64, "data": self.sample_numeric_data},
                {"datatype": dtypes.Float32, "data": self.sample_float_data},
                {"datatype": dtypes.Float64, "data": self.sample_float_data},
                # FIXME(deepyaman): polars notin cannot check List(Decimal(38, 0))
                # values in Decimal(38, 10) data; skip decimal for now
                {"datatype": dtypes.Date, "data": self.sample_date_data},
                {
                    "datatype": dtypes.DateTime,
                    "data": self.sample_datetime_data,
                },
                {"datatype": dtypes.Time, "data": self.sample_time_data},
                {
                    "datatype": dtypes.Timedelta,
                    "data": self.sample_duration_data,
                },
                {"datatype": "categorical", "data": self.sample_string_data},
                {"datatype": dtypes.String, "data": self.sample_string_data},
                {"datatype": dtypes.Binary, "data": self.sample_binary_data},
            ]
        }

    def test_notin_check(self, backend, dtype, data):
        self.check_function(
            backend,
            Check.notin,
            data["test_pass_data"],
            data["test_fail_data"],
            dtype,
            data["test_expression"],
        )


class TestStringType(BaseClass):
    def pytest_generate_tests(self, metafunc):
        pass  # no dtype parametrization; string checks always use the string dtype

    @staticmethod
    def _run_string_check(
        backend, check_fn, pass_data, fail_data, check_value
    ):
        dtype = backend.string_dtype
        schema = backend.DataFrameSchema(
            {
                "product": backend.Column(backend.string_dtype),
                "code": backend.Column(dtype, check_fn(check_value)),
            }
        )
        t_pass = backend.make_frame(pass_data, dtype)
        schema.validate(t_pass)
        with pytest.raises((SchemaError, SchemaErrors)):
            t_fail = backend.make_frame(fail_data, dtype)
            schema.validate(t_fail)

    def test_str_startswith_check(self, backend):
        self._run_string_check(
            backend,
            Check.str_startswith,
            pass_data=[("Bal", "Bread"), ("Bal", "Butter")],
            fail_data=[("Bal", "Test"), ("Bal", "Butter")],
            check_value="B",
        )

    def test_str_endswith_check(self, backend):
        self._run_string_check(
            backend,
            Check.str_endswith,
            pass_data=[("Bal", "Bread"), ("Bal", "Bad")],
            fail_data=[("Bal", "Test"), ("Bal", "Bad")],
            check_value="d",
        )

    @pytest.mark.parametrize(
        "check_value",
        ["Ba", r"Ba+", re.compile("Ba"), re.compile(r"Ba+")],
    )
    def test_str_matches_check(self, backend, check_value):
        self._run_string_check(
            backend,
            Check.str_matches,
            pass_data=[("Bal", "Bat!"), ("Bal", "Bat78")],
            fail_data=[("Bal", "fooBar"), ("Bal", "Bam!")],
            check_value=check_value,
        )

    @pytest.mark.parametrize(
        "check_value",
        ["Ba", r"Ba+", re.compile("Ba"), re.compile(r"Ba+")],
    )
    def test_str_contains_check(self, backend, check_value):
        self._run_string_check(
            backend,
            Check.str_contains,
            pass_data=[("Bal", "Bat!"), ("Bal", "Bat78")],
            fail_data=[("Bal", "Cs"), ("Bal", "Bam!")],
            check_value=check_value,
        )

    @pytest.mark.parametrize(
        "check_value",
        [(3, None), (None, 4), (3, 7), (1, 4), (3, 4), (None, None)],
    )
    def test_str_length_check(self, backend, check_value):
        pass_data = [("Bal", "Bat"), ("Bal", "Batt")]
        fail_data = [("Bal", "Cs"), ("Bal", "BamBam")]
        if check_value == (None, None):
            with pytest.raises(ValueError):
                Check.str_length(*check_value)
            return
        self._run_string_check(
            backend,
            lambda v: Check.str_length(*v),
            pass_data=pass_data,
            fail_data=fail_data,
            check_value=check_value,
        )

    @pytest.mark.parametrize("check_value", [3, 4])
    def test_str_length_exact_check(self, backend, check_value):
        if check_value == 3:
            pass_data = [("Bal", "Bat"), ("Bal", "Bam")]
            fail_data = [("Bal", "Batt"), ("Bal", "BamBam")]
        else:
            pass_data = [("Bal", "Batt"), ("Bal", "Bamm")]
            fail_data = [("Bal", "Bat"), ("Bal", "BamBam")]
        dtype = backend.string_dtype
        schema = backend.DataFrameSchema(
            {
                "product": backend.Column(backend.string_dtype),
                "code": backend.Column(dtype, Check.str_length(check_value)),
            }
        )
        t_pass = backend.make_frame(pass_data, dtype)
        schema.validate(t_pass)
        with pytest.raises((SchemaError, SchemaErrors)):
            t_fail = backend.make_frame(fail_data, dtype)
            schema.validate(t_fail)


class TestInRangeCheck(BaseClass):
    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 33)],
        "test_fail_data": [("foo", 35), ("bar", 31)],
    }
    sample_float_data = {
        "test_pass_data": [("foo", 31.0), ("bar", 33.0)],
        "test_fail_data": [("foo", 35.0), ("bar", 31.0)],
    }
    sample_decimal_data = {
        "test_pass_data": [
            ("foo", decimal.Decimal("31")),
            ("bar", decimal.Decimal("33")),
        ],
        "test_fail_data": [
            ("foo", decimal.Decimal("35")),
            ("bar", decimal.Decimal("31")),
        ],
    }
    sample_date_data = {
        "test_pass_data": [
            ("foo", datetime.date(2020, 10, 1)),
            ("bar", datetime.date(2020, 10, 2)),
        ],
        "test_fail_data": [
            ("foo", datetime.date(2020, 9, 30)),
            ("bar", datetime.date(2020, 10, 5)),
        ],
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
    sample_time_data = {
        "test_pass_data": [
            ("foo", datetime.time(11, 0)),
            ("bar", datetime.time(12, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.time(10, 0)),
            ("bar", datetime.time(15, 0)),
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

    def get_data_param(self):
        param_vals = [
            {"datatype": dtypes.UInt8, "data": self.sample_numeric_data},
            {"datatype": dtypes.UInt16, "data": self.sample_numeric_data},
            {"datatype": dtypes.UInt32, "data": self.sample_numeric_data},
            {"datatype": dtypes.UInt64, "data": self.sample_numeric_data},
            {"datatype": dtypes.Int8, "data": self.sample_numeric_data},
            {"datatype": dtypes.Int16, "data": self.sample_numeric_data},
            {"datatype": dtypes.Int32, "data": self.sample_numeric_data},
            {"datatype": dtypes.Int64, "data": self.sample_numeric_data},
            {"datatype": dtypes.Float32, "data": self.sample_float_data},
            {"datatype": dtypes.Float64, "data": self.sample_float_data},
            {"datatype": dtypes.Decimal, "data": self.sample_decimal_data},
            {"datatype": dtypes.Date, "data": self.sample_date_data},
            {"datatype": dtypes.DateTime, "data": self.sample_datetime_data},
            {"datatype": dtypes.Time, "data": self.sample_time_data},
            {"datatype": dtypes.Timedelta, "data": self.sample_duration_data},
        ]
        return {
            "test_inrange_exclude_min_max_check": param_vals,
            "test_inrange_exclude_min_only_check": param_vals,
            "test_inrange_exclude_max_only_check": param_vals,
            "test_inrange_include_min_max_check": param_vals,
        }

    def _create_min_max(self, data):
        values = [row[1] for row in data["test_pass_data"]]
        min_val, max_val = min(values), max(values)
        if isinstance(
            min_val, (datetime.datetime, datetime.date, datetime.timedelta)
        ):
            add_value = datetime.timedelta(1)
        elif isinstance(min_val, datetime.time):
            add_value = 1
        else:
            add_value = 1
        return min_val, max_val, add_value

    def _safe_add(self, val, delta):
        if isinstance(val, datetime.time):
            return datetime.time(val.hour + delta)
        return val + delta

    def _safe_sub(self, val, delta):
        if isinstance(val, datetime.time):
            return datetime.time(val.hour - delta)
        return val - delta

    @pytest.mark.parametrize("check_fn", [Check.in_range, Check.between])
    def test_inrange_exclude_min_max_check(
        self, backend, check_fn, dtype, data
    ):
        min_val, max_val, add_val = self._create_min_max(data)
        self.check_function(
            backend,
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            dtype,
            (
                self._safe_sub(min_val, add_val),
                self._safe_add(max_val, add_val),
                False,
                False,
            ),
        )

    @pytest.mark.parametrize("check_fn", [Check.in_range, Check.between])
    def test_inrange_exclude_min_only_check(
        self, backend, check_fn, dtype, data
    ):
        min_val, max_val, add_val = self._create_min_max(data)
        self.check_function(
            backend,
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            dtype,
            (min_val, self._safe_add(max_val, add_val), True, False),
        )

    @pytest.mark.parametrize("check_fn", [Check.in_range, Check.between])
    def test_inrange_exclude_max_only_check(
        self, backend, check_fn, dtype, data
    ):
        min_val, max_val, add_val = self._create_min_max(data)
        self.check_function(
            backend,
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            dtype,
            (self._safe_sub(min_val, add_val), max_val, False, True),
        )

    @pytest.mark.parametrize("check_fn", [Check.in_range, Check.between])
    def test_inrange_include_min_max_check(
        self, backend, check_fn, dtype, data
    ):
        min_val, max_val, _ = self._create_min_max(data)
        self.check_function(
            backend,
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            dtype,
            (min_val, max_val, True, True),
        )


class TestUniqueValuesEqCheck(BaseClass):
    sample_numeric_data = {
        "test_pass_data": [("foo", 32), ("bar", 31)],
        "test_fail_data": [("foo", 31), ("bar", 31)],
        "test_expression": [31, 32],
    }
    sample_float_data = {
        "test_pass_data": [("foo", 32.0), ("bar", 31.0)],
        "test_fail_data": [("foo", 31.0), ("bar", 31.0)],
        "test_expression": [31.0, 32.0],
    }
    sample_decimal_data = {
        "test_pass_data": [
            ("foo", decimal.Decimal("32")),
            ("bar", decimal.Decimal("31")),
        ],
        "test_fail_data": [
            ("foo", decimal.Decimal("31")),
            ("bar", decimal.Decimal("31")),
        ],
        "test_expression": [decimal.Decimal("31"), decimal.Decimal("32")],
    }
    sample_string_data = {
        "test_pass_data": [("foo", "b"), ("bar", "c")],
        "test_fail_data": [("foo", "a"), ("bar", "b")],
        "test_expression": ["b", "c"],
    }
    sample_binary_data = {
        "test_pass_data": [("foo", b"b"), ("bar", b"c")],
        "test_fail_data": [("foo", b"a"), ("bar", b"b")],
        "test_expression": [b"b", b"c"],
    }
    sample_date_data = {
        "test_pass_data": [
            ("foo", datetime.date(2020, 10, 1)),
            ("bar", datetime.date(2020, 10, 2)),
        ],
        "test_fail_data": [
            ("foo", datetime.date(2020, 10, 3)),
            ("bar", datetime.date(2020, 10, 3)),
        ],
        "test_expression": [
            datetime.date(2020, 10, 1),
            datetime.date(2020, 10, 2),
        ],
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
    sample_time_data = {
        "test_pass_data": [
            ("foo", datetime.time(10, 0)),
            ("bar", datetime.time(11, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.time(10, 0)),
            ("bar", datetime.time(10, 0)),
        ],
        "test_expression": [datetime.time(10, 0), datetime.time(11, 0)],
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

    def get_data_param(self):
        return {
            "test_unique_values_eq_check": [
                {"datatype": dtypes.UInt8, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt16, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt32, "data": self.sample_numeric_data},
                {"datatype": dtypes.UInt64, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int8, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int16, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int32, "data": self.sample_numeric_data},
                {"datatype": dtypes.Int64, "data": self.sample_numeric_data},
                {"datatype": dtypes.Float32, "data": self.sample_float_data},
                {"datatype": dtypes.Float64, "data": self.sample_float_data},
                {"datatype": dtypes.Decimal, "data": self.sample_decimal_data},
                {"datatype": dtypes.Date, "data": self.sample_date_data},
                {
                    "datatype": dtypes.DateTime,
                    "data": self.sample_datetime_data,
                },
                {"datatype": dtypes.Time, "data": self.sample_time_data},
                # TODO(deepyaman): ibis raises ArrowNotImplementedError for
                # Interval → duration cast; skip duration for now
                {"datatype": "categorical", "data": self.sample_string_data},
                {"datatype": dtypes.String, "data": self.sample_string_data},
                {"datatype": dtypes.Binary, "data": self.sample_binary_data},
            ]
        }

    @pytest.mark.xfail(
        condition=CONFIG.use_narwhals_backend,
        reason="unique_values_eq check not registered for Narwhals backend (KeyError: narwhals.stable.v1.Expr)",
    )
    def test_unique_values_eq_check(self, backend, dtype, data):
        self.check_function(
            backend,
            Check.unique_values_eq,
            data["test_pass_data"],
            data["test_fail_data"],
            dtype,
            data["test_expression"],
        )
