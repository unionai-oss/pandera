# pylint: disable=W0212
"""Unit tests for inferring statistics of pandas objects."""

import unittest.mock as mock
import pandas as pd
import pytest

import pandera as pa
import pandera.extensions as pa_ext
from pandera import PandasDtype, dtypes, schema_statistics

DEFAULT_INT = PandasDtype.from_str_alias(dtypes._DEFAULT_PANDAS_INT_TYPE)
DEFAULT_FLOAT = PandasDtype.from_str_alias(dtypes._DEFAULT_PANDAS_FLOAT_TYPE)


@pytest.fixture(scope="function")
def extra_registered_checks():
    """temporarily registers custom checks onto the Check class"""
    with mock.patch(
        "pandera.Check.REGISTERED_CUSTOM_CHECKS", new_callable=dict
    ):
        # register custom checks here
        @pa_ext.register_check_method()
        def no_param_check(_: pd.DataFrame) -> bool:
            return True

        yield


def _create_dataframe(multi_index=False, nullable=False):
    if multi_index:
        index = pd.MultiIndex.from_arrays(
            [[1, 1, 2], ["a", "b", "c"]],
            names=["int_index", "str_index"],
        )
    else:
        index = pd.Index([10, 11, 12], name="int_index")

    df = pd.DataFrame(
        data={
            "int": [1, 2, 3],
            "float": [1.0, 2.0, 3.0],
            "boolean": [True, False, True],
            "string": ["a", "b", "c"],
            "datetime": pd.to_datetime(["20180101", "20180102", "20180103"]),
        },
        index=index,
    )

    if nullable:
        df.iloc[0, :] = None

    return df


@pytest.mark.parametrize(
    "multi_index, nullable",
    [
        [False, False],
        [True, False],
        [False, True],
        [True, True],
    ],
)
def test_infer_dataframe_statistics(multi_index, nullable):
    """Test dataframe statistics are correctly inferred."""
    dataframe = _create_dataframe(multi_index, nullable)
    statistics = schema_statistics.infer_dataframe_statistics(dataframe)
    stat_columns = statistics["columns"]

    if nullable:
        # bool and int dtypes are cast to float in the nullable case
        assert stat_columns["int"]["pandas_dtype"] is DEFAULT_FLOAT
        assert stat_columns["boolean"]["pandas_dtype"] is DEFAULT_FLOAT
    else:
        assert stat_columns["int"]["pandas_dtype"] is DEFAULT_INT
        assert stat_columns["boolean"]["pandas_dtype"] is pa.Bool

    assert stat_columns["float"]["pandas_dtype"] is DEFAULT_FLOAT
    assert stat_columns["string"]["pandas_dtype"] is pa.String
    assert stat_columns["datetime"]["pandas_dtype"] is pa.DateTime

    if multi_index:
        stat_indices = statistics["index"]
        for stat_index, name, dtype in zip(
            stat_indices, ["int_index", "str_index"], [DEFAULT_INT, pa.String]
        ):
            assert stat_index["name"] == name
            assert stat_index["pandas_dtype"] is dtype
            assert not stat_index["nullable"]
    else:
        stat_index = statistics["index"][0]
        assert stat_index["name"] == "int_index"
        assert stat_index["pandas_dtype"] is DEFAULT_INT
        assert not stat_index["nullable"]

    for properties in stat_columns.values():
        if nullable:
            assert properties["nullable"]
        else:
            assert not properties["nullable"]


@pytest.mark.parametrize(
    "check_stats, expectation",
    [
        [
            {"greater_than_or_equal_to": {"min_value": 1}},
            [pa.Check.greater_than_or_equal_to(1)],
        ],
        [
            {"less_than_or_equal_to": {"max_value": 10}},
            [pa.Check.less_than_or_equal_to(10)],
        ],
        [
            {"isin": {"allowed_values": ["a", "b", "c"]}},
            [pa.Check.isin(["a", "b", "c"])],
        ],
        [
            {
                "greater_than_or_equal_to": {"min_value": 1},
                "less_than_or_equal_to": {"max_value": 10},
            },
            [
                pa.Check.greater_than_or_equal_to(1),
                pa.Check.less_than_or_equal_to(10),
            ],
        ],
        [{}, None],
    ],
)
def test_parse_check_statistics(check_stats, expectation):
    """Test that Checks are correctly parsed from check statistics."""
    if expectation is None:
        expectation = []
    checks = schema_statistics.parse_check_statistics(check_stats)
    if checks is None:
        checks = []
    assert set(checks) == set(expectation)


@pytest.mark.parametrize(
    "series, expectation",
    [
        *[
            [
                pd.Series([1, 2, 3], dtype=dtype.str_alias),
                {
                    "pandas_dtype": dtype,
                    "nullable": False,
                    "checks": {
                        "greater_than_or_equal_to": 1,
                        "less_than_or_equal_to": 3,
                    },
                    "name": None,
                },
            ]
            for dtype in (
                x
                for x in schema_statistics.NUMERIC_DTYPES
                if x != PandasDtype.DateTime
            )
        ],
        [
            pd.Series(["a", "b", "c", "a"], dtype="category"),
            {
                "pandas_dtype": pa.Category,
                "nullable": False,
                "checks": {"isin": ["a", "b", "c"]},
                "name": None,
            },
        ],
        [
            pd.Series(["a", "b", "c", "a"], name="str_series"),
            {
                "pandas_dtype": pa.String,
                "nullable": False,
                "checks": None,
                "name": "str_series",
            },
        ],
        [
            pd.Series(pd.to_datetime(["20180101", "20180102", "20180103"])),
            {
                "pandas_dtype": pa.DateTime,
                "nullable": False,
                "checks": {
                    "greater_than_or_equal_to": pd.Timestamp("20180101"),
                    "less_than_or_equal_to": pd.Timestamp("20180103"),
                },
                "name": None,
            },
        ],
    ],
)
def test_infer_series_schema_statistics(series, expectation):
    """Test series statistics are correctly inferred."""
    statistics = schema_statistics.infer_series_statistics(series)
    assert statistics == expectation


INTEGER_TYPES = [
    PandasDtype.Int,
    PandasDtype.Int8,
    PandasDtype.Int16,
    PandasDtype.Int32,
    PandasDtype.Int64,
    PandasDtype.UInt8,
    PandasDtype.UInt16,
    PandasDtype.UInt32,
    PandasDtype.UInt64,
]


@pytest.mark.parametrize(
    "null_index, series, expectation",
    [
        *[
            [
                0,
                pd.Series([1, 2, 3], dtype=dtype.value),
                {
                    # introducing nans to integer arrays upcasts to float
                    "pandas_dtype": DEFAULT_FLOAT,
                    "nullable": True,
                    "checks": {
                        "greater_than_or_equal_to": 2,
                        "less_than_or_equal_to": 3,
                    },
                    "name": None,
                },
            ]
            for dtype in INTEGER_TYPES
        ],
        [
            # introducing nans to integer arrays upcasts to float
            0,
            pd.Series([True, False, True, False]),
            {
                "pandas_dtype": DEFAULT_FLOAT,
                "nullable": True,
                "checks": {
                    "greater_than_or_equal_to": 0,
                    "less_than_or_equal_to": 1,
                },
                "name": None,
            },
        ],
        [
            0,
            pd.Series(["a", "b", "c", "a"], dtype="category"),
            {
                "pandas_dtype": pa.Category,
                "nullable": True,
                "checks": {"isin": ["a", "b", "c"]},
                "name": None,
            },
        ],
        [
            0,
            pd.Series(["a", "b", "c", "a"], name="str_series"),
            {
                "pandas_dtype": pa.String,
                "nullable": True,
                "checks": None,
                "name": "str_series",
            },
        ],
        [
            2,
            pd.Series(pd.to_datetime(["20180101", "20180102", "20180103"])),
            {
                "pandas_dtype": pa.DateTime,
                "nullable": True,
                "checks": {
                    "greater_than_or_equal_to": pd.Timestamp("20180101"),
                    "less_than_or_equal_to": pd.Timestamp("20180102"),
                },
                "name": None,
            },
        ],
    ],
)
def test_infer_nullable_series_schema_statistics(
    null_index, series, expectation
):
    """Test nullable series statistics are correctly inferred."""
    series.iloc[null_index] = None
    statistics = schema_statistics.infer_series_statistics(series)
    assert statistics == expectation


@pytest.mark.parametrize(
    "index, expectation",
    [
        [
            pd.RangeIndex(20),
            [
                {
                    "name": None,
                    "pandas_dtype": PandasDtype.Int,
                    "nullable": False,
                    "checks": {
                        "greater_than_or_equal_to": 0,
                        "less_than_or_equal_to": 19,
                    },
                }
            ],
        ],
        [
            pd.Index([1, 2, 3], name="int_index"),
            [
                {
                    "name": "int_index",
                    "pandas_dtype": PandasDtype.Int,
                    "nullable": False,
                    "checks": {
                        "greater_than_or_equal_to": 1,
                        "less_than_or_equal_to": 3,
                    },
                }
            ],
        ],
        [
            pd.Index(["foo", "bar", "baz"], name="str_index"),
            [
                {
                    "name": "str_index",
                    "pandas_dtype": PandasDtype.String,
                    "nullable": False,
                    "checks": None,
                },
            ],
        ],
        [
            pd.MultiIndex.from_arrays(
                [[10, 11, 12], pd.Series(["a", "b", "c"], dtype="category")],
                names=["int_index", "str_index"],
            ),
            [
                {
                    "name": "int_index",
                    "pandas_dtype": PandasDtype.Int,
                    "nullable": False,
                    "checks": {
                        "greater_than_or_equal_to": 10,
                        "less_than_or_equal_to": 12,
                    },
                },
                {
                    "name": "str_index",
                    "pandas_dtype": PandasDtype.Category,
                    "nullable": False,
                    "checks": {"isin": ["a", "b", "c"]},
                },
            ],
        ],
        # UserWarning cases
        [1, UserWarning],
        ["foo", UserWarning],
        [{"foo": "bar"}, UserWarning],
        [["foo", "bar"], UserWarning],
        [pd.Series(["foo", "bar"]), UserWarning],
        [pd.DataFrame({"column": ["foo", "bar"]}), UserWarning],
    ],
)
def test_infer_index_statistics(index, expectation):
    """Test that index statistics are correctly inferred."""
    if expectation is UserWarning:
        with pytest.warns(UserWarning, match="^index type .+ not recognized"):
            schema_statistics.infer_index_statistics(index)
    else:
        assert schema_statistics.infer_index_statistics(index) == expectation


def test_get_dataframe_schema_statistics():
    """Test that dataframe schema statistics logic is correct."""
    schema = pa.DataFrameSchema(
        columns={
            "int": pa.Column(
                pa.Int,
                checks=[
                    pa.Check.greater_than_or_equal_to(0),
                    pa.Check.less_than_or_equal_to(100),
                ],
                nullable=True,
            ),
            "float": pa.Column(
                pa.Float,
                checks=[
                    pa.Check.greater_than_or_equal_to(50),
                    pa.Check.less_than_or_equal_to(100),
                ],
            ),
            "str": pa.Column(
                pa.String, checks=[pa.Check.isin(["foo", "bar", "baz"])]
            ),
        },
        index=pa.Index(
            pa.Int,
            checks=pa.Check.greater_than_or_equal_to(0),
            nullable=False,
            name="int_index",
        ),
    )
    expectation = {
        "checks": None,
        "columns": {
            "int": {
                "pandas_dtype": pa.Int,
                "checks": {
                    "greater_than_or_equal_to": {"min_value": 0},
                    "less_than_or_equal_to": {"max_value": 100},
                },
                "nullable": True,
                "allow_duplicates": True,
                "coerce": False,
                "required": True,
                "regex": False,
            },
            "float": {
                "pandas_dtype": pa.Float,
                "checks": {
                    "greater_than_or_equal_to": {"min_value": 50},
                    "less_than_or_equal_to": {"max_value": 100},
                },
                "nullable": False,
                "allow_duplicates": True,
                "coerce": False,
                "required": True,
                "regex": False,
            },
            "str": {
                "pandas_dtype": pa.String,
                "checks": {"isin": {"allowed_values": ["foo", "bar", "baz"]}},
                "nullable": False,
                "allow_duplicates": True,
                "coerce": False,
                "required": True,
                "regex": False,
            },
        },
        "index": [
            {
                "pandas_dtype": pa.Int,
                "checks": {"greater_than_or_equal_to": {"min_value": 0}},
                "nullable": False,
                "coerce": False,
                "name": "int_index",
            }
        ],
        "coerce": False,
    }
    statistics = schema_statistics.get_dataframe_schema_statistics(schema)
    assert statistics == expectation


def test_get_series_schema_statistics():
    """Test that series schema statistics logic is correct."""
    schema = pa.SeriesSchema(
        pa.Int,
        nullable=False,
        checks=[
            pa.Check.greater_than_or_equal_to(0),
            pa.Check.less_than_or_equal_to(100),
        ],
    )
    statistics = schema_statistics.get_series_schema_statistics(schema)
    assert statistics == {
        "pandas_dtype": pa.Int,
        "nullable": False,
        "checks": {
            "greater_than_or_equal_to": {"min_value": 0},
            "less_than_or_equal_to": {"max_value": 100},
        },
        "name": None,
        "coerce": False,
    }


@pytest.mark.parametrize(
    "index_schema_component, expectation",
    [
        [
            pa.Index(
                pa.Int,
                checks=[
                    pa.Check.greater_than_or_equal_to(10),
                    pa.Check.less_than_or_equal_to(20),
                ],
                nullable=False,
                name="int_index",
            ),
            [
                {
                    "pandas_dtype": pa.Int,
                    "nullable": False,
                    "checks": {
                        "greater_than_or_equal_to": {"min_value": 10},
                        "less_than_or_equal_to": {"max_value": 20},
                    },
                    "name": "int_index",
                    "coerce": False,
                }
            ],
        ]
    ],
)
def test_get_index_schema_statistics(index_schema_component, expectation):
    """Test that index schema statistics logic is correct."""
    statistics = schema_statistics.get_index_schema_statistics(
        index_schema_component
    )
    assert statistics == expectation


@pytest.mark.parametrize(
    "checks, expectation",
    [
        *[
            [[check], {check.name: check.statistics}]
            for check in [
                pa.Check.greater_than(1),
                pa.Check.less_than(1),
                pa.Check.in_range(1, 3),
                pa.Check.equal_to(1),
                pa.Check.not_equal_to(1),
                pa.Check.notin([1, 2, 3]),
                pa.Check.str_matches("foobar"),
                pa.Check.str_contains("foobar"),
                pa.Check.str_startswith("foobar"),
                pa.Check.str_endswith("foobar"),
                pa.Check.str_length(5, 10),
            ]
        ],
        # multiple checks at once
        [
            [
                pa.Check.greater_than_or_equal_to(10),
                pa.Check.less_than_or_equal_to(50),
                pa.Check.isin([10, 20, 30, 40, 50]),
            ],
            {
                "greater_than_or_equal_to": {"min_value": 10},
                "less_than_or_equal_to": {"max_value": 50},
                "isin": {"allowed_values": [10, 20, 30, 40, 50]},
            },
        ],
        # incompatible checks
        *[
            [
                [
                    pa.Check.greater_than_or_equal_to(min_value),
                    pa.Check.less_than_or_equal_to(max_value),
                ],
                ValueError,
            ]
            for min_value, max_value in [
                (5, 1),
                (10, 1),
                (100, 10),
                (1000, 100),
            ]
        ],
    ],
)
def test_parse_checks_and_statistics_roundtrip(checks, expectation):
    """
    Test that parse checks correctly obtain statistics from checks and
    vice-versa.
    """
    if expectation is ValueError:
        with pytest.raises(ValueError):
            schema_statistics.parse_checks(checks)
        return
    assert schema_statistics.parse_checks(checks) == expectation

    check_statistics = {check.name: check.statistics for check in checks}
    check_list = schema_statistics.parse_check_statistics(check_statistics)
    assert set(check_list) == set(checks)


def test_parse_checks_and_statistics_no_param(extra_registered_checks):
    """Ensure that an edge case where a check does not have parameters is appropriately handled."""

    checks = [pa.Check.no_param_check()]
    expectation = {"no_param_check": {}}
    assert schema_statistics.parse_checks(checks) == expectation

    check_statistics = {check.name: check.statistics for check in checks}
    check_list = schema_statistics.parse_check_statistics(check_statistics)
    assert set(check_list) == set(checks)
