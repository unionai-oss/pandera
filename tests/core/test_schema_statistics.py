# pylint: disable=W0212
"""Unit tests for inferring statistics of pandas objects."""
import pandas as pd
import pytest

import pandera as pa
from pandera import dtypes, schema_statistics
from pandera.engines import pandas_engine

DEFAULT_FLOAT = pandas_engine.Engine.dtype(float)
DEFAULT_INT = pandas_engine.Engine.dtype(int)

NUMERIC_TYPES = [
    pa.Int(),
    pa.UInt(),
    pa.Float(),
    pa.Complex(),
    pandas_engine.Engine.dtype("Int32"),
    pandas_engine.Engine.dtype("UInt32"),
]
INTEGER_TYPES = [
    dtypes.Int(),
    dtypes.Int8(),
    dtypes.Int16(),
    dtypes.Int32(),
    dtypes.Int64(),
    dtypes.UInt8(),
    dtypes.UInt16(),
    dtypes.UInt32(),
    dtypes.UInt64(),
]


def _create_dataframe(
    multi_index: bool = False, nullable: bool = False
) -> pd.DataFrame:
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
def test_infer_dataframe_statistics(multi_index: bool, nullable: bool) -> None:
    """Test dataframe statistics are correctly inferred."""
    dataframe = _create_dataframe(multi_index, nullable)
    statistics = schema_statistics.infer_dataframe_statistics(dataframe)
    stat_columns = statistics["columns"]

    if pa.pandas_version().release >= (1, 3, 0):
        if nullable:
            assert DEFAULT_FLOAT.check(stat_columns["int"]["dtype"])
        else:
            assert DEFAULT_INT.check(stat_columns["int"]["dtype"])
    else:
        if nullable:
            assert DEFAULT_FLOAT.check(stat_columns["boolean"]["dtype"])
        else:
            assert pandas_engine.Engine.dtype(bool).check(
                stat_columns["boolean"]["dtype"]
            )

    assert DEFAULT_FLOAT.check(stat_columns["float"]["dtype"])
    assert pandas_engine.Engine.dtype(str).check(
        stat_columns["string"]["dtype"]
    )
    assert pandas_engine.Engine.dtype(pa.DateTime).check(
        stat_columns["datetime"]["dtype"]
    )

    if multi_index:
        stat_indices = statistics["index"]
        for stat_index, name, dtype in zip(
            stat_indices,
            ["int_index", "str_index"],
            [DEFAULT_INT, pandas_engine.Engine.dtype(str)],
        ):
            assert stat_index["name"] == name
            assert dtype.check(stat_index["dtype"])
            assert not stat_index["nullable"]
    else:
        stat_index = statistics["index"][0]
        assert stat_index["name"] == "int_index"
        assert stat_index["dtype"] == DEFAULT_INT
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
def test_parse_check_statistics(check_stats, expectation) -> None:
    """Test that Checks are correctly parsed from check statistics."""
    if expectation is None:
        expectation = []
    checks = schema_statistics.parse_check_statistics(check_stats)
    if checks is None:
        checks = []
    assert set(checks) == set(expectation)


def _test_statistics(statistics, expectations):
    if not isinstance(statistics, list):
        statistics = [statistics]
    if not isinstance(expectations, list):
        expectations = [expectations]

    for stats, expectation in zip(statistics, expectations):
        stat_dtype = stats.pop("dtype")
        expectation_dtype = expectation.pop("dtype")

        assert stats == expectation
        assert expectation_dtype.check(stat_dtype)


@pytest.mark.parametrize(
    "series, expectation",
    [
        *[
            [
                pd.Series(
                    [1, 2, 3], dtype=str(pandas_engine.Engine.dtype(data_type))
                ),
                {
                    "dtype": pandas_engine.Engine.dtype(data_type),
                    "nullable": False,
                    "checks": {
                        "greater_than_or_equal_to": 1,
                        "less_than_or_equal_to": 3,
                    },
                    "name": None,
                },
            ]
            for data_type in NUMERIC_TYPES
        ],
        [
            pd.Series(["a", "b", "c", "a"], dtype="category"),
            {
                "dtype": pandas_engine.Engine.dtype(pa.Category),
                "nullable": False,
                "checks": {"isin": ["a", "b", "c"]},
                "name": None,
            },
        ],
        [
            pd.Series(["a", "b", "c", "a"], dtype="string", name="str_series"),
            {
                "dtype": pandas_engine.Engine.dtype("string"),
                "nullable": False,
                "checks": None,
                "name": "str_series",
            },
        ],
        [
            pd.Series(pd.to_datetime(["20180101", "20180102", "20180103"])),
            {
                "dtype": pandas_engine.Engine.dtype(pa.DateTime),
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
def test_infer_series_schema_statistics(series, expectation) -> None:
    """Test series statistics are correctly inferred."""
    statistics = schema_statistics.infer_series_statistics(series)
    _test_statistics(statistics, expectation)


@pytest.mark.parametrize(
    "null_index, series, expectation",
    [
        *[
            [
                0,
                pd.Series([1, 2, 3], dtype=str(data_type)),
                {
                    # introducing nans to integer arrays upcasts to float
                    "dtype": DEFAULT_FLOAT,
                    "nullable": True,
                    "checks": {
                        "greater_than_or_equal_to": 2,
                        "less_than_or_equal_to": 3,
                    },
                    "name": None,
                },
            ]
            for data_type in INTEGER_TYPES
        ],
        [
            # introducing nans to bool arrays upcasts to float except
            # for pandas >= 1.3.0
            0,
            pd.Series([True, False, True, False]),
            {
                "dtype": (
                    pandas_engine.Engine.dtype(pa.BOOL)
                    if pa.PANDAS_1_3_0_PLUS
                    else DEFAULT_FLOAT
                ),
                "nullable": True,
                "checks": (
                    None
                    if pa.PANDAS_1_3_0_PLUS
                    else {
                        "greater_than_or_equal_to": 0,
                        "less_than_or_equal_to": 1,
                    }
                ),
                "name": None,
            },
        ],
        [
            0,
            pd.Series(["a", "b", "c", "a"], dtype="category"),
            {
                "dtype": pandas_engine.Engine.dtype(pa.Category),
                "nullable": True,
                "checks": {"isin": ["a", "b", "c"]},
                "name": None,
            },
        ],
        [
            0,
            pd.Series(["a", "b", "c", "a"], name="str_series"),
            {
                "dtype": pandas_engine.Engine.dtype(str),
                "nullable": True,
                "checks": None,
                "name": "str_series",
            },
        ],
        [
            2,
            pd.Series(pd.to_datetime(["20180101", "20180102", "20180103"])),
            {
                "dtype": pandas_engine.Engine.dtype(pa.DateTime),
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
    _test_statistics(statistics, expectation)


@pytest.mark.parametrize(
    "index, expectation",
    [
        [
            pd.RangeIndex(20),
            [
                {
                    "name": None,
                    "dtype": DEFAULT_INT,
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
                    "dtype": DEFAULT_INT,
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
                    "dtype": pandas_engine.Engine.dtype("object"),
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
                    "dtype": DEFAULT_INT,
                    "nullable": False,
                    "checks": {
                        "greater_than_or_equal_to": 10,
                        "less_than_or_equal_to": 12,
                    },
                },
                {
                    "name": "str_index",
                    "dtype": pandas_engine.Engine.dtype(pa.Category),
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
        _test_statistics(
            schema_statistics.infer_index_statistics(index), expectation
        )


def test_get_dataframe_schema_statistics():
    """Test that dataframe schema statistics logic is correct."""
    schema = pa.DataFrameSchema(
        columns={
            "int": pa.Column(
                int,
                checks=[
                    pa.Check.greater_than_or_equal_to(0),
                    pa.Check.less_than_or_equal_to(100),
                ],
                nullable=True,
            ),
            "float": pa.Column(
                float,
                checks=[
                    pa.Check.greater_than_or_equal_to(50),
                    pa.Check.less_than_or_equal_to(100),
                ],
            ),
            "str": pa.Column(
                str,
                checks=[pa.Check.isin(["foo", "bar", "baz"])],
            ),
        },
        index=pa.Index(
            int,
            checks=pa.Check.greater_than_or_equal_to(0),
            nullable=False,
            name="int_index",
        ),
    )
    expectation = {
        "checks": None,
        "columns": {
            "int": {
                "dtype": DEFAULT_INT,
                "checks": {
                    "greater_than_or_equal_to": {"min_value": 0},
                    "less_than_or_equal_to": {"max_value": 100},
                },
                "nullable": True,
                "unique": False,
                "coerce": False,
                "required": True,
                "regex": False,
            },
            "float": {
                "dtype": DEFAULT_FLOAT,
                "checks": {
                    "greater_than_or_equal_to": {"min_value": 50},
                    "less_than_or_equal_to": {"max_value": 100},
                },
                "nullable": False,
                "unique": False,
                "coerce": False,
                "required": True,
                "regex": False,
            },
            "str": {
                "dtype": pandas_engine.Engine.dtype(str),
                "checks": {"isin": {"allowed_values": ["foo", "bar", "baz"]}},
                "nullable": False,
                "unique": False,
                "coerce": False,
                "required": True,
                "regex": False,
            },
        },
        "index": [
            {
                "dtype": DEFAULT_INT,
                "checks": {"greater_than_or_equal_to": {"min_value": 0}},
                "nullable": False,
                "coerce": False,
                "name": "int_index",
                "unique": False,
            }
        ],
        "coerce": False,
    }
    statistics = schema_statistics.get_dataframe_schema_statistics(schema)
    assert statistics == expectation


def test_get_series_schema_statistics():
    """Test that series schema statistics logic is correct."""
    schema = pa.SeriesSchema(
        int,
        nullable=False,
        checks=[
            pa.Check.greater_than_or_equal_to(0),
            pa.Check.less_than_or_equal_to(100),
        ],
    )
    statistics = schema_statistics.get_series_schema_statistics(schema)
    assert statistics == {
        "dtype": pandas_engine.Engine.dtype(int),
        "nullable": False,
        "checks": {
            "greater_than_or_equal_to": {"min_value": 0},
            "less_than_or_equal_to": {"max_value": 100},
        },
        "name": None,
        "coerce": False,
        "unique": False,
    }


@pytest.mark.parametrize(
    "index_schema_component, expectation",
    [
        [
            pa.Index(
                int,
                checks=[
                    pa.Check.greater_than_or_equal_to(10),
                    pa.Check.less_than_or_equal_to(20),
                ],
                nullable=False,
                name="int_index",
            ),
            [
                {
                    "dtype": pandas_engine.Engine.dtype(int),
                    "nullable": False,
                    "checks": {
                        "greater_than_or_equal_to": {"min_value": 10},
                        "less_than_or_equal_to": {"max_value": 20},
                    },
                    "name": "int_index",
                    "coerce": False,
                    "unique": False,
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
    _test_statistics(statistics, expectation)


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


# pylint: disable=unused-argument
def test_parse_checks_and_statistics_no_param(extra_registered_checks):
    """
    Ensure that an edge case where a check does not have parameters is
    appropriately handled.
    """

    checks = [pa.Check.no_param_check()]
    expectation = {"no_param_check": {}}
    assert schema_statistics.parse_checks(checks) == expectation

    check_statistics = {check.name: check.statistics for check in checks}
    check_list = schema_statistics.parse_check_statistics(check_statistics)
    assert set(check_list) == set(checks)


# pylint: enable=unused-argument
