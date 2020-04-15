# pylint: disable=W0212
"""Unit tests for schema inference module."""

import pandas as pd
import pytest

import pandera as pa
from pandera import schema_inference
from pandera import dtypes, PandasDtype


DEFAULT_INT = PandasDtype.from_str_alias(dtypes._DEFAULT_PANDAS_INT_TYPE)
DEFAULT_FLOAT = PandasDtype.from_str_alias(dtypes._DEFAULT_PANDAS_FLOAT_TYPE)


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
            "float": [1., 2., 3.],
            "boolean": [True, False, True],
            "string": ["a", "b", "c"],
            "datetime": pd.to_datetime(["20180101", "20180102", "20180103"]),
        },
        index=index,
    )

    if nullable:
        df.iloc[0, :] = None

    return df


@pytest.mark.parametrize("pandas_obj, expectation", [
    [pd.DataFrame({"col": [1, 2, 3]}), pa.DataFrameSchema],
    [pd.Series([1, 2, 3]), pa.SeriesSchema],

    # error cases
    [int, TypeError],
    [pd.Index([1, 2, 3]), TypeError],
    ["foobar", TypeError],
    [1, TypeError],
    [[1, 2, 3], TypeError],
    [{"key": "value"}, TypeError],
])
def test_infer_schema(pandas_obj, expectation):
    """Test that convenience function correctly infers dataframe or series."""
    if expectation is TypeError:
        with pytest.raises(TypeError, match="^pandas_obj type not recognized"):
            schema_inference.infer_schema(pandas_obj)
    else:
        assert isinstance(
            schema_inference.infer_schema(pandas_obj), expectation
        )


@pytest.mark.parametrize("multi_index, nullable", [
    [False, False],
    [True, False],
    [False, True],
    [True, True],
])
def test_infer_dataframe_statistics(multi_index, nullable):
    """Test dataframe statistics are correctly inferred."""
    dataframe = _create_dataframe(multi_index, nullable)
    statistics = schema_inference.infer_dataframe_statistics(dataframe)
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
                stat_indices,
                ["int_index", "str_index"],
                [DEFAULT_INT, pa.String]):
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


@pytest.mark.parametrize("check_stats, expectation", [
    [{"min": 1}, [pa.Check.greater_than_or_equal_to(1)]],
    [{"max": 10}, [pa.Check.less_than_or_equal_to(10)]],
    [{"levels": ["a", "b", "c"]}, [pa.Check.isin(["a", "b", "c"])]],
    [
        {"min": 1, "max": 10},
        [pa.Check.greater_than_or_equal_to(1),
         pa.Check.less_than_or_equal_to(10)]
    ],
    [{}, None],
])
def test_parse_check_statistics(check_stats, expectation):
    """Test that Checks are correctly parsed from check statistics."""
    assert schema_inference._parse_check_statistics(check_stats) == expectation


def test_infer_dataframe_schema():
    """Test dataframe schema is correctly inferred."""
    dataframe = _create_dataframe()
    schema = schema_inference.infer_dataframe_schema(dataframe)
    assert isinstance(schema, pa.DataFrameSchema)

    with pytest.warns(
            UserWarning,
            match="^This .+ is an inferred schema that hasn't been modified"):
        schema.validate(dataframe)

    # modifying an inferred schema should set _is_inferred to False
    schema_with_added_cols = schema.add_columns(
        {"foo": pa.Column(pa.String)})
    assert schema._is_inferred
    assert not schema_with_added_cols._is_inferred
    assert isinstance(
        schema_with_added_cols.validate(dataframe.assign(foo="a")),
        pd.DataFrame)

    schema_with_removed_cols = schema.remove_columns(["int"])
    assert schema._is_inferred
    assert not schema_with_removed_cols._is_inferred
    assert isinstance(
        schema_with_removed_cols.validate(dataframe.drop("int", axis=1)),
        pd.DataFrame)


@pytest.mark.parametrize("series, expectation", [
    *[
        [
            pd.Series([1, 2, 3], dtype=dtype.str_alias), {
                "pandas_dtype": dtype, "nullable": False,
                "checks": {"min": 1, "max": 3},
                "name": None,
            }
        ]
        for dtype in schema_inference.NUMERIC_DTYPES
    ],
    [
        pd.Series(["a", "b", "c", "a"], dtype="category"), {
            "pandas_dtype": pa.Category, "nullable": False,
            "checks": {"levels": ["a", "b", "c"]},
            "name": None,
        }
    ],
    [
        pd.Series(["a", "b", "c", "a"], name="str_series"), {
            "pandas_dtype": pa.String, "nullable": False,
            "checks": None, "name": "str_series",
        }
    ],
    [
        pd.Series(pd.to_datetime(["20180101", "20180102", "20180103"])), {
            "pandas_dtype": pa.DateTime, "nullable": False,
            "checks": {
                "min": pd.Timestamp("20180101"),
                "max": pd.Timestamp("20180103")
            },
            "name": None,
        }
    ],
])
def test_infer_series_schema_statistics(series, expectation):
    """Test series statistics are correctly inferred."""
    statistics = schema_inference.infer_series_statistics(series)
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


@pytest.mark.parametrize("null_index, series, expectation", [
    *[
        [
            0, pd.Series([1, 2, 3], dtype=dtype.value), {
                # introducing nans to integer arrays upcasts to float
                "pandas_dtype": DEFAULT_FLOAT, "nullable": True,
                "checks": {"min": 2, "max": 3},
                "name": None,
            }
        ]
        for dtype in INTEGER_TYPES
    ],
    [
        # introducing nans to integer arrays upcasts to float
        0, pd.Series([True, False, True, False]), {
            "pandas_dtype": DEFAULT_FLOAT, "nullable": True,
            "checks": {"min": 0, "max": 1},
            "name": None,
        }
    ],
    [
        0, pd.Series(["a", "b", "c", "a"], dtype="category"), {
            "pandas_dtype": pa.Category, "nullable": True,
            "checks": {"levels": ["a", "b", "c"]},
            "name": None,
        }
    ],
    [
        0, pd.Series(["a", "b", "c", "a"], name="str_series"), {
            "pandas_dtype": pa.String, "nullable": True,
            "checks": None, "name": "str_series",
        }
    ],
    [
        2, pd.Series(pd.to_datetime(["20180101", "20180102", "20180103"])), {
            "pandas_dtype": pa.DateTime, "nullable": True,
            "checks": {
                "min": pd.Timestamp("20180101"),
                "max": pd.Timestamp("20180102")
            },
            "name": None,
        }
    ],
])
def test_infer_nullable_series_schema_statistics(
        null_index, series, expectation):
    """Test nullable series statistics are correctly inferred."""
    series.iloc[null_index] = None
    statistics = schema_inference.infer_series_statistics(series)
    assert statistics == expectation


@pytest.mark.parametrize("series", [
    pd.Series([1, 2, 3]),
    pd.Series([1., 2., 3.]),
    pd.Series([True, False, True]),
    pd.Series(list("abcdefg")),
    pd.Series(list("abcdefg"), dtype="category"),
    pd.Series(pd.to_datetime(["20180101", "20180102", "20180103"])),
])
def test_infer_series_schema(series):
    """Test series schema is correctly inferred."""
    schema = schema_inference.infer_series_schema(series)
    assert isinstance(schema, pa.SeriesSchema)

    with pytest.warns(
            UserWarning,
            match="^This .+ is an inferred schema that hasn't been modified"):
        schema.validate(series)

    # modifying an inferred schema should set _is_inferred to False
    schema_with_new_checks = schema.set_checks(
        [pa.Check(lambda x: x is not None)])
    assert schema._is_inferred
    assert not schema_with_new_checks._is_inferred
    assert isinstance(schema_with_new_checks.validate(series), pd.Series)


@pytest.mark.parametrize("index, expectation", [
    [
        pd.RangeIndex(20), [
            {"name": None, "pandas_dtype": PandasDtype.Int,
             "nullable": False, "checks": {"min": 0, "max": 19}}
        ],
    ],
    [
        pd.Index([1, 2, 3], name="int_index"), [
            {"name": "int_index", "pandas_dtype": PandasDtype.Int,
             "nullable": False, "checks": {"min": 1, "max": 3}}
        ],
    ],
    [
        pd.Index(["foo", "bar", "baz"], name="str_index"), [
            {"name": "str_index", "pandas_dtype": PandasDtype.String,
             "nullable": False, "checks": None},
        ],
    ],
    [
        pd.MultiIndex.from_arrays(
            [[10, 11, 12], pd.Series(["a", "b", "c"], dtype="category")],
            names=["int_index", "str_index"],
        ),
        [
            {"name": "int_index", "pandas_dtype": PandasDtype.Int,
             "nullable": False, "checks": {"min": 10, "max": 12}},
            {"name": "str_index", "pandas_dtype": PandasDtype.Category,
             "nullable": False, "checks": {"levels": ["a", "b", "c"]}}
        ],
    ],

    # UserWarning cases
    [1, UserWarning],
    ["foo", UserWarning],
    [{"foo": "bar"}, UserWarning],
    [["foo", "bar"], UserWarning],
    [pd.Series(["foo", "bar"]), UserWarning],
    [pd.DataFrame({"column": ["foo", "bar"]}), UserWarning],
])
def test_infer_index_statistics(index, expectation):
    """Test that index statistics are correctly inferred."""
    if expectation is UserWarning:
        with pytest.warns(UserWarning, match="^index type .+ not recognized"):
            schema_inference.infer_index_statistics(index)
    else:
        assert schema_inference.infer_index_statistics(index) == expectation
