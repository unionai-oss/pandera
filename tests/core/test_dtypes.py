"""Tests a variety of python and pandas dtypes, and tests some specific
coercion examples."""
# pylint doesn't know about __init__ generated with dataclass
# pylint:disable=unexpected-keyword-arg,no-value-for-parameter
import dataclasses
import datetime
import inspect
from decimal import Decimal
from typing import Any, Dict, List, Tuple

import hypothesis
import numpy as np
import pandas as pd
import pytest
from _pytest.mark.structures import ParameterSet
from _pytest.python import Metafunc
from hypothesis import strategies as st

import pandera as pa
from pandera.engines import pandas_engine

# List dtype classes and associated pandas alias,
# except for parameterizable dtypes that should also list examples of instances.
int_dtypes = {
    int: "int",
    pa.Int: "int64",
    pa.Int8: "int8",
    pa.Int16: "int16",
    pa.Int32: "int32",
    pa.Int64: "int64",
    np.int8: "int8",
    np.int16: "int16",
    np.int32: "int32",
    np.int64: "int64",
}


nullable_int_dtypes = {
    pandas_engine.Int8: "Int8",
    pandas_engine.Int16: "Int16",
    pandas_engine.Int32: "Int32",
    pandas_engine.Int64: "Int64",
}

uint_dtypes = {
    pa.UInt: "uint64",
    pa.UInt8: "uint8",
    pa.UInt16: "uint16",
    pa.UInt32: "uint32",
    pa.UInt64: "uint64",
    np.uint8: "uint8",
    np.uint16: "uint16",
    np.uint32: "uint32",
    np.uint64: "uint64",
}

nullable_uint_dtypes = {
    pandas_engine.UInt8: "UInt8",
    pandas_engine.UInt16: "UInt16",
    pandas_engine.UInt32: "UInt32",
    pandas_engine.UInt64: "UInt64",
}

float_dtypes = {
    float: "float",
    pa.Float: "float64",
    pa.Float16: "float16",
    pa.Float32: "float32",
    pa.Float64: "float64",
    pa.Float128: "float128",
    np.float16: "float16",
    np.float32: "float32",
    np.float64: "float64",
    np.float128: "float128",
}

complex_dtypes = {
    complex: "complex",
    pa.Complex: "complex128",
    pa.Complex64: "complex64",
    pa.Complex128: "complex128",
}

boolean_dtypes = {bool: "bool", pa.Bool: "bool", np.bool_: "bool"}
nullable_boolean_dtypes = {pd.BooleanDtype: "boolean", pa.BOOL: "boolean"}

string_dtypes = {
    str: "str",
    pa.String: "str",
    np.str_: "str",
}
nullable_string_dtypes = {pd.StringDtype: "string"}

object_dtypes = {object: "object", np.object_: "object"}

category_dtypes = {
    pa.Category: "category",
    pa.Category(["A", "B"], ordered=True): pd.CategoricalDtype(
        ["A", "B"], ordered=True
    ),
    pd.CategoricalDtype(["A", "B"], ordered=True): pd.CategoricalDtype(
        ["A", "B"], ordered=True
    ),
}

timestamp_dtypes = {
    datetime.datetime: "datetime64[ns]",
    np.datetime64: "datetime64[ns]",
    pa.Timestamp: "datetime64[ns]",
    pd.DatetimeTZDtype(tz="CET"): "datetime64[ns, CET]",
    pandas_engine.DateTime: "datetime64[ns]",
    pandas_engine.DateTime(unit="ns", tz="CET"): "datetime64[ns, CET]",  # type: ignore
}

timedelta_dtypes = {
    datetime.timedelta: "timedelta64",
    datetime.timedelta: "timedelta64",
    np.timedelta64: "timedelta64",
    pd.Timedelta: "timedelta64",
    pa.Timedelta: "timedelta64",
}

period_dtypes = {pd.PeriodDtype(freq="D"): "period[D]"}
# Series.astype does not accept a string alias for SparseDtype.
sparse_dtypes = {
    pd.SparseDtype: pd.SparseDtype(),
    pd.SparseDtype(np.float64): pd.SparseDtype(np.float64),
}
interval_dtypes = {pd.IntervalDtype(subtype=np.int64): "interval[int64]"}

dtype_fixtures: List[Tuple[Dict, List]] = [
    (int_dtypes, [-1]),
    (nullable_int_dtypes, [-1, None]),
    (uint_dtypes, [1]),
    (nullable_uint_dtypes, [1, None]),
    (float_dtypes, [1.0]),
    (complex_dtypes, [complex(1)]),
    (boolean_dtypes, [True, False]),
    (nullable_boolean_dtypes, [True, None]),
    (string_dtypes, ["A", "B"]),
    (object_dtypes, ["A", "B"]),
    (nullable_string_dtypes, [1, 2, None]),
    (category_dtypes, [1, 2, None]),
    (
        timestamp_dtypes,
        pd.to_datetime(["2019/01/01", "2018/05/21"]).to_series(),
    ),
    (
        period_dtypes,
        pd.to_datetime(["2019/01/01", "2018/05/21"]).to_period("D").to_series(),
    ),
    (sparse_dtypes, pd.Series([1, None], dtype=pd.SparseDtype(float))),
    (interval_dtypes, pd.interval_range(-10.0, 10.0).to_series()),
]


def pretty_param(*values: Any, **kw: Any) -> ParameterSet:
    """Return a pytest parameter with a human-readable id."""
    id_ = kw.pop("id", None)
    if not id_:
        id_ = "-".join(
            f"{val.__module__}.{val.__name__}"
            if inspect.isclass(val)
            else repr(val)
            for val in values
        )
    return pytest.param(*values, id=id_, **kw)


def pytest_generate_tests(metafunc: Metafunc) -> None:
    """Inject `dtype`, `data_type` (filter pandera DataTypes), `alias`, `data`
    fixtures from `dtype_fixtures`.
    """
    fixtures = [
        fixture
        for fixture in ("data_type", "dtype", "pd_dtype", "data")
        if fixture in metafunc.fixturenames
    ]
    arg_names = ",".join(fixtures)

    if arg_names:
        arg_values = []
        for dtypes, data in dtype_fixtures:
            for dtype, pd_dtype in dtypes.items():
                if "data_type" in fixtures and not (
                    isinstance(dtype, pa.DataType)
                    or (
                        inspect.isclass(dtype)
                        and issubclass(dtype, pa.DataType)
                    )
                ):
                    # not a pa.DataType class or instance
                    continue

                params = [dtype]
                if "pd_dtype" in fixtures:
                    params.append(pd_dtype)
                if "data" in fixtures:
                    params.append(data)
                arg_values.append(pretty_param(*params))

        metafunc.parametrize(arg_names, arg_values)


def test_datatype_init(data_type: Any):
    """Test that a default pa.DataType can be constructed."""
    if not inspect.isclass(data_type):
        pytest.skip(
            "test_datatype_init tests pa.DataType classes, not instances."
        )
    assert isinstance(data_type(), pa.DataType)


def test_datatype_alias(data_type: Any, pd_dtype: Any):
    """Test that a default pa.DataType can be constructed."""
    assert str(pandas_engine.Engine.dtype(data_type)) == str(pd_dtype)


def test_frozen_datatype(data_type: Any):
    """Test that pa.DataType instances are immutable."""
    data_type = data_type() if inspect.isclass(data_type) else data_type
    with pytest.raises(dataclasses.FrozenInstanceError):
        data_type.foo = 1


def test_invalid_pandas_extension_dtype():
    """Test that an invalid dtype is rejected."""
    with pytest.raises(TypeError):
        pandas_engine.Engine.dtype(
            pd.PeriodDtype
        )  # PerioDtype has required parameters


def test_check_equivalent(dtype: Any, pd_dtype: Any):
    """Test that a pandas-compatible dtype can be validated by check()."""
    actual_dtype = pandas_engine.Engine.dtype(pd_dtype)
    expected_dtype = pandas_engine.Engine.dtype(dtype)
    assert actual_dtype.check(expected_dtype)


def test_check_not_equivalent(dtype: Any):
    """Test that check() rejects non-equivalent dtypes."""
    if str(pandas_engine.Engine.dtype(dtype)) == "object":
        actual_dtype = pandas_engine.Engine.dtype(int)
    else:
        actual_dtype = pandas_engine.Engine.dtype(object)
    expected_dtype = pandas_engine.Engine.dtype(dtype)
    assert actual_dtype.check(expected_dtype) is False


def test_coerce_no_cast(dtype: Any, pd_dtype: Any, data: List[Any]):
    """Test that dtypes can be coerced without casting."""
    expected_dtype = pandas_engine.Engine.dtype(dtype)
    series = pd.Series(data, dtype=pd_dtype)
    coerced_series = expected_dtype.coerce(series)

    assert series.equals(coerced_series)
    assert expected_dtype.check(
        pandas_engine.Engine.dtype(coerced_series.dtype)
    )

    df = pd.DataFrame({"col": data}, dtype=pd_dtype)
    coerced_df = expected_dtype.coerce(df)

    assert df.equals(coerced_df)
    assert expected_dtype.check(
        pandas_engine.Engine.dtype(coerced_df["col"].dtype)
    )


def _flatten_dtypes_dict(*dtype_kinds):
    return [
        (datatype, pd_dtype)
        for dtype_kind in dtype_kinds
        for datatype, pd_dtype in dtype_kind.items()
    ]


numeric_dtypes = _flatten_dtypes_dict(
    int_dtypes,
    uint_dtypes,
    float_dtypes,
    complex_dtypes,
    boolean_dtypes,
)

nullable_numeric_dtypes = _flatten_dtypes_dict(
    nullable_int_dtypes,
    nullable_uint_dtypes,
    nullable_boolean_dtypes,
)

nominal_dtypes = _flatten_dtypes_dict(
    string_dtypes,
    nullable_string_dtypes,
    category_dtypes,
)


@pytest.mark.parametrize(
    "dtypes, examples",
    [
        (numeric_dtypes, [1]),
        (nullable_numeric_dtypes, [1, None]),
        (nominal_dtypes, ["A", "B"]),
    ],
)
@hypothesis.given(st.data())
def test_coerce_cast(dtypes, examples, data):
    """Test that dtypes can be coerced with casting."""
    _, from_pd_dtype = data.draw(st.sampled_from(dtypes))
    to_datatype, _ = data.draw(st.sampled_from(dtypes))

    expected_dtype = pandas_engine.Engine.dtype(to_datatype)

    series = pd.Series(examples, dtype=from_pd_dtype)
    coerced_dtype = expected_dtype.coerce(series).dtype
    assert expected_dtype.check(pandas_engine.Engine.dtype(coerced_dtype))

    df = pd.DataFrame({"col": examples}, dtype=from_pd_dtype)
    coerced_dtype = expected_dtype.coerce(df)["col"].dtype
    assert expected_dtype.check(pandas_engine.Engine.dtype(coerced_dtype))


def test_coerce_string():
    """Test that strings can be coerced."""
    data = pd.Series([1, None], dtype="Int32")
    coerced = pandas_engine.Engine.dtype(str).coerce(data).to_list()
    assert isinstance(coerced[0], str)
    assert pd.isna(coerced[1])


def test_default_numeric_dtypes():
    """Test that default numeric dtypes int, float and complex are consistent."""
    default_int_dtype = pd.Series([1], dtype=int).dtype
    assert (
        pandas_engine.Engine.dtype(default_int_dtype)
        == pandas_engine.Engine.dtype(int)
        == pandas_engine.Engine.dtype("int")
    )

    default_float_dtype = pd.Series([1], dtype=float).dtype
    assert (
        pandas_engine.Engine.dtype(default_float_dtype)
        == pandas_engine.Engine.dtype(float)
        == pandas_engine.Engine.dtype("float")
    )

    default_complex_dtype = pd.Series([1], dtype=complex).dtype
    assert (
        pandas_engine.Engine.dtype(default_complex_dtype)
        == pandas_engine.Engine.dtype(complex)
        == pandas_engine.Engine.dtype("complex")
    )


@pytest.mark.parametrize(
    "examples",
    [
        pretty_param(param)
        for param in [
            ["A", "B"],  # string
            [b"foo", b"bar"],  # bytes
            [1, 2, 3],  # integer
            ["a", datetime.date(2013, 1, 1)],  # mixed
            ["a", 1],  # mixed-integer
            [1, 2, 3.5, "foo"],  # mixed-integer-float
            [1.0, 2.0, 3.5],  # floating
            [Decimal(1), Decimal(2.0)],  # decimal
            [pd.Timestamp("20130101")],  # datetime
            [datetime.date(2013, 1, 1)],  # date
            [datetime.timedelta(0, 1, 1)],  # timedelta
            pd.Series(list("aabc")).astype("category"),  # categorical
            [Decimal(1), Decimal(2.0)],  # decimal
        ]
    ],
)
def test_inferred_dtype(examples: pd.Series):
    """Test compatibility with pd.api.types.infer_dtype's outputs."""
    alias = pd.api.types.infer_dtype(examples)
    if "mixed" in alias or alias in ("date", "string"):
        # infer_dtype returns "string", "date"
        # whereas a Series will default to a "np.object_" dtype
        inferred_datatype = pandas_engine.Engine.dtype(object)
    else:
        inferred_datatype = pandas_engine.Engine.dtype(alias)
    actual_dtype = pandas_engine.Engine.dtype(pd.Series(examples).dtype)
    assert actual_dtype.check(inferred_datatype)


@pytest.mark.parametrize(
    "int_dtype, expected",
    [(dtype, True) for dtype in (*int_dtypes, *nullable_int_dtypes)]
    + [("string", False)],
)
def test_is_int(int_dtype: Any, expected: bool):
    """Test is_int."""
    pandera_dtype = pandas_engine.Engine.dtype(int_dtype)
    assert pa.dtypes_.is_int(pandera_dtype) == expected


@pytest.mark.parametrize(
    "uint_dtype, expected",
    [(dtype, True) for dtype in (*uint_dtypes, *nullable_uint_dtypes)]
    + [("string", False)],
)
def test_is_uint(uint_dtype: Any, expected: bool):
    """Test is_uint."""
    pandera_dtype = pandas_engine.Engine.dtype(uint_dtype)
    assert pa.dtypes_.is_uint(pandera_dtype) == expected


@pytest.mark.parametrize(
    "float_dtype, expected",
    [(dtype, True) for dtype in float_dtypes] + [("string", False)],
)
def test_is_float(float_dtype: Any, expected: bool):
    """Test is_float."""
    pandera_dtype = pandas_engine.Engine.dtype(float_dtype)
    assert pa.dtypes_.is_float(pandera_dtype) == expected


@pytest.mark.parametrize(
    "complex_dtype, expected",
    [(dtype, True) for dtype in complex_dtypes]
    + [("string", False)],  # type: ignore
)
def test_is_complex(complex_dtype: Any, expected: bool):
    """Test is_complex."""
    pandera_dtype = pandas_engine.Engine.dtype(complex_dtype)
    assert pa.dtypes_.is_complex(pandera_dtype) == expected


@pytest.mark.parametrize(
    "bool_dtype, expected",
    [(dtype, True) for dtype in (*boolean_dtypes, *nullable_boolean_dtypes)]
    + [("string", False)],
)
def test_is_bool(bool_dtype: Any, expected: bool):
    """Test is_bool."""
    pandera_dtype = pandas_engine.Engine.dtype(bool_dtype)
    assert pa.dtypes_.is_bool(pandera_dtype) == expected


@pytest.mark.parametrize(
    "string_dtype, expected",
    [(dtype, True) for dtype in string_dtypes] + [("int", False)],
)
def test_is_string(string_dtype: Any, expected: bool):
    """Test is_string."""
    pandera_dtype = pandas_engine.Engine.dtype(string_dtype)
    assert pa.dtypes_.is_string(pandera_dtype) == expected


@pytest.mark.parametrize(
    "category_dtype, expected",
    [(dtype, True) for dtype in category_dtypes] + [("string", False)],
)
def test_is_category(category_dtype: Any, expected: bool):
    """Test is_category."""
    pandera_dtype = pandas_engine.Engine.dtype(category_dtype)
    assert pa.dtypes_.is_category(pandera_dtype) == expected


@pytest.mark.parametrize(
    "datetime_dtype, expected",
    [(dtype, True) for dtype in timestamp_dtypes] + [("string", False)],
)
def test_is_datetime(datetime_dtype: Any, expected: bool):
    """Test is_datetime."""
    pandera_dtype = pandas_engine.Engine.dtype(datetime_dtype)
    assert pa.dtypes_.is_datetime(pandera_dtype) == expected


@pytest.mark.parametrize(
    "timedelta_dtype, expected",
    [(dtype, True) for dtype in timedelta_dtypes] + [("string", False)],
)
def test_is_timedelta(timedelta_dtype: Any, expected: bool):
    """Test is_timedelta."""
    pandera_dtype = pandas_engine.Engine.dtype(timedelta_dtype)
    assert pa.dtypes_.is_timedelta(pandera_dtype) == expected


@pytest.mark.parametrize(
    "numeric_dtype, expected",
    [(dtype, True) for dtype, _ in numeric_dtypes] + [("string", False)],
)
def test_is_numeric(numeric_dtype: Any, expected: bool):
    """Test is_timedelta."""
    pandera_dtype = pandas_engine.Engine.dtype(numeric_dtype)
    assert pa.dtypes_.is_numeric(pandera_dtype) == expected
