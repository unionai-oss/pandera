"""Tests a variety of python and pandas dtypes, and tests some specific
coercion examples."""
import dataclasses
import inspect
from decimal import Decimal
from typing import Any, List

import hypothesis
import numpy as np
import pandas as pd
import pytest
from _pytest.mark.structures import ParameterSet
from _pytest.python import Metafunc
from hypothesis import strategies as st
from hypothesis.strategies._internal.strategies import one_of
from packaging import version

import pandera as pa
from pandera.dtypes_ import DataType
from pandera.engines.pandas_engine import *

PANDAS_VERSION = version.parse(pd.__version__)


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
    PandasInt8: "Int8",
    PandasInt16: "Int16",
    PandasInt32: "Int32",
    PandasInt64: "Int64",
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
    PandasUInt8: "UInt8",
    PandasUInt16: "UInt16",
    PandasUInt32: "UInt32",
    PandasUInt64: "UInt64",
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
    PandasDateTime: "datetime64[ns]",
    PandasDateTime(unit="ns", tz="CET"): "datetime64[ns, CET]",
}

timedelta_dtypes = {
    datetime.timedelta: "timedelta64",
    datetime.timedelta: "timedelta64",
    np.timedelta64: "timedelta64",
    pd.Timedelta: "timedelta64",
    Timedelta: "timedelta64",
}

period_dtypes = {pd.PeriodDtype(freq="D"): "period[D]"}
# Series.astype does not accept a string alias for SparseDtype.
sparse_dtypes = {
    pd.SparseDtype: pd.SparseDtype(),
    pd.SparseDtype(np.float64): pd.SparseDtype(np.float64),
}
interval_dtypes = {pd.IntervalDtype(subtype=np.int64): "interval[int64]"}

dtype_fixtures = [
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
        pd.to_datetime(["2019/01/01", "2018/05/21"])
        .to_period("D")
        .to_series(),
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
    """Inject dtype, alias, data fixtures from `dtype_fixtures`.

    Filter pandera.dtypes.DataType classes if the test name contains "datatype".
    """
    fixtures = [
        fixture
        for fixture in ("dtype", "pd_dtype", "data")
        if fixture in metafunc.fixturenames
    ]
    arg_names = ",".join(fixtures)

    if arg_names:
        arg_values = []
        for dtypes, data in dtype_fixtures:
            for dtype, pd_dtype in dtypes.items():
                if "datatype" in metafunc.function.__name__ and not (
                    isinstance(dtype, DataType)
                    or (inspect.isclass(dtype) and issubclass(dtype, DataType))
                ):
                    # not a DataType class or instance
                    continue

                params = [dtype]
                if "pd_dtype" in fixtures:
                    params.append(pd_dtype)
                if "data" in fixtures:
                    params.append(data)
                arg_values.append(pretty_param(*params))

        metafunc.parametrize(arg_names, arg_values)


def test_datatype_init(dtype: Any):
    """Test that a default DataType can be constructed."""
    if not inspect.isclass(dtype):
        pytest.skip(
            "test_datatype_init tests DataType classes, not instances."
        )
    assert isinstance(dtype(), DataType)


def test_datatype_alias(dtype: Any, pd_dtype: Any):
    """Test that a default DataType can be constructed."""
    data_type = dtype() if inspect.isclass(dtype) else dtype
    assert str(PandasEngine.dtype(dtype)) == str(pd_dtype)


def test_frozen_datatype(dtype: Any):
    """Test that DataType instances are immutable."""
    data_type = dtype() if inspect.isclass(dtype) else dtype
    with pytest.raises(dataclasses.FrozenInstanceError):
        data_type.foo = 1


def test_invalid_pandas_extension_dtype():
    with pytest.raises(TypeError):
        PandasEngine.dtype(
            pd.PeriodDtype
        )  # PerioDtype has required parameters


def test_check_equivalent(dtype: Any, pd_dtype: Any):
    """Test that a pandas-compatible dtype can be validated by check()."""
    actual_dtype = PandasEngine.dtype(pd_dtype)
    expected_dtype = PandasEngine.dtype(dtype)
    assert actual_dtype.check(expected_dtype)


def test_check_not_equivalent(dtype: Any):
    """Test that check() rejects non-equivalent dtypes."""
    if str(PandasEngine.dtype(dtype)) == "object":
        actual_dtype = PandasEngine.dtype(int)
    else:
        actual_dtype = PandasEngine.dtype(object)
    expected_dtype = PandasEngine.dtype(dtype)
    assert actual_dtype.check(expected_dtype) is False


def test_coerce_no_cast(dtype: Any, pd_dtype: Any, data: List[Any]):
    """Test that dtypes can be coerced without casting."""
    expected_dtype = PandasEngine.dtype(dtype)
    print(pd_dtype)
    series = pd.Series(data, dtype=pd_dtype)
    coerced_series = expected_dtype.coerce(series)
    assert series.equals(coerced_series)
    print(expected_dtype)
    print(series)
    print(coerced_series)
    print(coerced_series.dtype)
    print(PandasEngine.dtype(coerced_series.dtype))
    assert expected_dtype.check(PandasEngine.dtype(coerced_series.dtype))

    df = pd.DataFrame({"col": data}, dtype=pd_dtype)
    coerced_df = expected_dtype.coerce(df)
    assert df.equals(coerced_df)
    assert expected_dtype.check(PandasEngine.dtype(coerced_df["col"].dtype))


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

    expected_dtype = PandasEngine.dtype(to_datatype)

    series = pd.Series(examples, dtype=from_pd_dtype)
    coerced_dtype = expected_dtype.coerce(series).dtype
    assert expected_dtype.check(PandasEngine.dtype(coerced_dtype))

    df = pd.DataFrame({"col": examples}, dtype=from_pd_dtype)
    coerced_dtype = expected_dtype.coerce(df)["col"].dtype
    assert expected_dtype.check(PandasEngine.dtype(coerced_dtype))


def test_coerce_string():
    """Test that strings can be coerced."""
    data = pd.Series([1, None], dtype="Int32")
    coerced = PandasEngine.dtype(str).coerce(data).to_list()
    assert isinstance(coerced[0], str)
    assert pd.isna(coerced[1])


def test_default_numeric_dtypes():
    """Test that default numeric dtypes int, float and complex are consistent."""
    default_int_dtype = pd.Series([1], dtype=int).dtype
    assert (
        PandasEngine.dtype(default_int_dtype)
        == PandasEngine.dtype(int)
        == PandasEngine.dtype("int")
    )

    default_float_dtype = pd.Series([1], dtype=float).dtype
    assert (
        PandasEngine.dtype(default_float_dtype)
        == PandasEngine.dtype(float)
        == PandasEngine.dtype("float")
    )

    default_complex_dtype = pd.Series([1], dtype=complex).dtype
    assert (
        PandasEngine.dtype(default_complex_dtype)
        == PandasEngine.dtype(complex)
        == PandasEngine.dtype("complex")
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
    alias = pd.api.types.infer_dtype(examples)
    if "mixed" in alias or alias in ("date", "string"):
        # infer_dtype returns "string", "date"
        # whereas a Series will default to a "np.object_" dtype
        inferred_datatype = PandasEngine.dtype(object)
    else:
        inferred_datatype = PandasEngine.dtype(alias)
    actual_dtype = PandasEngine.dtype(pd.Series(examples).dtype)
    assert actual_dtype.check(inferred_datatype)
