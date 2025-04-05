"""Tests a variety of python and pandas dtypes, and tests some specific
coercion examples."""

# pylint doesn't know about __init__ generated with dataclass
# pylint:disable=unexpected-keyword-arg,no-value-for-parameter
# pylint:disable=unsubscriptable-object
import dataclasses
import datetime
import inspect
import re
import sys
from decimal import Decimal
from typing import Any, Dict, List, NamedTuple, Tuple

import hypothesis
import numpy as np
import pandas as pd
import pytest
from _pytest.mark.structures import ParameterSet
from _pytest.python import Metafunc
from hypothesis import strategies as st
from pandas import DatetimeTZDtype, to_datetime

import pandera as pa
from pandera.engines import pandas_engine
from pandera.engines.utils import pandas_version
from pandera.system import FLOAT_128_AVAILABLE

# List dtype classes and associated pandas alias,
# except for parameterizable dtypes that should also list examples of
# instances.
from pandera.typing.geopandas import GEOPANDAS_INSTALLED

# register different TypedDict type depending on python version
if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict  # noqa


int_dtypes = {
    int: "int64",
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
    pandas_engine.INT8: "Int8",
    pandas_engine.INT16: "Int16",
    pandas_engine.INT32: "Int32",
    pandas_engine.INT64: "Int64",
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
    pandas_engine.UINT8: "UInt8",
    pandas_engine.UINT16: "UInt16",
    pandas_engine.UINT32: "UInt32",
    pandas_engine.UINT64: "UInt64",
}

float_dtypes = {
    float: "float",
    pa.Float: "float64",
    pa.Float16: "float16",
    pa.Float32: "float32",
    pa.Float64: "float64",
    np.float16: "float16",
    np.float32: "float32",
    np.float64: "float64",
}

complex_dtypes = {
    complex: "complex",
    pa.Complex: "complex128",
    pa.Complex64: "complex64",
    pa.Complex128: "complex128",
}

if FLOAT_128_AVAILABLE:
    float_dtypes.update(
        {
            pa.Float128: "float128",
            np.float128: "float128",
        }
    )
    complex_dtypes.update(
        {
            pa.Complex256: "complex256",
            np.complex256: "complex256",
        }
    )

NULLABLE_FLOAT_DTYPES = None
if pa.PANDAS_1_2_0_PLUS:
    NULLABLE_FLOAT_DTYPES = {
        pandas_engine.FLOAT32: "Float32",
        pandas_engine.FLOAT64: "Float64",
    }

boolean_dtypes = {bool: "bool", pa.Bool: "bool", np.bool_: "bool"}
nullable_boolean_dtypes = {pd.BooleanDtype: "boolean", pa.BOOL: "boolean"}

string_dtypes = {
    str: "str",
    pa.String: "str",
    np.str_: "str",
}

nullable_string_dtypes = {pd.StringDtype: "string"}
if pa.PANDAS_1_3_0_PLUS and pandas_engine.PYARROW_INSTALLED:
    nullable_string_dtypes.update(
        {pd.StringDtype(storage="pyarrow"): "string[pyarrow]"}  # type: ignore
    )

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
    *(
        []
        if NULLABLE_FLOAT_DTYPES is None
        else [(NULLABLE_FLOAT_DTYPES, [1.0, None])]
    ),
    (complex_dtypes, [complex(1)]),
    (boolean_dtypes, [True, False]),
    (nullable_boolean_dtypes, [True, None]),
    (string_dtypes, ["A", "B"]),
    (object_dtypes, ["A", "B"]),
    (nullable_string_dtypes, [1, 2, None]),
    (category_dtypes, [1, 2, None]),
    (  # type: ignore
        timestamp_dtypes,
        pd.to_datetime(["2019/01/01", "2018/05/21"]).to_series(),
    ),
    (  # type: ignore
        period_dtypes,
        pd.to_datetime(["2019/01/01", "2018/05/21"])
        .to_period("D")
        .to_series(),
    ),
    (sparse_dtypes, pd.Series([1, None], dtype=pd.SparseDtype(float))),  # type: ignore
    (interval_dtypes, pd.interval_range(-10.0, 10.0).to_series()),  # type: ignore
]


if GEOPANDAS_INSTALLED:
    from shapely.geometry import Polygon

    # pylint:disable=ungrouped-imports
    from pandera.engines.geopandas_engine import Geometry

    geometry_dtypes = {Geometry: "geometry"}
    dtype_fixtures.append(
        (geometry_dtypes, [Polygon(((0, 0), (0, 1), (1, 1)))])
    )


def pretty_param(*values: Any, **kw: Any) -> ParameterSet:
    """Return a pytest parameter with a human-readable id."""
    id_ = kw.pop("id", None)
    if not id_:
        id_ = "-".join(
            (
                f"{val.__module__}.{val.__name__}"
                if inspect.isclass(val)
                else repr(val)
            )
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

    with pytest.raises(TypeError, match="DataType may not be instantiated"):
        pa.dtypes.DataType()


def test_datatype_call():
    """Test DataType call method."""

    class CustomDataType(pa.dtypes.DataType):
        """Custom data type."""

        def coerce(self, data_container: List[int]) -> List[str]:
            """Convert list of ints into a list of strings."""
            return [str(x) for x in data_container]

    custom_dtype = CustomDataType()
    coerced_data = custom_dtype([1, 2, 3, 4, 5])
    assert coerced_data == ["1", "2", "3", "4", "5"]


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
    if (
        pandas_engine.PYARROW_INSTALLED
        and pandas_engine.PANDAS_2_0_0_PLUS
        and dtype == "string[pyarrow]"
    ):
        pytest.skip("`string[pyarrow]` gets parsed to type `string` by pandas")
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

    if isinstance(pd_dtype, str) and "datetime64" in pd_dtype:
        # handle dtype case
        tz_match = re.match(r"datetime64\[ns, (.+)\]", pd_dtype)
        tz = None if not tz_match else tz_match.group(1)
        if pandas_version().release >= (2, 0, 0):
            series = pd.Series(data, dtype=pd_dtype).dt.tz_localize(tz)
        else:
            series = pd.Series(data, dtype=pd_dtype)  # type: ignore[assignment]
    else:
        series = pd.Series(data, dtype=pd_dtype)  # type: ignore[assignment]

    coerced_series = expected_dtype.coerce(series)

    assert series.equals(coerced_series)
    assert expected_dtype.check(
        pandas_engine.Engine.dtype(coerced_series.dtype)
    )

    df = pd.DataFrame({"col": series})
    coerced_df = expected_dtype.coerce(df)

    assert df.equals(coerced_df)
    assert expected_dtype.check(
        pandas_engine.Engine.dtype(coerced_df["col"].dtype)
    )


def _flatten_dtypesdict(*dtype_kinds):
    return [
        (datatype, pd_dtype)
        for dtype_kind in dtype_kinds
        for datatype, pd_dtype in dtype_kind.items()
    ]


numeric_dtypes = _flatten_dtypesdict(
    int_dtypes,
    uint_dtypes,
    float_dtypes,
    complex_dtypes,
    boolean_dtypes,
)


nullable_numeric_dtypes = _flatten_dtypesdict(
    nullable_int_dtypes,
    nullable_uint_dtypes,
    NULLABLE_FLOAT_DTYPES,
    nullable_boolean_dtypes,
    *([NULLABLE_FLOAT_DTYPES] if NULLABLE_FLOAT_DTYPES else []),
)


nominal_dtypes = _flatten_dtypesdict(
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


dt_examples_to_coerce = [
    "2021-09-01",
    datetime.datetime(2021, 1, 7),
    pd.NaT,
    "not_a_date",
]


@pytest.mark.parametrize(
    "examples, type_, failure_indices",
    [
        (["a", 0, "b"], int, [0, 2]),
        (dt_examples_to_coerce, datetime.datetime, [3]),
        (dt_examples_to_coerce, DatetimeTZDtype(tz="UTC"), [3]),
    ],
)
def test_try_coerce(examples, type_, failure_indices):
    """Test that try_coerce raises a ParseError."""
    data_type = pandas_engine.Engine.dtype(type_)
    data = pd.Series(examples)

    with pytest.raises(pa.errors.ParserError):
        data_type.try_coerce(data)

    try:
        data_type.try_coerce(data)
    except pa.errors.ParserError as exc:
        assert exc.failure_cases["index"].to_list() == failure_indices


@pytest.mark.parametrize(
    "examples, type_, has_tz",
    [
        (
            ["2022-04-30T00:00:00Z", "2022-04-30T00:00:01Z"],
            DatetimeTZDtype(tz="UTC"),
            True,
        ),
        (
            ["2022-04-30T00:00:00Z", "2022-04-30T00:00:01Z"],
            DatetimeTZDtype(tz="CET"),
            True,
        ),
        (
            ["2022-04-30T00:00:00", "2022-04-30T00:00:01"],
            DatetimeTZDtype(tz="UTC"),
            True,
        ),
        (
            ["2022-04-30T00:00:00Z", "2022-04-30T00:00:01Z"],
            pandas_engine.DateTime,
            False,
        ),
        (
            ["2022-04-30T00:00:00", "2022-04-30T00:00:01"],
            pandas_engine.DateTime,
            False,
        ),
    ],
)
@pytest.mark.parametrize("is_index", [True, False])
def test_coerce_dt(examples, type_, has_tz, is_index):
    """Test coercion of Series and Indexes to DateTime with and without Time Zones."""
    data = pd.Index(examples) if is_index else pd.Series(examples)
    data_type = pandas_engine.Engine.dtype(type_)
    if has_tz:
        expected = [
            to_datetime("2022-04-30 00:00:00", utc=True),
            to_datetime("2022-04-30 00:00:01", utc=True),
        ]
    else:
        expected = [
            to_datetime("2022-04-30 00:00:00"),
            to_datetime("2022-04-30 00:00:01"),
        ]
    assert data_type.try_coerce(data).to_list() == expected


def test_coerce_string():
    """Test that strings can be coerced."""
    data = pd.Series([1, None], dtype="Int32")
    coerced = pandas_engine.Engine.dtype(str).coerce(data).to_list()
    assert isinstance(coerced[0], str)
    assert pd.isna(coerced[1])


def test_default_numeric_dtypes():
    """
    Test that default numeric dtypes int, float and complex are consistent.
    """
    default_int_dtype = pd.Series([1]).dtype
    assert (
        pandas_engine.Engine.dtype(default_int_dtype)
        == pandas_engine.Engine.dtype(int)
        == pandas_engine.Engine.dtype("int")
    )

    default_float_dtype = pd.Series([1.0]).dtype
    assert (
        pandas_engine.Engine.dtype(default_float_dtype)
        == pandas_engine.Engine.dtype(float)
        == pandas_engine.Engine.dtype("float")
    )

    default_complex_dtype = pd.Series([complex(1)]).dtype
    assert (
        pandas_engine.Engine.dtype(default_complex_dtype)
        == pandas_engine.Engine.dtype(complex)
        == pandas_engine.Engine.dtype("complex")
    )


@pytest.mark.parametrize(
    "alias, np_dtype",
    [
        (alias, np_dtype)
        for alias, np_dtype in np.sctypeDict.items()
        # int, uint have different bitwidth under pandas and numpy.
        if np_dtype != np.void and alias not in ("int", "uint")
    ],
)
def test_numpy_dtypes(alias, np_dtype):
    """Test that all pandas-compatible numpy dtypes are understood."""
    try:
        np.dtype(alias)
    except TypeError:
        # not a valid alias
        assert pandas_engine.Engine.dtype(np_dtype)
    else:
        assert pandas_engine.Engine.dtype(alias) == pandas_engine.Engine.dtype(
            np_dtype
        )


def test_numpy_string():
    """
    Test that numpy string dtype check identifies the correct failure case.
    """
    schema: pa.DataFrameSchema = pa.DataFrameSchema(
        columns={
            "col1": pa.Column(str),
        }
    )
    df: pd.DataFrame = pd.DataFrame({"col1": ["1", "2", 3]})
    try:
        print(schema(df, lazy=True))
    except pa.errors.SchemaErrors as exc:
        assert exc.failure_cases["failure_case"].item() == 3
        assert exc.failure_cases["index"].item() == 2


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
            [datetime.date(2022, 1, 1)],  # date,
        ]
    ],
)
def test_inferred_dtype(examples: pd.Series):
    """Test compatibility with pd.api.types.infer_dtype's outputs."""
    alias = pd.api.types.infer_dtype(examples)
    if "mixed" in alias or alias == "string":
        # infer_dtype returns "string"
        # whereas a Series will default to a "np.object_" dtype
        inferred_datatype = pandas_engine.Engine.dtype(object)
    else:
        inferred_datatype = pandas_engine.Engine.dtype(alias)

    if alias.startswith(("decimal", "date")):
        assert str(inferred_datatype).lower().startswith(alias)
    else:
        actual_dtype = pandas_engine.Engine.dtype(pd.Series(examples).dtype)
        assert actual_dtype.check(inferred_datatype)


@pytest.mark.parametrize(
    "int_dtype, expected",
    [(dtype, True) for dtype in (*int_dtypes, *nullable_int_dtypes)]
    + [("string", False)],  # type:ignore
)
def test_is_int(int_dtype: Any, expected: bool):
    """Test is_int."""
    pandera_dtype = pandas_engine.Engine.dtype(int_dtype)
    assert pa.dtypes.is_int(pandera_dtype) == expected


@pytest.mark.parametrize(
    "uint_dtype, expected",
    [(dtype, True) for dtype in (*uint_dtypes, *nullable_uint_dtypes)]
    + [("string", False)],  # type:ignore
)
def test_is_uint(uint_dtype: Any, expected: bool):
    """Test is_uint."""
    pandera_dtype = pandas_engine.Engine.dtype(uint_dtype)
    assert pa.dtypes.is_uint(pandera_dtype) == expected


@pytest.mark.parametrize(
    "float_dtype, expected",
    [
        (dtype, True)
        for dtype in (
            *float_dtypes,
            *(NULLABLE_FLOAT_DTYPES if NULLABLE_FLOAT_DTYPES else []),
        )
    ]
    + [("string", False)],  # type: ignore
)
def test_is_float(float_dtype: Any, expected: bool):
    """Test is_float."""
    pandera_dtype = pandas_engine.Engine.dtype(float_dtype)
    assert pa.dtypes.is_float(pandera_dtype) == expected


@pytest.mark.parametrize(
    "complex_dtype, expected",
    [(dtype, True) for dtype in complex_dtypes] + [("string", False)],  # type: ignore
)
def test_is_complex(complex_dtype: Any, expected: bool):
    """Test is_complex."""
    pandera_dtype = pandas_engine.Engine.dtype(complex_dtype)
    assert pa.dtypes.is_complex(pandera_dtype) == expected


@pytest.mark.parametrize(
    "bool_dtype, expected",
    [(dtype, True) for dtype in (*boolean_dtypes, *nullable_boolean_dtypes)]
    + [("string", False)],
)
def test_is_bool(bool_dtype: Any, expected: bool):
    """Test is_bool."""
    pandera_dtype = pandas_engine.Engine.dtype(bool_dtype)
    assert pa.dtypes.is_bool(pandera_dtype) == expected


@pytest.mark.parametrize(
    "string_dtype, expected",
    [(dtype, True) for dtype in string_dtypes]
    + [("int", False)],  # type:ignore
)
def test_is_string(string_dtype: Any, expected: bool):
    """Test is_string."""
    pandera_dtype = pandas_engine.Engine.dtype(string_dtype)
    assert pa.dtypes.is_string(pandera_dtype) == expected


@pytest.mark.parametrize(
    "category_dtype, expected",
    [(dtype, True) for dtype in category_dtypes] + [("string", False)],
)
def test_is_category(category_dtype: Any, expected: bool):
    """Test is_category."""
    pandera_dtype = pandas_engine.Engine.dtype(category_dtype)
    assert pa.dtypes.is_category(pandera_dtype) == expected


@pytest.mark.parametrize(
    "datetime_dtype, expected",
    [(dtype, True) for dtype in timestamp_dtypes] + [("string", False)],
)
def test_is_datetime(datetime_dtype: Any, expected: bool):
    """Test is_datetime."""
    pandera_dtype = pandas_engine.Engine.dtype(datetime_dtype)
    assert pa.dtypes.is_datetime(pandera_dtype) == expected


@pytest.mark.parametrize(
    "timedelta_dtype, expected",
    [(dtype, True) for dtype in timedelta_dtypes] + [("string", False)],
)
def test_is_timedelta(timedelta_dtype: Any, expected: bool):
    """Test is_timedelta."""
    pandera_dtype = pandas_engine.Engine.dtype(timedelta_dtype)
    assert pa.dtypes.is_timedelta(pandera_dtype) == expected


def test_pandera_timedelta_dtype():
    """Test is pandera-native timedelta dtype."""
    assert pa.Timedelta().__str__() == "timedelta"


@pytest.mark.parametrize(
    "numeric_dtype, expected",
    [(dtype, True) for dtype, _ in numeric_dtypes] + [("string", False)],
)
def test_is_numeric(numeric_dtype: Any, expected: bool):
    """Test is_timedelta."""
    pandera_dtype = pandas_engine.Engine.dtype(numeric_dtype)
    assert pa.dtypes.is_numeric(pandera_dtype) == expected


def test_is_binary():
    """Test is_binary."""
    assert pa.dtypes.is_binary(pa.dtypes.Binary)
    assert pa.dtypes.is_binary(pa.dtypes.Binary())
    assert pa.dtypes.Binary().__str__() == "binary"


def test_python_typing_dtypes():
    """Test that supporting typing module dtypes work."""

    class PointDict(TypedDict):
        """Custom TypedDict type"""

        x: float
        y: float

    class PointTuple(NamedTuple):
        """Custom NamedTuple type"""

        x: float
        y: float

    schema = pa.DataFrameSchema(
        {
            "Dict_column": pa.Column(Dict[str, int]),
            "List_column": pa.Column(List[float]),
            "Tuple_column": pa.Column(Tuple[int, str, float]),
            "typeddict_column": pa.Column(PointDict),
            "namedtuple_column": pa.Column(PointTuple),
        },
    )

    class Model(pa.DataFrameModel):
        Dict_column: Dict[str, int]
        List_column: List[float]
        Tuple_column: Tuple[int, str, float]
        typeddict_column: PointDict
        namedtuple_column: PointTuple

    data = pd.DataFrame(
        {
            "Dict_column": [{"foo": 1, "bar": 2}],
            "List_column": [[1.0]],
            "Tuple_column": [(1, "bar", 1.0)],
            "typeddict_column": [PointDict(x=2.1, y=4.8)],
            "namedtuple_column": [PointTuple(x=9.2, y=1.6)],
        }
    )

    schema.validate(data)
    Model.validate(data)


@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="standard collection generics are not supported in python < 3.9",
)
def test_python_std_list_dict_generics():
    """Test that supporting std list/dict generic dtypes work."""
    schema = pa.DataFrameSchema(
        {
            "dict_column": pa.Column(dict[str, int]),
            "list_column": pa.Column(list[float]),
            "tuple_column": pa.Column(tuple[int, str, float]),
        },
    )

    class Model(pa.DataFrameModel):
        dict_column: dict[str, int]
        list_column: list[float]
        tuple_column: tuple[int, str, float]

    data = pd.DataFrame(
        {
            "dict_column": [{"foo": 1, "bar": 2}],
            "list_column": [[1.0]],
            "tuple_column": [(1, "bar", 1.0)],
        }
    )
    schema.validate(data)
    Model.validate(data)


@pytest.mark.parametrize("nullable", [True, False])
@pytest.mark.parametrize(
    "data_dict",
    [
        {
            "dict_column": [{"foo": 1}, {"foo": 1, "bar": 2}, {}, None],
            "list_column": [[1.0], [1.0, 2.0], [], None],
        },
        {
            "dict_column": [{"foo": "1"}, {"foo": "1", "bar": "2"}, {}, None],
            "list_column": [["1.0"], ["1.0", "2.0"], [], None],
        },
    ],
)
def test_python_typing_handle_empty_list_dict_and_none(nullable, data_dict):
    """
    Test that None values for dicts and lists are handled by nullable check.
    """

    schema = pa.DataFrameSchema(
        {
            "dict_column": pa.Column(Dict[str, int], nullable=nullable),
            "list_column": pa.Column(List[float], nullable=nullable),
        },
        coerce=True,
    )

    class Model(pa.DataFrameModel):
        dict_column: Dict[str, int] = pa.Field(nullable=nullable)
        list_column: List[float] = pa.Field(nullable=nullable)

        class Config:
            coerce = True

    data = pd.DataFrame(data_dict)

    expected = pd.DataFrame(
        {
            "dict_column": [{"foo": 1}, {"foo": 1, "bar": 2}, {}, pd.NA],
            "list_column": [[1.0], [1.0, 2.0], [], pd.NA],
        }
    )

    if nullable:
        assert schema.validate(data).equals(expected)
        assert Model.validate(data).equals(expected)
    else:
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(data)
        with pytest.raises(pa.errors.SchemaError):
            Model.validate(data)


@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="standard collection generics are not supported in python < 3.9",
)
@pytest.mark.parametrize("nullable", [True, False])
@pytest.mark.parametrize(
    "data_dict",
    [
        {
            "dict_column": [{"foo": 1}, {"foo": 1, "bar": 2}, {}, None],
            "list_column": [[1.0], [1.0, 2.0], [], None],
        },
        {
            "dict_column": [{"foo": "1"}, {"foo": "1", "bar": "2"}, {}, None],
            "list_column": [["1.0"], ["1.0", "2.0"], [], None],
        },
    ],
)
def test_python_std_list_dict_empty_and_none(nullable, data_dict):
    """
    Test that None values for standard dicts and lists are handled by nullable
    check.
    """

    schema = pa.DataFrameSchema(
        {
            "dict_column": pa.Column(dict[str, int], nullable=nullable),
            "list_column": pa.Column(list[float], nullable=nullable),
        },
        coerce=True,
    )

    class Model(pa.DataFrameModel):
        dict_column: dict[str, int] = pa.Field(nullable=nullable)
        list_column: list[float] = pa.Field(nullable=nullable)

        class Config:
            coerce = True

    data = pd.DataFrame(data_dict)

    expected = pd.DataFrame(
        {
            "dict_column": [{"foo": 1}, {"foo": 1, "bar": 2}, {}, pd.NA],
            "list_column": [[1.0], [1.0, 2.0], [], pd.NA],
        }
    )

    if nullable:
        assert schema.validate(data).equals(expected)
        assert Model.validate(data).equals(expected)
    else:
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(data)
        with pytest.raises(pa.errors.SchemaError):
            Model.validate(data)


def test_python_std_list_dict_error():
    """Test that non-standard dict/list invalid values raise Schema Error."""
    schema = pa.DataFrameSchema(
        {
            "dict_column": pa.Column(Dict[str, int]),
            "list_column": pa.Column(List[float]),
        },
    )

    class Model(pa.DataFrameModel):
        dict_column: Dict[str, int]
        list_column: List[float]

    data = pd.DataFrame(
        {
            "dict_column": [{"foo": 1}, {"foo": 1, "bar": "2"}, {}],
            "list_column": [[1.0], ["1.0", 2.0], []],
        }
    )

    for validator in (schema, Model):
        try:
            validator.validate(data, lazy=True)
        except pa.errors.SchemaErrors as exc:
            assert exc.failure_cases["failure_case"].iloc[0] == {
                "foo": 1,
                "bar": "2",
            }
            assert exc.failure_cases["failure_case"].iloc[1] == ["1.0", 2.0]
