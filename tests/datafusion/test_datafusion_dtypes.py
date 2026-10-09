"""Tests for dtype resolution in the DataFusion schema API."""

import datetime

import pyarrow
import pytest

import pandera.datafusion as pa
from pandera.api.datafusion.utils import resolve_dtype
from pandera.errors import SchemaErrorReason

_NOW = datetime.datetime(2024, 1, 1)

SUPPORTED = [
    (pyarrow.int8(), [1]),
    (pyarrow.int16(), [1]),
    (pyarrow.int32(), [1]),
    (pyarrow.int64(), [1]),
    (pyarrow.uint8(), [1]),
    (pyarrow.uint16(), [1]),
    (pyarrow.uint32(), [1]),
    (pyarrow.uint64(), [1]),
    (pyarrow.float32(), [1.0]),
    (pyarrow.float64(), [1.0]),
    (pyarrow.bool_(), [True]),
    (pyarrow.string(), ["a"]),
    (pyarrow.large_string(), ["a"]),
    (pyarrow.string_view(), ["a"]),
    (pyarrow.date32(), [_NOW.date()]),
    (pyarrow.timestamp("us"), [_NOW]),
    (pyarrow.timestamp("us", tz="UTC"), [_NOW]),
    (pyarrow.duration("us"), [datetime.timedelta(days=1)]),
    (pyarrow.list_(pyarrow.int32()), [[1]]),
    (pyarrow.dictionary(pyarrow.int32(), pyarrow.string()), ["a"]),
]

UNSUPPORTED = [
    pyarrow.binary(),
    pyarrow.time64("us"),
    pyarrow.struct([("x", pyarrow.int64())]),
    pyarrow.decimal128(10, 2),
]


def _table(dtype, values):
    return pyarrow.table({"a": pyarrow.array(values, type=dtype)})


@pytest.mark.parametrize("dtype,values", SUPPORTED, ids=str)
def test_arrow_dtype_is_accepted(make_df, dtype, values):
    schema = pa.DataFrameSchema({"a": pa.Column(dtype)})
    validated = schema.validate(make_df(_table(dtype, values)))
    assert validated.schema().field("a").type == dtype


@pytest.mark.parametrize("dtype,values", SUPPORTED, ids=str)
def test_arrow_dtype_mismatch_is_reported(make_df, fails, dtype, values):
    is_string = pyarrow.types.is_string(dtype) or dtype in (
        pyarrow.large_string(),
        pyarrow.string_view(),
    )
    other = {"a": [1]} if is_string else {"a": ["x"]}
    schema = pa.DataFrameSchema({"a": pa.Column(dtype)})
    fails(schema, make_df(other), SchemaErrorReason.WRONG_DATATYPE)


@pytest.mark.parametrize("dtype", UNSUPPORTED, ids=str)
def test_unsupported_arrow_dtype_raises_at_definition(dtype):
    with pytest.raises(TypeError, match="not understood"):
        pa.Column(dtype)


@pytest.mark.parametrize(
    "builtin,arrow",
    [
        (int, pyarrow.int64()),
        (float, pyarrow.float64()),
        (str, pyarrow.string()),
        (bool, pyarrow.bool_()),
    ],
)
def test_python_builtins_resolve_to_arrow_dtypes(builtin, arrow):
    assert resolve_dtype(builtin) == resolve_dtype(arrow)


@pytest.mark.parametrize(
    "alias,arrow",
    [
        ("int8", pyarrow.int8()),
        ("int64", pyarrow.int64()),
        ("uint32", pyarrow.uint32()),
        ("float32", pyarrow.float32()),
        ("float64", pyarrow.float64()),
        ("bool", pyarrow.bool_()),
        ("string", pyarrow.string()),
        ("date", pyarrow.date32()),
    ],
)
def test_string_aliases_resolve_to_arrow_dtypes(alias, arrow):
    assert resolve_dtype(alias) == resolve_dtype(arrow)


@pytest.mark.parametrize("alias", ["double", "utf8", "large_string", "nope"])
def test_unknown_string_alias_raises(alias):
    with pytest.raises(TypeError):
        pa.Column(alias)


def test_unknown_python_type_raises():
    with pytest.raises(TypeError):
        pa.Column(complex)


def test_no_dtype_accepts_any_column(make_df):
    schema = pa.DataFrameSchema({"a": pa.Column()})
    assert schema.validate(make_df({"a": [1]})).to_pydict() == {"a": [1]}
    assert schema.validate(make_df({"a": ["x"]})).to_pydict() == {"a": ["x"]}


@pytest.mark.parametrize(
    "arrow",
    [pyarrow.string(), pyarrow.large_string(), pyarrow.string_view()],
    ids=str,
)
def test_str_accepts_every_arrow_string_type(make_df, arrow):
    """DataFusion plans yield ``string_view`` as readily as ``string``."""
    schema = pa.DataFrameSchema({"a": pa.Column(str)})
    assert schema.validate(make_df(_table(arrow, ["x"]))) is not None


def test_integer_width_is_enforced(make_df, fails):
    schema = pa.DataFrameSchema({"a": pa.Column(pyarrow.int64())})
    fails(
        schema,
        make_df(_table(pyarrow.int32(), [1])),
        SchemaErrorReason.WRONG_DATATYPE,
    )


def test_list_inner_type_is_enforced(make_df, fails):
    schema = pa.DataFrameSchema(
        {"a": pa.Column(pyarrow.list_(pyarrow.int32()))}
    )
    fails(
        schema,
        make_df(_table(pyarrow.list_(pyarrow.int64()), [[1]])),
        SchemaErrorReason.WRONG_DATATYPE,
    )


@pytest.mark.xfail(
    reason="the narwhals engine compares datetimes without their time unit",
    strict=True,
)
def test_timestamp_unit_is_enforced(make_df, fails):
    schema = pa.DataFrameSchema({"a": pa.Column(pyarrow.timestamp("us"))})
    fails(
        schema,
        make_df(_table(pyarrow.timestamp("ns"), [_NOW])),
        SchemaErrorReason.WRONG_DATATYPE,
    )


def test_schema_level_dtype(make_df, fails):
    schema = pa.DataFrameSchema(
        {"a": pa.Column(), "b": pa.Column()}, dtype=pyarrow.int64()
    )
    assert schema.validate(make_df({"a": [1], "b": [2]})) is not None
    fails(
        schema,
        make_df({"a": [1], "b": ["x"]}),
        SchemaErrorReason.WRONG_DATATYPE,
    )


def test_dtypes_of_a_sql_plan(ctx, make_df):
    """Types produced by the engine itself, not by an Arrow table."""
    ctx.register_table("dtype_source", make_df({"a": [1, 2], "s": ["x", "y"]}))
    plan = ctx.sql(
        "SELECT count(*) AS n, avg(a) AS mean, cast(max(a) AS varchar) AS c "
        "FROM dtype_source"
    )
    schema = pa.DataFrameSchema(
        {"n": pa.Column(int), "mean": pa.Column(float), "c": pa.Column(str)}
    )
    assert schema.validate(plan).to_pydict() == {
        "n": [2],
        "mean": [1.5],
        "c": ["2"],
    }
