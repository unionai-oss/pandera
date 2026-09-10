"""Tests for dtype resolution in the pyarrow schema API."""

import pyarrow
import pytest

import pandera.pyarrow as pa
from pandera.api.pyarrow.utils import pyarrow_dtype_to_narwhals, resolve_dtype
from pandera.errors import SchemaError


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (pyarrow.int8(), "Int8"),
        (pyarrow.int64(), "Int64"),
        (pyarrow.uint16(), "UInt16"),
        (pyarrow.float32(), "Float32"),
        # pa.float64() stringifies as "double" — the string alias path
        # cannot resolve it, so this covers the translation helper.
        (pyarrow.float64(), "Float64"),
        (pyarrow.bool_(), "Boolean"),
        (pyarrow.string(), "String"),
        (pyarrow.large_string(), "String"),
        (pyarrow.date32(), "Date"),
    ],
)
def test_pyarrow_dtype_translation(dtype, expected):
    assert str(pyarrow_dtype_to_narwhals(dtype)) == expected


def test_parametrized_dtype_translation():
    assert str(pyarrow_dtype_to_narwhals(pyarrow.timestamp("us"))).startswith(
        "Datetime"
    )
    assert str(
        pyarrow_dtype_to_narwhals(pyarrow.decimal128(10, 2))
    ).startswith("Decimal")
    assert str(pyarrow_dtype_to_narwhals(pyarrow.list_(pyarrow.int32()))) == (
        "List(Int32)"
    )


@pytest.mark.parametrize("builtin", [int, float, str, bool])
def test_python_builtins_resolve(builtin):
    """Regression test: the narwhals engine used to reject python builtins."""
    assert resolve_dtype(builtin) is not None


@pytest.mark.parametrize(
    "column_dtype,data,valid",
    [
        (pyarrow.int64(), [1, 2, 3], True),
        (pyarrow.float64(), [1.0, 2.0], True),
        (pyarrow.float64(), [1, 2], False),
        (int, [1, 2, 3], True),
        (float, [1.0, 2.0], True),
        (str, ["a"], True),
        (bool, [True, False], True),
        ("int64", [1, 2], True),
        ("string", ["a"], True),
    ],
)
def test_column_dtype_validation(column_dtype, data, valid):
    schema = pa.DataFrameSchema({"a": pa.Column(column_dtype)})
    tbl = pyarrow.table({"a": data})
    if valid:
        assert schema.validate(tbl).equals(tbl)
    else:
        with pytest.raises(SchemaError):
            schema.validate(tbl)


def test_timestamp_column():
    import datetime

    schema = pa.DataFrameSchema({"ts": pa.Column(pyarrow.timestamp("us"))})
    tbl = pyarrow.table({"ts": pyarrow.array([datetime.datetime(2024, 1, 1)])})
    assert schema.validate(tbl).equals(tbl)


def test_unknown_dtype_raises():
    with pytest.raises(TypeError):
        pa.Column(complex)
