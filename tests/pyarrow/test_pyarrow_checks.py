"""Tests for checks against pyarrow Tables."""

import pyarrow
import pyarrow.compute as pc
import pytest

import pandera.pyarrow as pa
from pandera.api.pyarrow.types import PyArrowData
from pandera.errors import SchemaError


@pytest.mark.parametrize(
    "check,passing,failing",
    [
        (pa.Check.gt(0), [1, 2], [0, 1]),
        (pa.Check.ge(0), [0, 1], [-1, 0]),
        (pa.Check.lt(10), [1, 2], [10, 1]),
        (pa.Check.le(10), [10], [11]),
        (pa.Check.eq(1), [1, 1], [1, 2]),
        (pa.Check.ne(1), [2, 3], [1, 2]),
        (pa.Check.isin([1, 2]), [1, 2], [1, 3]),
        (pa.Check.notin([3]), [1, 2], [3]),
        (pa.Check.between(0, 5), [1, 5], [6]),
    ],
)
def test_builtin_numeric_checks(check, passing, failing):
    schema = pa.DataFrameSchema({"a": pa.Column(int, check)})
    assert schema.validate(pyarrow.table({"a": passing})) is not None
    with pytest.raises(SchemaError):
        schema.validate(pyarrow.table({"a": failing}))


@pytest.mark.parametrize(
    "check,passing,failing",
    [
        (pa.Check.str_startswith("a"), ["ab", "ac"], ["ab", "bc"]),
        (pa.Check.str_endswith("z"), ["az"], ["za"]),
        (pa.Check.str_contains("b"), ["abc"], ["acd"]),
        (pa.Check.str_matches(r"^\d+$"), ["123"], ["12a"]),
    ],
)
def test_builtin_string_checks(check, passing, failing):
    schema = pa.DataFrameSchema({"a": pa.Column(str, check)})
    assert schema.validate(pyarrow.table({"a": passing})) is not None
    with pytest.raises(SchemaError):
        schema.validate(pyarrow.table({"a": failing}))


def test_native_check_pyarrow_data_convention():
    """A 1-arg native check receives a ``PyArrowData`` container.

    This mirrors ``PolarsData`` / ``IbisData`` on the other backends.
    """
    seen = {}

    def check_fn(data):
        seen["type"] = type(data)
        seen["key"] = data.key
        return pc.greater(data.table[data.key], 0)

    schema = pa.DataFrameSchema({"a": pa.Column(int, pa.Check(check_fn))})
    schema.validate(pyarrow.table({"a": [1, 2]}))

    assert seen["type"] is PyArrowData
    assert seen["key"] == "a"

    with pytest.raises(SchemaError):
        schema.validate(pyarrow.table({"a": [1, -2]}))


def test_native_check_two_arg_convention():
    """A 2-arg native check gets ``(native_table, key)``."""
    schema = pa.DataFrameSchema(
        {
            "a": pa.Column(
                int, pa.Check(lambda tbl, key: pc.greater(tbl[key], 0))
            )
        }
    )
    assert schema.validate(pyarrow.table({"a": [1, 2]})) is not None
    with pytest.raises(SchemaError):
        schema.validate(pyarrow.table({"a": [1, -2]}))


def test_native_check_returning_boolean_scalar():
    """An aggregate pyarrow scalar is normalized to a python bool."""
    schema = pa.DataFrameSchema(
        {
            "a": pa.Column(
                int,
                pa.Check(lambda tbl, key: pc.all(pc.greater(tbl[key], 0))),
            )
        }
    )
    assert schema.validate(pyarrow.table({"a": [1, 2]})) is not None
    with pytest.raises(SchemaError):
        schema.validate(pyarrow.table({"a": [1, -2]}))


def test_non_native_expression_check():
    """``native=False`` uses the narwhals expression protocol."""
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int, pa.Check(lambda col: col > 0, native=False))}
    )
    assert schema.validate(pyarrow.table({"a": [1, 2]})) is not None
    with pytest.raises(SchemaError):
        schema.validate(pyarrow.table({"a": [1, -2]}))


def test_element_wise_check():
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int, pa.Check(lambda v: v > 0, element_wise=True))}
    )
    assert schema.validate(pyarrow.table({"a": [1, 2]})) is not None
    with pytest.raises(SchemaError):
        schema.validate(pyarrow.table({"a": [1, -2]}))


def test_failure_cases_are_reported():
    schema = pa.DataFrameSchema({"a": pa.Column(int, pa.Check.gt(0))})
    with pytest.raises(SchemaError) as exc_info:
        schema.validate(pyarrow.table({"a": [1, -2, -3]}), lazy=False)
    failure_cases = exc_info.value.failure_cases
    assert failure_cases is not None
