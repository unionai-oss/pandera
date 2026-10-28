"""Dataframe-wide coercion must not silently wrap integer values."""

import pandas as pd
import pytest

import pandera.pandas as pa
from pandera.engines import numpy_engine


@pytest.mark.parametrize(
    "target, value",
    [
        ("int8", 300),
        ("uint8", -1),
        ("int16", 70000),
        ("uint64", -1),
    ],
)
@pytest.mark.parametrize("lazy", [False, True])
def test_dataframe_dtype_rejects_overflow(target, value, lazy):
    data = pd.DataFrame([[1, value]], columns=["safe", "bad"])
    schema = pa.DataFrameSchema(dtype=target, coerce=True)
    with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
        schema.validate(data, lazy=lazy)


@pytest.mark.parametrize(
    "values, target, expected",
    [
        ([1, 2], "int8", [1, 2]),
        ([1.7, 2.9], "int64", [1, 2]),
        (["1", "2"], "int64", [1, 2]),
        ([True, False], "int64", [1, 0]),
    ],
)
def test_dataframe_dtype_valid_coercion(values, target, expected):
    data = pd.DataFrame({"value": values})
    result = pa.DataFrameSchema(dtype=target, coerce=True).validate(data)
    assert result["value"].tolist() == expected


def test_dataframe_dtype_empty_integer_columns():
    data = pd.DataFrame(
        {"a": pd.Series([], dtype="int64"), "b": pd.Series([], dtype="int64")}
    )
    result = pa.DataFrameSchema(dtype="int8", coerce=True).validate(data)
    assert result.empty
    assert all(dtype == "int8" for dtype in result.dtypes)


@pytest.mark.parametrize(
    "target, value",
    [
        ("int8", 300),
        ("uint8", -1),
        ("int16", 70000),
        ("uint64", -1),
    ],
)
def test_dataframe_dtype_duplicate_columns_reject_overflow(target, value):
    data = pd.DataFrame([[1, value]], columns=["value", "value"])
    with pytest.raises(OverflowError):
        numpy_engine.Engine.dtype(target).coerce(data)
