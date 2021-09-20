"""Test pandera on koalas data structures."""

import os
from unittest.mock import MagicMock

import databricks.koalas as ks
import pandas as pd
import pytest

import pandera as pa
from pandera.engines import numpy_engine, pandas_engine
from tests.strategies.test_strategies import (
    UNSUPPORTED_DTYPES as UNSUPPORTED_STRATEGY_DTYPES,
)

try:
    import hypothesis
    import hypothesis.strategies as st
except ImportError:
    HAS_HYPOTHESIS = False
    hypothesis = MagicMock()
    st = MagicMock()
else:
    HAS_HYPOTHESIS = True


UNSUPPORTED_STRATEGY_DTYPES = set(UNSUPPORTED_STRATEGY_DTYPES)
UNSUPPORTED_STRATEGY_DTYPES.add(numpy_engine.Object)


os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"


KOALAS_UNSUPPORTED = [
    numpy_engine.Complex256,
    numpy_engine.Complex128,
    numpy_engine.Complex64,
    numpy_engine.Float128,
    numpy_engine.Float16,
    numpy_engine.Object,
    numpy_engine.Timedelta64,
    numpy_engine.UInt64,
    numpy_engine.UInt32,
    numpy_engine.UInt16,
    numpy_engine.UInt8,
    pandas_engine.Category,
    pandas_engine.Interval,
    pandas_engine.Period,
    pandas_engine.Sparse,
    pandas_engine.UINT64,
    pandas_engine.UINT32,
    pandas_engine.UINT16,
    pandas_engine.UINT8,
]


def _test_datatype(
    dtype: pandas_engine.DataType,
    sample: pd.DataFrame,
    schema: pa.DataFrameSchema,
):
    """Test pandera datatypes against koalas.

    Handle case where koalas can't handle datetimes before 1900-01-01 00:04:00,
    raising an overflow
    """
    if dtype is pandas_engine.DateTime:
        if (sample < pd.Timestamp("1900-01-01 00:04:00")).any(axis=None):
            with pytest.raises(
                OverflowError, match="mktime argument out of range"
            ):
                ks.DataFrame(sample)
    else:
        assert isinstance(schema(ks.DataFrame(sample)), ks.DataFrame)


@pytest.mark.parametrize("coerce", [True, False])
def test_dataframe_schema_case(coerce):
    """Test a simple schema case."""
    schema = pa.DataFrameSchema(
        {
            "int_column": pa.Column(int, pa.Check.ge(0)),
            "float_column": pa.Column(float, pa.Check.le(0)),
            "str_column": pa.Column(str, pa.Check.isin(list("abcde"))),
        },
        coerce=coerce,
    )
    kdf = ks.DataFrame(
        {
            "int_column": range(10),
            "float_column": [float(-x) for x in range(10)],
            "str_column": list("aabbcceedd"),
        }
    )
    assert isinstance(schema.validate(kdf), ks.DataFrame)


@pytest.mark.parametrize(
    "dtype",
    pandas_engine.Engine.get_registered_dtypes(),
)
@pytest.mark.parametrize("coerce", [True])
@hypothesis.given(st.data())
def test_dataframe_schema_dtypes(
    dtype: pandas_engine.DataType,
    coerce: bool,
    data: st.DataObject,
):
    """Test that all supported koalas data types work as expected."""
    if dtype in UNSUPPORTED_STRATEGY_DTYPES:
        pytest.skip(
            f"type {dtype} currently not supported by the strategies module"
        )

    schema = pa.DataFrameSchema({"column": pa.Column(dtype)}, coerce=coerce)
    if dtype in KOALAS_UNSUPPORTED:
        with pytest.raises(TypeError):
            ks.DataFrame(data.draw(schema.strategy(size=3)))
        return

    sample = data.draw(schema.strategy(size=3))
    _test_datatype(dtype, sample, schema)


def test_dataframe_schema_checks():
    """Test that all built-in checks work."""
    ...


# Test settings:
# - check coercing one dtype to another
# - make sure errors are correctly reported
# - test customized checks
