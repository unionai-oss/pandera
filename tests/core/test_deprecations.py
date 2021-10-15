"""Unit tests for deprecated features."""

import pytest

import pandera as pa
from pandera.system import FLOAT_128_AVAILABLE


@pytest.mark.parametrize(
    "schema_cls,as_pos_arg",
    [
        [pa.DataFrameSchema, False],
        [pa.SeriesSchema, True],
        [pa.Column, True],
        [pa.Index, True],
    ],
)
def test_deprecate_pandas_dtype(schema_cls, as_pos_arg):
    """Test that pandas_dtype deprecation warnings/errors are raised."""
    assert schema_cls(dtype=int).dtype.check(pa.Int())
    with pytest.warns(DeprecationWarning):
        assert schema_cls(pandas_dtype=int).dtype.check(pa.Int())

    with pytest.warns(DeprecationWarning):
        schema_cls(pandas_dtype=int)
    with pytest.raises(pa.errors.SchemaInitError):
        schema_cls(dtype=int, pandas_dtype=int)

    if as_pos_arg:
        assert schema_cls(int).dtype.check(pa.Int())
        with pytest.raises(pa.errors.SchemaInitError):
            schema_cls(int, pandas_dtype=int)


@pytest.mark.parametrize(
    "schema_cls",
    [
        pa.DataFrameSchema,
        pa.SeriesSchema,
        pa.Column,
        pa.Index,
    ],
)
def test_deprecate_pandas_dtype_enum(schema_cls):
    """Test that using the PandasDtype enum raises a DeprecationWarning."""
    for attr in pa.PandasDtype:
        if not FLOAT_128_AVAILABLE and attr in {
            "Float128",
            "Complex256",
        }:
            continue
        with pytest.warns(DeprecationWarning):
            pandas_dtype = getattr(pa.PandasDtype, attr)
            schema_cls(dtype=pandas_dtype)
