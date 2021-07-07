"""Unit tests for deprecated features."""

import pytest

import pandera as pa


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
    assert schema_cls(pandas_dtype=int).dtype.check(pa.Int())

    with pytest.warns(DeprecationWarning):
        schema_cls(pandas_dtype=int)
    with pytest.raises(pa.errors.SchemaInitError):
        schema_cls(dtype=int, pandas_dtype=int)

    if as_pos_arg:
        assert schema_cls(int).dtype.check(pa.Int())
        with pytest.raises(pa.errors.SchemaInitError):
            schema_cls(int, pandas_dtype=int)
