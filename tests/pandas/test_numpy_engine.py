"""Test numpy engine."""

import numpy as np
import pytest

from pandera.engines import numpy_engine


@pytest.mark.parametrize(
    "data_type", list(numpy_engine.Engine.get_registered_dtypes())
)
def test_numpy_data_type(data_type):
    """Test base numpy engine DataType."""
    numpy_engine.Engine.dtype(data_type)
    numpy_engine.Engine.dtype(data_type.type)
    numpy_engine.Engine.dtype(str(data_type.type))
    with pytest.warns(UserWarning):
        np_dtype = numpy_engine.DataType(data_type.type)
    with pytest.warns(UserWarning):
        np_dtype_from_str = numpy_engine.DataType(str(data_type.type))
    assert np_dtype == np_dtype_from_str


@pytest.mark.parametrize("data_type", ["foo", "bar", 1, 2, 3.14, np.void])
def test_numpy_engine_dtype_exceptions(data_type):
    """Test invalid inputs to numpy data-types."""
    if data_type != np.void:
        with pytest.raises(
            TypeError, match="data type '.+' not understood by"
        ):
            numpy_engine.Engine.dtype(data_type)
    else:
        numpy_engine.Engine._registered_dtypes = set()
        numpy_engine.Engine.dtype(data_type)


def test_numpy_string():
    """Test numpy engine String data type coercion."""
    # pylint: disable=no-value-for-parameter
    string_type = numpy_engine.String()
    assert (
        string_type.coerce(np.array([1, 2, 3, 4, 5], dtype=int))
        == np.array(list("12345"))
    ).all()
    assert string_type.check(numpy_engine.String())
