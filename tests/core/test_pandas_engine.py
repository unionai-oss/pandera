"""Test numpy engine."""

import pandas as pd
import pytest

from pandera.engines import pandas_engine
from pandera.errors import ParserError


@pytest.mark.parametrize(
    "data_type", list(pandas_engine.Engine.get_registered_dtypes())
)
def test_pandas_data_type(data_type):
    """Test numpy engine DataType base class."""
    if data_type.type is None:
        # don't test data types that require parameters e.g. Category
        return

    pandas_engine.Engine.dtype(data_type)
    pandas_engine.Engine.dtype(data_type.type)
    pandas_engine.Engine.dtype(str(data_type.type))

    with pytest.warns(UserWarning):
        pd_dtype = pandas_engine.DataType(data_type.type)
    with pytest.warns(UserWarning):
        pd_dtype_from_str = pandas_engine.DataType(str(data_type.type))
    assert pd_dtype == pd_dtype_from_str
    assert not pd_dtype.check("foo")


@pytest.mark.parametrize(
    "data_type", list(pandas_engine.Engine.get_registered_dtypes())
)
def test_pandas_data_type_coerce(data_type):
    """
    Test that pandas data type coercion will raise a ParserError. on failure.
    """
    if data_type.type is None:
        # don't test data types that require parameters e.g. Category
        return
    try:
        data_type().try_coerce(pd.Series(["1", "2", "a"]))
    except ParserError as exc:
        assert exc.failure_cases.shape[0] > 0
