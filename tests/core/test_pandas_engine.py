"""Test numpy engine."""

import hypothesis.strategies as st
import pandas as pd
import pytest
from hypothesis import given

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


CATEGORIES = ["A", "B", "C"]


@given(st.lists(st.sampled_from(CATEGORIES), min_size=5))
def test_pandas_category_dtype(data):
    """Test pandas_engine.Category correctly coerces valid categorical data."""
    data = pd.Series(data)
    dtype = pandas_engine.Category(CATEGORIES)
    coerced_data = dtype.coerce(data)
    assert dtype.check(coerced_data.dtype)

    for _, value in data.iteritems():
        coerced_value = dtype.coerce_value(value)
        assert coerced_value in CATEGORIES


@given(st.lists(st.sampled_from(["X", "Y", "Z"]), min_size=5))
def test_pandas_category_dtype_error(data):
    """Test pandas_engine.Category raises TypeErrors on invalid data."""
    data = pd.Series(data)
    dtype = pandas_engine.Category(CATEGORIES)

    with pytest.raises(TypeError):
        dtype.coerce(data)

    for _, value in data.iteritems():
        with pytest.raises(TypeError):
            dtype.coerce_value(value)


@given(st.lists(st.sampled_from([1, 0, 1.0, 0.0, True, False]), min_size=5))
def test_pandas_boolean_native_type(data):
    """Test native pandas bool type correctly coerces valid bool-like data."""
    data = pd.Series(data)
    dtype = pandas_engine.Engine.dtype("boolean")

    # the BooleanDtype can't handle Series of non-boolean, mixed dtypes
    if data.dtype == "object":
        with pytest.raises(TypeError):
            dtype.coerce(data)
    else:
        coerced_data = dtype.coerce(data)
        assert dtype.check(coerced_data.dtype)

    for _, value in data.iteritems():
        dtype.coerce_value(value)


@given(st.lists(st.sampled_from(["A", "True", "False", 5, -1]), min_size=5))
def test_pandas_boolean_native_type_error(data):
    """Test native pandas bool type raises TypeErrors on non-bool-like data."""
    data = pd.Series(data)
    dtype = pandas_engine.Engine.dtype("boolean")

    with pytest.raises(TypeError):
        dtype.coerce(data)

    for _, value in data.iteritems():
        with pytest.raises(TypeError):
            dtype.coerce_value(value)
