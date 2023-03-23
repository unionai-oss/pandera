"""Tests Engine subclassing and registring DataTypes."""
# pylint:disable=redefined-outer-name,unused-argument
# pylint:disable=missing-function-docstring,missing-class-docstring
import re
from typing import Any, Generator, List, Union

import pytest

from pandera.engines import pyspark_engine


@pytest.mark.parametrize(
    "data_type", list(pyspark_engine.Engine.get_registered_dtypes())
)
def test_pyspark_data_type(data_type):
    if data_type.type is None:
        return
    breakpoint()
    pyspark_engine.Engine.dtype(data_type)
    pyspark_engine.Engine.dtype(data_type.type)
    pyspark_engine.Engine.dtype(str(data_type.type))

    with pytest.warns(UserWarning):
        ps_dtype = pyspark_engine.DataType(data_type.type)
    with pytest.warns(UserWarning):
        ps_dtype_from_str = pyspark_engine.DataType(str(data_type.type))

    assert ps_dtype == ps_dtype_from_str
    assert not ps_dtype.check("foo")
