"""Tests Engine subclassing and registring DataTypes."""
# pylint:disable=redefined-outer-name,unused-argument
# pylint:disable=missing-function-docstring,missing-class-docstring
import re
from typing import Any, Generator, List, Union

import pytest

from pandera.engines import pyspark_engine
import pyspark.sql.types as pst


@pytest.mark.parametrize(
    "data_type", list(pyspark_engine.Engine.get_registered_dtypes())
)
def test_pyspark_data_type(data_type):

    with pytest.warns(UserWarning):
        ps_dtype = pyspark_engine.DataType(data_type.type)

    assert ps_dtype != None
