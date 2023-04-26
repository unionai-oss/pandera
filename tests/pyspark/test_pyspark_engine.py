"""Tests Engine subclassing and registring DataTypes."""
# pylint:disable=redefined-outer-name,unused-argument
# pylint:disable=missing-function-docstring,missing-class-docstring
import re
from typing import Any, Generator, List, Union

import pytest

from pandera.engines import pyspark_engine
import pyspark.sql.types as pst
from pyspark.sql import SparkSession

"""Test pyspark engine."""

from datetime import date

import hypothesis
import hypothesis.extra.pandas as pd_st
import hypothesis.strategies as st
import pandas as pd
import pytest
import pytz
from hypothesis import given

from pandera.engines import pyspark_engine
from pandera.errors import ParserError


@pytest.mark.parametrize(
    "data_type", list(pyspark_engine.Engine.get_registered_dtypes())
)
def test_pyspark_data_type(data_type):
    """Test pyspark engine DataType base class."""
    if data_type.type is None:
        # don't test data types that require parameters e.g. Category
        return
    parameterized_datatypes = ["daytimeinterval", "decimal", "array", "map"]
    breakpoint()
    pyspark_engine.Engine.dtype(data_type)
    pyspark_engine.Engine.dtype(data_type.type)
    if data_type.type.typeName() not in parameterized_datatypes:
        print(data_type.type.typeName())
        pyspark_engine.Engine.dtype(str(data_type.type))


    with pytest.warns(UserWarning):
        pd_dtype = pyspark_engine.DataType(data_type.type)
    if data_type.type.typeName() not in parameterized_datatypes:
        with pytest.warns(UserWarning):
            pd_dtype_from_str = pyspark_engine.DataType(str(data_type.type))
            assert pd_dtype == pd_dtype_from_str




