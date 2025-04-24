"""Tests Engine subclassing and registering DataTypes.Test pyspark engine."""

# pylint:disable=redefined-outer-name,unused-argument

import pytest

from pandera.engines import pyspark_engine


@pytest.mark.parametrize(
    "data_type",
    list(
        pyspark_engine.Engine.get_registered_dtypes()
    ),  # pylint:disable=no-value-for-parameter
)
def test_pyspark_data_type(data_type):
    """Test pyspark engine DataType base class."""
    if data_type.type is None:
        # don't test data types that require parameters e.g. Category
        return
    parameterized_datatypes = ["decimal", "array", "map"]

    pyspark_engine.Engine.dtype(
        data_type
    )  # pylint:disable=no-value-for-parameter
    pyspark_engine.Engine.dtype(
        data_type.type
    )  # pylint:disable=no-value-for-parameter
    if data_type.type.typeName() not in parameterized_datatypes:
        pyspark_engine.Engine.dtype(
            str(data_type.type)
        )  # pylint:disable=no-value-for-parameter

    with pytest.warns(UserWarning):
        pd_dtype = pyspark_engine.DataType(data_type.type)
    if data_type.type.typeName() not in parameterized_datatypes:
        with pytest.warns(UserWarning):
            pd_dtype_from_str = pyspark_engine.DataType(str(data_type.type))
            assert pd_dtype == pd_dtype_from_str
