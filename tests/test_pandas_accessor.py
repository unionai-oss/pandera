"""Unit tests for pandas_accessor module."""

import pandas as pd
import pytest

import pandera as pa


@pytest.mark.parametrize(
    "schema1, schema2, data",
    [
        [
            pa.DataFrameSchema({"col": pa.Column(int)}, coerce=True),
            pa.DataFrameSchema({"col": pa.Column(float)}, coerce=True),
            pd.DataFrame({"col": [1, 2, 3]}),
        ],
        [
            pa.SeriesSchema(int, coerce=True),
            pa.SeriesSchema(float, coerce=True),
            pd.Series([1, 2, 3]),
        ],
    ],
)
@pytest.mark.parametrize("inplace", [False, True])
def test_dataframe_series_add_schema(schema1, schema2, data, inplace):
    """
    Test that pandas object contains schema metadata after pandera validation.
    """
    validated_data_1 = schema1(data, inplace=inplace)
    if inplace:
        assert data.pandera.schema == schema1
    else:
        assert data.pandera.schema is None
    assert validated_data_1.pandera.schema == schema1

    validated_data_2 = schema2(validated_data_1, inplace=inplace)
    if inplace:
        assert validated_data_1.pandera.schema == schema2
    else:
        assert validated_data_1.pandera.schema == schema1
    assert validated_data_2.pandera.schema == schema2
