"""Unit tests for pandas_accessor module."""
from typing import Union
from unittest.mock import patch

import pandas as pd
import pytest

import pandera as pa


@pytest.mark.parametrize(
    "schema1, schema2, data, invalid_data",
    [
        [
            pa.DataFrameSchema({"col": pa.Column(int)}, coerce=True),
            pa.DataFrameSchema({"col": pa.Column(float)}, coerce=True),
            pd.DataFrame({"col": [1, 2, 3]}),
            pd.Series([1, 2, 3]),
        ],
        [
            pa.SeriesSchema(int, coerce=True),
            pa.SeriesSchema(float, coerce=True),
            pd.Series([1, 2, 3]),
            pd.DataFrame({"col": [1, 2, 3]}),
        ],
    ],
)
@pytest.mark.parametrize("inplace", [False, True])
def test_dataframe_series_add_schema(
    schema1: Union[pa.DataFrameSchema, pa.SeriesSchema],
    schema2: Union[pa.DataFrameSchema, pa.SeriesSchema],
    data: Union[pd.DataFrame, pd.Series],
    invalid_data: Union[pd.DataFrame, pd.Series],
    inplace: bool,
) -> None:
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

    with pytest.raises(TypeError, match=f"expected pd.{type(data).__name__}"):
        schema1(invalid_data)

    with pytest.raises(TypeError, match=f"expected pd.{type(data).__name__}"):
        schema2(invalid_data)

    with patch.object(pa.schemas.check_utils, "is_table", return_value=True):
        with patch.object(
            pa.schemas.check_utils,
            "is_field",
            return_value=True,
        ):
            with pytest.raises(TypeError, match="schema arg"):
                schema1(invalid_data)

            with pytest.raises(TypeError, match="schema arg"):
                schema2(invalid_data)
