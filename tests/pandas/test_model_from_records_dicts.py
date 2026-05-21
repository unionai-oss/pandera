import pandas as pd
import pytest

import pandera.pandas as pa
from pandera.pandas import DataFrameModel
from pandera.typing import DataFrame, Index, Series


def test_from_records_with_list_of_dicts():
    """Test that DataFrame.from_records accepts a list of dictionaries."""

    class Schema(DataFrameModel):
        state: Series[str]
        city: Series[str]
        price: Series[float]

    raw_data = [
        {"state": "NY", "format": "New York", "price": 8.0},
        {"state": "FL", "format": "Miami", "price": 12.0},
    ]
    # Adjusting raw_data to match Schema keys for a passing test
    raw_data = [
        {"state": "NY", "city": "New York", "price": 8.0},
        {"state": "FL", "city": "Miami", "price": 12.0},
    ]

    pandera_validated_df = DataFrame.from_records(Schema, raw_data)
    pandas_df = pd.DataFrame.from_records(raw_data)
    assert pandera_validated_df.equals(Schema.validate(pandas_df))
    assert isinstance(pandera_validated_df, DataFrame)
