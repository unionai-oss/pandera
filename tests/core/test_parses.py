"""Tests the way Columns are Parsed"""

import pandas as pd
import numpy as np

from pandera.api.pandas.container import DataFrameSchema
from pandera.api.parsers import Parser


def test_dataframe_schema_parse() -> None:
    """Test that DataFrameSchema-level Parses work properly."""
    data = pd.DataFrame([[1, 4, 9, 16, 25] for _ in range(10)])

    schema_check_return_bool = DataFrameSchema(
        parsers=Parser(lambda df: df.transform("sqrt"))
    )
    assert schema_check_return_bool.validate(data).equals(
        data.applymap(np.sqrt)
    )
