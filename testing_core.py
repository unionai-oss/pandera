import pandas as pd

import pandera as pa
import pandera.core.pandas.checks as C
from pandera.core.checks import Check
from pandera.core.pandas import (
    Column,
    DataFrameSchema,
    Index,
    MultiIndex,
    SeriesSchema,
)

series_schema = SeriesSchema(int, pa.Check.gt(0), name="foo")

data = pd.Series([1, 2, 3], name="foo")
print(series_schema(data))


dataframe_schema = DataFrameSchema(
    {
        "col1": Column(int),
        "col2": Column(float),
        "col3": Column(str),
    },
    index=Index(str),
)

df = pd.DataFrame(
    {"col1": [1, 2, 3], "col2": [4.0, 5.0, 6.0], "col3": [*"abc"]},
    index=[*"abc"],
)
print(dataframe_schema(df))


df_multiiindex_schema = DataFrameSchema(
    {
        "col1": Column(int),
        "col2": Column(float),
        "col3": Column(str),
    },
    index=MultiIndex(
        [
            Index(int),
            Index(int),
            Index(int),
        ]
    ),
)

df_multiindex = pd.DataFrame(
    {"col1": [1, 2, 3], "col2": [4.0, 5.0, 6.0], "col3": [*"abc"]},
    index=pd.MultiIndex.from_arrays(
        [
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ]
    ),
)
print(df_multiiindex_schema(df_multiindex))


eq = Check.eq(0)
other_eq = C.eq(0)
print(eq(pd.Series([0, -1])))
print(eq(pd.DataFrame([[0, -1]])))
print(other_eq(pd.DataFrame([[0, -1]])))
print(Check.eq.__signature__)
print(Check.eq.__doc__)
print(Check.eq.__annotations__)
