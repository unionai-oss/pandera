import pandas as pd

import pandera as pa
from pandera.core.pandas import Column, DataFrameSchema, SeriesSchema, Index, MultiIndex

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
    {
        "col1": [1, 2, 3],
        "col2": [4.0, 5.0, 6.0],
        "col3": [*"abc"]
    },
    index=[*"abc"],
)
print(dataframe_schema(df))


df_multiiindex_schema = DataFrameSchema(
    {
        "col1": Column(int),
        "col2": Column(float),
        "col3": Column(str),
    },
    index=MultiIndex([
        Index(int),
        Index(int),
        Index(int),
    ]),
)

df_multiindex = pd.DataFrame(
    {
        "col1": [1, 2, 3],
        "col2": [4.0, 5.0, 6.0],
        "col3": [*"abc"]
    },
    index=pd.MultiIndex.from_arrays(
        [
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ]
    )
)
print(df_multiiindex_schema(df_multiindex))
