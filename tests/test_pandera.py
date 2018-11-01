"""Some unit tests."""

import numpy as np
import pandas as pd

from pandera import DataFrameSchema, SeriesSchema, Column, PandasDtype, \
    validate_dataframe_arg


def test_dataframe_schema():
    schema = DataFrameSchema(
        [
            Column("a", PandasDtype.Int, lambda x: x >= 1),
            Column("a", PandasDtype.Int, lambda series: series.mean(),
                   element_wise=False),
            Column("b", PandasDtype.String, lambda x: x in ["x", "y", "z"]),
            Column("c", PandasDtype.DateTime,
                   lambda x: pd.Timestamp("2018-01-01") <= x),
            Column("d", PandasDtype.Float, lambda x: np.isnan(x) or x < 3)
        ],
        transformer=lambda df: df.assign(e="foo")
    )

    @validate_dataframe_arg("my_dataframe", schema)
    def test_func(x, my_dataframe, y=1, z=100):
        print(x, y, z)
        print(my_dataframe)

    df = pd.DataFrame({
      "a": [1, 2, 3],
      "b": ["x", "y", "z"],
      "c": [pd.Timestamp("2018-01-01"),
            pd.Timestamp("2018-01-03"),
            pd.Timestamp("2018-01-02")],
      "d": [np.nan, 1.0, 2.0],
    })
    test_func("foo", df)


def test_series_schema():
    schema = SeriesSchema(PandasDtype.Int, lambda x: 0 <= x <= 100)
    schema.validate(pd.Series([0, 30, 50, 100]))
