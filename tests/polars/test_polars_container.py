"""Unit tests for polars container."""

import polars as pl


def test_polars_lazy_dataframe():
    """Test basic polars lazy dataframe."""
    query = pl.DataFrame({"foo": ["a", "b", "c"], "bar": [0, 1, 2]}).lazy()
    df = query.collect()
    print(df)
