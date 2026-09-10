import polars as pl

import pandera.polars as pa

pa.Column(pl.Decimal(precision=38, scale=6))
pa.Column(pl.Struct({"a": pl.Int64()}))
