"""Polars backend implementation for schemas and checks."""

import polars as pl

from pandera.api.polars.container import DataFrameSchema
from pandera.api.polars.components import Column
from pandera.backends.polars.container import DataFrameSchemaBackend
from pandera.backends.polars.components import ColumnBackend


DataFrameSchema.register_backend(pl.LazyFrame, DataFrameSchemaBackend)
Column.register_backend(pl.LazyFrame, ColumnBackend)
