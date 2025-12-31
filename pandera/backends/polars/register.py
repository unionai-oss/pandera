"""Register polars backends."""

from functools import lru_cache
from typing import Optional

import polars as pl


@lru_cache
def register_polars_backends(
    check_cls_fqn: str | None = None,
):
    """Register polars backends.

    This function is called at schema initialization in the _register_*_backends
    method.
    """

    from pandera.api.checks import Check
    from pandera.api.polars.components import Column
    from pandera.api.polars.container import DataFrameSchema
    from pandera.backends.polars import builtin_checks
    from pandera.backends.polars.checks import PolarsCheckBackend
    from pandera.backends.polars.components import ColumnBackend
    from pandera.backends.polars.container import DataFrameSchemaBackend

    DataFrameSchema.register_backend(pl.LazyFrame, DataFrameSchemaBackend)
    DataFrameSchema.register_backend(pl.DataFrame, DataFrameSchemaBackend)
    Column.register_backend(pl.LazyFrame, ColumnBackend)
    Check.register_backend(pl.LazyFrame, PolarsCheckBackend)
