"""Register narwhals backends."""

from functools import lru_cache
from typing import Any, Optional

import narwhals as nw


@lru_cache(maxsize=1)
def register_narwhals_backends(
    check_cls_fqn: Optional[str] = None,
):  # pylint: disable=unused-argument
    """Register narwhals backends."""
    # pylint: disable=import-outside-toplevel
    from pandera.api.narwhals.container import DataFrameSchema
    from pandera.api.narwhals.components import Column
    from pandera.backends.narwhals.container import DataFrameSchemaBackend
    from pandera.backends.narwhals.components import ColumnBackend
    from pandera.backends.narwhals.checks import NarwhalsCheckBackend
    from pandera.api.checks import Check

    # Register DataFrame backends
    DataFrameSchema.register_backend(nw.DataFrame, DataFrameSchemaBackend)
    DataFrameSchema.register_backend(nw.LazyFrame, DataFrameSchemaBackend)

    # Register Column backends
    Column.register_backend(nw.DataFrame, ColumnBackend)
    Column.register_backend(nw.LazyFrame, ColumnBackend)

    # Register Check backends
    Check.register_backend(nw.DataFrame, NarwhalsCheckBackend)
    Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend)
