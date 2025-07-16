"""Register narwhals backends."""

from functools import lru_cache
from typing import Any

import narwhals as nw

from pandera.backends.base import BACKEND_REGISTRY


@lru_cache(maxsize=1)
def register_narwhals_backends():
    """Register narwhals backends."""
    # pylint: disable=import-outside-toplevel
    from pandera.api.narwhals.container import DataFrameSchema
    from pandera.api.narwhals.components import Column
    from pandera.backends.narwhals.container import DataFrameSchemaBackend
    from pandera.backends.narwhals.components import ColumnBackend
    from pandera.backends.narwhals.checks import NarwhalsCheckBackend

    # Register DataFrame backends
    BACKEND_REGISTRY.register(
        DataFrameSchema,
        nw.DataFrame,
        DataFrameSchemaBackend,
    )
    
    BACKEND_REGISTRY.register(
        DataFrameSchema,
        nw.LazyFrame,
        DataFrameSchemaBackend,
    )

    # Register Column backends
    BACKEND_REGISTRY.register(
        Column,
        nw.DataFrame,
        ColumnBackend,
    )
    
    BACKEND_REGISTRY.register(
        Column,
        nw.LazyFrame,
        ColumnBackend,
    )

    # Register Check backends
    from pandera.api.checks import Check
    
    BACKEND_REGISTRY.register(
        Check,
        nw.DataFrame,
        NarwhalsCheckBackend,
    )
    
    BACKEND_REGISTRY.register(
        Check,
        nw.LazyFrame,
        NarwhalsCheckBackend,
    )