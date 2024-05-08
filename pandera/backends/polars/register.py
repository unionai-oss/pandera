"""Register polars backends."""

import polars as pl


def register_polars_backends():
    """Register polars backends.

    This function is called at schema initialization in the _register_*_backends
    method.
    """

    # pylint: disable=import-outside-toplevel,unused-import,cyclic-import
    from pandera.api.checks import Check
    from pandera.api.polars.components import Column
    from pandera.api.polars.container import DataFrameSchema
    from pandera.backends.polars import builtin_checks
    from pandera.backends.polars.checks import PolarsCheckBackend
    from pandera.backends.polars.components import ColumnBackend
    from pandera.backends.polars.container import DataFrameSchemaBackend

    DataFrameSchema.register_backend(pl.LazyFrame, DataFrameSchemaBackend)
    Column.register_backend(pl.LazyFrame, ColumnBackend)
    Check.register_backend(pl.LazyFrame, PolarsCheckBackend)
