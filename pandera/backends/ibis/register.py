"""Register Ibis backends."""

import ibis


def register_ibis_backends():
    """Register Ibis backends.

    This function is called at schema initialization in the _register_*_backends
    method.
    """

    from pandera.api.checks import Check
    from pandera.api.ibis.components import Column
    from pandera.api.ibis.container import DataFrameSchema
    from pandera.backends.ibis import builtin_checks
    from pandera.backends.ibis.checks import IbisCheckBackend
    from pandera.backends.ibis.components import ColumnBackend
    from pandera.backends.ibis.container import DataFrameSchemaBackend

    DataFrameSchema.register_backend(ibis.Table, DataFrameSchemaBackend)
    Column.register_backend(ibis.Table, ColumnBackend)
    Check.register_backend(ibis.Table, IbisCheckBackend)
    Check.register_backend(ibis.Column, IbisCheckBackend)
