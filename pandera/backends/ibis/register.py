"""Register Ibis backends."""

import ibis.expr.types as ir


def register_ibis_backends():
    """Register Ibis backends.

    This function is called at schema initialization in the _register_*_backends
    method.
    """

    # pylint: disable=import-outside-toplevel,unused-import,cyclic-import
    from pandera.api.checks import Check
    from pandera.api.ibis.components import Column
    from pandera.api.ibis.container import DataFrameSchema
    from pandera.backends.ibis.components import ColumnBackend
    from pandera.backends.ibis.container import DataFrameSchemaBackend

    DataFrameSchema.register_backend(ir.Table, DataFrameSchemaBackend)
    Column.register_backend(ir.Table, ColumnBackend)
