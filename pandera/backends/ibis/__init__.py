"""Ibis backend implementation for schemas and checks."""

import ibis.expr.types as ir

from pandera.api.checks import Check
from pandera.api.ibis.container import DataFrameSchema
from pandera.api.ibis.components import Column
from pandera.backends.ibis.components import ColumnBackend
from pandera.backends.ibis.container import DataFrameSchemaBackend

DataFrameSchema.register_backend(ir.Table, DataFrameSchemaBackend)
Column.register_backend(ir.Table, ColumnBackend)
