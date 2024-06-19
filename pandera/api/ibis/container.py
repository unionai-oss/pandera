"""Core Ibis table container specification."""

from typing import TYPE_CHECKING

from pandera.api.dataframe.container import DataFrameSchema as _DataFrameSchema

if TYPE_CHECKING:
    import ibis.expr.type as ir


class DataFrameSchema(_DataFrameSchema[ir.Table]):
    """A lightweight Ibis table validator."""

    ...
