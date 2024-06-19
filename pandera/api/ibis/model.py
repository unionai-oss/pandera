"""Class-based API for Ibis models."""

import ibis.expr.types as ir

from pandera.api.dataframe.model import DataFrameModel as _DataFrameModel
from pandera.api.ibis.container import DataFrameSchema


class DataFrameModel(_DataFrameModel[ir.Table, DataFrameSchema]):
    """Definition of a :class:`~pandera.api.ibis.container.DataFrameSchema`.

    *new in 0.1815.0*

    See the :ref:`User Guide <dataframe-models>` for more.
    """
