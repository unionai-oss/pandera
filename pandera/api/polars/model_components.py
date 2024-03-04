"""DataFrameModel components"""
from functools import partial

from pandera.api.dataframe.model_components import (
    FieldInfo as _FieldInfo,
    _field,
)
from pandera.api.polars.components import Column


CHECK_KEY = "__check_config__"
DATAFRAME_CHECK_KEY = "__dataframe_check_config__"


class FieldInfo(_FieldInfo[Column, None]):
    """Captures extra information about a field.

    *new in 0.5.0*
    """

    __column_cls__ = Column


Field = partial(_field, FieldInfo)
