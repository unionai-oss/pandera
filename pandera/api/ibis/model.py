"""Class-based API for Ibis models."""

from typing import (
    Any,
    Callable,
    Dict,
    Tuple,
    Type,
    TypeVar,
)

from pandera.api.base.model import BaseModel
from pandera.api.ibis.container import DataFrameSchema

_CONFIG_KEY = "Config"

MODEL_CACHE: Dict[Type["DataFrameModel"], DataFrameSchema] = {}
GENERIC_SCHEMA_CACHE: Dict[
    Tuple[Type["DataFrameModel"], Tuple[Type[Any], ...]],
    Type["DataFrameModel"],
] = {}

F = TypeVar("F", bound=Callable)
TDataFrameModel = TypeVar("TDataFrameModel", bound="DataFrameModel")


class DataFrameModel(BaseModel):
    """Definition of a :class:`~pandera.api.ibis.container.DataFrameSchema`.

    *new in 0.1815.0*

    See the :ref:`User Guide <dataframe_models>` for more.
    """
