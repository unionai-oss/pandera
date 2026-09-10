"""Class-based dataframe model API configuration for pyarrow."""

from __future__ import annotations

from pandera.api.dataframe.model_config import BaseConfig as _BaseConfig
from pandera.api.pyarrow.types import PyArrowDtypeInputTypes


class BaseConfig(_BaseConfig):
    """Define pyarrow DataFrameSchema-wide options."""

    #: datatype of the table. This overrides the data types specified in
    #: any of the fields.
    dtype: PyArrowDtypeInputTypes | None = None
