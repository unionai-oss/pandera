"""Class-based dataframe model API configuration for pandas."""

from typing import Optional

from pandera.api.dataframe.model_config import BaseConfig as _BaseConfig
from pandera.api.pandas.types import PandasDtypeInputTypes


class BaseConfig(_BaseConfig):  # pylint:disable=R0903
    """Define pandas DataFrameSchema-wide options."""

    #: datatype of the dataframe. This overrides the data types specified in
    #: any of the fields.
    dtype: Optional[PandasDtypeInputTypes] = None
