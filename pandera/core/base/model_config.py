"""Class-based dataframe model API configuration."""

from typing import Any, Optional


class BaseModelConfig:  # pylint:disable=R0903
    """Model configuration base class."""

    #: datatype of the data container. This overrides the data types specified
    #: in any of the fields.
    dtype: Optional[Any] = None

    name: Optional[str] = None  #: name of schema
    title: Optional[str] = None  #: human-readable label for schema
    description: Optional[str] = None  #: arbitrary textual description
    coerce: bool = False  #: coerce types of all schema components
