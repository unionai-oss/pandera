"""Base functions for Parsing, Validation, and Error Reporting Backends.

This class should implement a common interface of operations needed for
data validation. These operations are exposed as methods that are composed
together to implement the pandera schema specification.
"""

from abc import ABC, abstractmethod, abstractproperty
from functools import singledispatch
from typing import Any, Dict, List, Optional, Sequence, Type, Union

import pandas as pd

from pandera.checks import Check
from pandera.dtypes import DataType

DtypeInputTypes = Union[str, type, DataType, Type]


class BaseSchema(ABC):
    """Core schema specification."""

    def __init__(
        self,
        dtype: DtypeInputTypes = None,
        checks: List["BaseCheck"] = None,
        coerce: bool = False,
        name: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ):
        ...

    @abstractproperty
    def checks(self):
        ...

    @abstractproperty
    def name(self):
        ...

    @abstractproperty
    def dtype(self):
        ...

    @abstractproperty
    def coerce(self):
        ...

    @abstractproperty
    def title(self):
        ...

    @abstractproperty
    def description(self):
        ...

    def validate(self):
        ...

    def coerce_dtype(self):
        ...


class BaseSchemaTransformsMixin(ABC):
    """Core protocol for transforming a schema specification."""

    def add_fields(self):
        ...

    def remove_fields(self):
        ...

    def update_field(self):
        ...

    def update_fields(self):
        ...

    def rename_fields(self):
        ...

    def select_fields(self):
        ...

    def set_index(self):
        ...

    def reset_index(self):
        ...


class BaseSchemaIOMixin(ABC):
    """Core protocol for serializing/deserializing a schema."""

    def to_script(self):
        ...

    def from_yaml(self):
        ...

    def to_yaml(self):
        ...


class BaseSchemaStrategyMixin(ABC):
    """Core protocol for data generation strategies."""

    def strategy(self):
        ...

    def example(self):
        ...


class BaseSchemaComponent(ABC):
    """Core protocol for a schema """
    ...

class BaseSchemaModel(ABC):
    """Base class for schemas defined as python classes"""
    pass


class BaseCheck(ABC):
    """Core check specification."""
    ...


class BaseErrorFormatter(ABC):
    """Core protocol for formatting schema errors."""
    ...
