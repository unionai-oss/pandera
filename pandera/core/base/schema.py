"""Base functions for Parsing, Validation, and Error Reporting Backends.

This class should implement a common interface of operations needed for
data validation. These operations are exposed as methods that are composed
together to implement the pandera schema specification.
"""

from abc import ABC, abstractmethod, abstractproperty
from functools import singledispatch, wraps
from typing import Any, Dict, List, Optional, Sequence, Type, Union

import pandas as pd

from pandera.core.base.checks import BaseCheck
from pandera.dtypes import DataType

DtypeInputTypes = Union[str, type, DataType, Type]


class BaseSchema(ABC):
    """Core schema specification."""

    def __init__(
        self,
        fields: "BaseSchema" = None,
        dtype: DtypeInputTypes = None,
        checks: List[BaseCheck] = None,
        coerce: bool = False,
        name: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ):
        ...

    # @abstractproperty
    def checks(self):
        ...

    # @abstractproperty
    def name(self):
        ...

    # @abstractproperty
    def dtype(self):
        ...

    # @abstractproperty
    def coerce(self):
        ...

    # @abstractproperty
    def title(self):
        ...

    # @abstractproperty
    def description(self):
        ...

    # @abstractproperty
    def _allow_groupby(self) -> bool:
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

    def update_checks(self):
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

    def strategy_component(self):
        ...


class BaseSchemaModel(ABC):
    """Base class for schemas defined as python classes"""

    ...



def inferred_schema_guard(method):
    """
    Invoking a method wrapped with this decorator will set _is_inferred to
    False.
    """

    @wraps(method)
    def wrapper(schema, *args, **kwargs):
        new_schema = method(schema, *args, **kwargs)
        if new_schema is not None and id(new_schema) != id(schema):
            # if method returns a copy of the schema object,
            # the original schema instance and the copy should be set to
            # not inferred.
            new_schema._is_inferred = False
        return new_schema

    return wrapper
