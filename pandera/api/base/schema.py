"""Base functions for Parsing, Validation, and Error Reporting Backends.

This class should implement a common interface of operations needed for
data validation. These operations are exposed as methods that are composed
together to implement the pandera schema specification.
"""

import inspect
from abc import ABC
from functools import wraps
from typing import Any, Dict, Tuple, Type, Union

from pandera.backends.base import BaseSchemaBackend
from pandera.errors import BackendNotFoundError
from pandera.dtypes import DataType

DtypeInputTypes = Union[str, type, DataType, Type]


class BaseSchema(ABC):
    """Core schema specification."""

    BACKEND_REGISTRY: Dict[
        Tuple[Type, Type], Type[BaseSchemaBackend]
    ] = {}  # noqa

    def __init__(
        self,
        dtype=None,
        checks=None,
        coerce=False,
        name=None,
        title=None,
        description=None,
    ):
        """Abstract base schema initializer."""
        self.dtype = dtype
        self.checks = checks
        self.coerce = coerce
        self.name = name
        self.title = title
        self.description = description

    def validate(
        self,
        check_obj,
        head=None,
        tail=None,
        sample=None,
        random_state=None,
        lazy=False,
        inplace=False,
    ):
        """Validate method to be implemented by subclass."""
        raise NotImplementedError

    def coerce_dtype(self, check_obj):
        """Coerce object to the expected type."""
        raise NotImplementedError

    @property
    def properties(self):
        """Get the properties of the schema for serialization purposes."""
        raise NotImplementedError

    @classmethod
    def register_backend(cls, type_: Type, backend: Type[BaseSchemaBackend]):
        """Register a schema backend for this class."""
        cls.BACKEND_REGISTRY[(cls, type_)] = backend

    @classmethod
    def get_backend(cls, check_obj: Any) -> BaseSchemaBackend:
        """Get the backend associated with the type of ``check_obj`` ."""
        check_obj_cls = type(check_obj)
        classes = inspect.getmro(check_obj_cls)
        for _class in classes:
            try:
                return cls.BACKEND_REGISTRY[(cls, _class)]()
            except KeyError:
                pass
        raise BackendNotFoundError(
            f"Backend not found for backend, class: {(cls, check_obj_cls)}. "
            f"Looked up the following base classes: {classes}"
        )


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
