"""Base functions for Parsing, Validation, and Error Reporting Backends.

This class should implement a common interface of operations needed for
data validation. These operations are exposed as methods that are composed
together to implement the pandera schema specification.
"""

import inspect
import os
from abc import ABC
from functools import wraps
from typing import Any, Dict, Optional, Tuple, Type, Union

from pandera.backends.base import BaseSchemaBackend
from pandera.dtypes import DataType
from pandera.errors import BackendNotFoundError

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
        parsers=None,
        coerce=False,
        name=None,
        title=None,
        description=None,
        metadata=None,
        drop_invalid_rows=False,
    ):
        """Abstract base schema initializer."""
        self.dtype = dtype
        self.checks = checks
        self.coerce = coerce
        self.parsers = parsers
        self.name = name
        self.title = title
        self.description = description
        self.metadata = metadata
        self.drop_invalid_rows = drop_invalid_rows
        self._register_default_backends()

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
        """Validate a DataFrame based on the schema specification.

        :param pd.DataFrame check_obj: the dataframe to be validated.
        :param head: validate the first n rows. Rows overlapping with `tail` or
            `sample` are de-duplicated.
        :param tail: validate the last n rows. Rows overlapping with `head` or
            `sample` are de-duplicated.
        :param sample: validate a random sample of n rows. Rows overlapping
            with `head` or `tail` are de-duplicated.
        :param random_state: random seed for the ``sample`` argument.
        :param lazy: if True, lazily evaluates dataframe against all validation
            checks and raises a ``SchemaErrors``. Otherwise, raise
            ``SchemaError`` as soon as one occurs.
        :param inplace: if True, applies coercion to the object of validation,
            otherwise creates a copy of the data.
        :returns: validated ``DataFrame``

        :raises SchemaError: when ``DataFrame`` violates built-in or custom
            checks.
        """
        raise NotImplementedError

    def coerce_dtype(self, check_obj):
        """Coerce object to the expected type."""
        raise NotImplementedError

    def to_yaml(self, stream: Optional[os.PathLike] = None) -> Optional[str]:
        """Write DataFrameSchema to yaml file."""
        raise NotImplementedError

    @property
    def properties(self):
        """Get the properties of the schema for serialization purposes."""
        raise NotImplementedError

    @classmethod
    def register_backend(cls, type_: Type, backend: Type[BaseSchemaBackend]):
        """Register a schema backend for this class."""
        if (cls, type_) not in cls.BACKEND_REGISTRY:
            cls.BACKEND_REGISTRY[(cls, type_)] = backend

    @classmethod
    def get_backend(
        cls,
        check_obj: Optional[Any] = None,
        check_type: Optional[Type] = None,
    ) -> BaseSchemaBackend:
        """Get the backend associated with the type of ``check_obj`` ."""
        if check_obj is not None:
            check_obj_cls = type(check_obj)
        elif check_type is not None:
            check_obj_cls = check_type
        else:
            raise ValueError(
                "Must pass in one of `check_obj` or `check_type`."
            )
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

    def _register_default_backends(self):
        """Register default backends.

        This method is invoked in the `__init__` method for subclasses that
        implement the API for a specific dataframe object, and should be
        overridden in those subclasses.
        """

    def __setstate__(self, state):
        self.__dict__ = state
        self._register_default_backends()


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
