"""Base functions for Parsing, Validation, and Error Reporting Backends.

This class should implement a common interface of operations needed for
data validation. These operations are exposed as methods that are composed
together to implement the pandera schema specification.
"""

import inspect
import os
from abc import ABC
from typing import Any, Optional, Union

from typing_extensions import Self

from pandera.backends.base import BaseSchemaBackend
from pandera.dtypes import DataType
from pandera.errors import BackendNotFoundError

DtypeInputTypes = Union[str, type, DataType, type]


class BaseSchema(ABC):
    """Core schema specification."""

    BACKEND_REGISTRY: dict[tuple[type, type], type[BaseSchemaBackend]] = {}  # noqa

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

    def to_yaml(self, stream: os.PathLike | None = None) -> str | None:
        """Write DataFrameSchema to yaml file."""
        raise NotImplementedError

    @property
    def properties(self):
        """Get the properties of the schema for serialization purposes."""
        raise NotImplementedError

    @classmethod
    def register_backend(cls, type_: type, backend: type[BaseSchemaBackend]):
        """Register a schema backend for this class."""
        if (cls, type_) not in cls.BACKEND_REGISTRY:
            cls.BACKEND_REGISTRY[(cls, type_)] = backend

    @classmethod
    def get_backend(
        cls,
        check_obj: Any | None = None,
        check_type: type | None = None,
    ) -> BaseSchemaBackend:
        """Get the backend associated with the type of ``check_obj``."""
        if check_obj is not None:
            check_obj_cls = type(check_obj)
        elif check_type is not None:
            check_obj_cls = check_type
        else:
            raise ValueError(
                "Must pass in one of `check_obj` or `check_type`."
            )

        cls.register_default_backends(check_obj_cls)
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

    @staticmethod
    def register_default_backends(check_obj_cls: type):
        """Register default backends.

        This method is invoked in the `get_backend` method so that the
        appropriate validation backend is loaded at validation time instead of
        schema-definition time.

        This method needs to be implemented by the schema subclass.
        """

    def __setstate__(self, state):
        self.__dict__ = state

    def set_name(self, name: str) -> Self:
        """Used to set or modify the name of a base model object.

        :param str name: the name of the column object

        """
        self.name = name
        return self

    def strategy(self, *, size: int | None = None, n_regex_columns: int = 1):
        """Create a data synthesis strategy."""
        raise NotImplementedError

    def example(
        self, size: int | None = None, n_regex_columns: int = 1
    ) -> Any:
        """Generate an example of this data model specification."""
        raise NotImplementedError
