"""Base functions for Parsing, Validation, and Error Reporting Backends.

This class should implement a common interface of operations needed for
data validation. These operations are exposed as methods that are composed
together to implement the pandera schema specification.
"""

from abc import ABC
from functools import wraps
from typing import Type, Union

from pandera.dtypes import DataType

DtypeInputTypes = Union[str, type, DataType, Type]


class BaseSchema(ABC):
    """Core schema specification."""

    def __init__(
        self,
        dtype=None,
        checks=None,
        coerce=False,
        name=None,
        title=None,
        description=None,
    ):
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
        raise NotImplementedError

    def coerce_dtype(self, check_obj):
        raise NotImplementedError

    @property
    def properties(self):
        raise NotImplementedError


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
