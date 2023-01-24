"""Base functions for Parsing, Validation, and Error Reporting Backends.

This class should implement a common interface of operations needed for
data validation. These operations are exposed as methods that are composed
together to implement the pandera schema specification.
"""

from abc import ABC
from typing import Optional


class BaseSchemaBackend(ABC):
    """Abstract base class for a schema backend implementation."""

    def preprocess(
        self,
        check_obj,
        inplace: bool = False,
    ):
        """Preprocesses a check object before applying check functions."""
        raise NotImplementedError

    def subsample(
        self,
        check_obj,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        """Subsamples a check object before applying check functions."""
        raise NotImplementedError

    def validate(
        self,
        check_obj,
        schema,
        *,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ):
        """
        Parse and validate a check object, returning type-coerced and validated
        object.
        """
        raise NotImplementedError

    def coerce_dtype(
        self,
        check_obj,
        *,
        schema=None,
        error_handler=None,
    ):
        """Coerce the data type of the check object."""
        raise NotImplementedError

    def run_check(
        self,
        check_obj,
        schema,
        check,
        check_index: int,
        *args,
    ):
        """Run a single check on the check object."""
        raise NotImplementedError

    def run_checks(self, check_obj, schema, error_handler):
        """Run a list of checks on the check object."""
        raise NotImplementedError

    def run_schema_component_checks(
        self,
        check_obj,
        schema_components,
        lazy,
        error_handler,
    ):
        """Run checks for all schema components."""
        raise NotImplementedError

    def check_name(self, check_obj, schema):
        """Core check that checks the name of the check object."""
        raise NotImplementedError

    def check_nullable(self, check_obj, schema):
        """Core check that checks the nullability of a check object."""
        raise NotImplementedError

    def check_unique(self, check_obj, schema):
        """Core check that checks the uniqueness of values in a check object."""
        raise NotImplementedError

    def check_dtype(self, check_obj, schema):
        """Core check that checks the data type of a check object."""
        raise NotImplementedError


class BaseCheckBackend(ABC):
    """Abstract base class for a check backend implementation."""

    def __init__(self, check):  # pylint: disable=unused-argument
        """Initializes a check backend object."""
        ...

    def __call__(self, check_obj, key=None):
        raise NotImplementedError

    def query(self, check_obj):
        """Implements querying behavior to produce subset of check object."""
        raise NotImplementedError

    def groupby(self, check_obj):
        """Implements groupby behavior for check object."""
        raise NotImplementedError

    def aggregate(self, check_obj):
        """Implements aggregation behavior for check object."""
        raise NotImplementedError

    def preprocess(self, check_obj, key):
        """Preprocesses a check object before applying the check function."""
        raise NotImplementedError

    def postprocess(self, check_obj, check_output):
        """Postprocesses the result of applying the check function."""
        raise NotImplementedError

    def apply(self, check_obj):
        """Apply the check function to a check object."""
        raise NotImplementedError

    def statistics(self):
        """Check statistics property."""
        raise NotImplementedError

    def strategy(self):
        """Return a data generation strategy."""
        raise NotImplementedError
