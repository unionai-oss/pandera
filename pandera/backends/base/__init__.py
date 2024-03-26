"""Base classes for parsing, validation, and error Reporting Backends.

These classes implement a common interface of operations needed for
data validation. These operations are exposed as methods that are composed
together to implement the pandera schema specification.
"""

from abc import ABC
from typing import Any, FrozenSet, Iterable, List, NamedTuple, Optional, Union

# from pandera.api.base.checks import BaseCheck
from pandera.errors import SchemaError, SchemaErrorReason


class CoreCheckResult(NamedTuple):
    """Namedtuple for holding results of core checks."""

    passed: bool
    check: Optional[Union[str, "BaseCheck"]] = None  # type: ignore
    check_index: Optional[int] = None
    check_output: Optional[Any] = None
    reason_code: Optional[SchemaErrorReason] = None
    message: Optional[str] = None
    failure_cases: Optional[Any] = None
    schema_error: Optional[SchemaError] = None
    original_exc: Optional[Exception] = None


class ColumnInfo(NamedTuple):
    """Column metadata used during validation."""

    sorted_column_names: Iterable
    expanded_column_names: FrozenSet
    destuttered_column_names: List
    absent_column_names: List
    regex_match_patterns: List


class CoreParserResult(NamedTuple):
    """Namedtuple for holding core parser results."""

    passed: bool
    parser: Optional[Union[str, "BaseParser"]] = None  # type: ignore
    parser_index: Optional[int] = None
    parser_output: Optional[Any] = None
    reason_code: Optional[SchemaErrorReason] = None
    message: Optional[str] = None
    failure_cases: Optional[Any] = None
    schema_error: Optional[SchemaError] = None
    original_exc: Optional[Exception] = None


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
        schema=None,
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

    def run_checks(self, check_obj, schema):
        """Run a list of checks on the check object."""
        raise NotImplementedError

    def run_schema_component_checks(
        self,
        check_obj,
        schema_components,
        lazy,
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

    def failure_cases_metadata(
        self, schema_name: str, schema_errors: List[SchemaError]
    ):
        """Get failure cases metadata for lazy validation."""
        raise NotImplementedError

    def drop_invalid_rows(self, check_obj, error_handler):
        """Remove invalid elements in a `check_obj` according to failures in caught by the `error_handler`"""
        raise NotImplementedError


class BaseCheckBackend(ABC):
    """Abstract base class for a check backend implementation."""

    def __init__(self, check):  # pylint: disable=unused-argument
        """Initializes a check backend object."""

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


class BaseParserBackend(ABC):
    """Abstract base class for a parser backend implementation."""

    def __init__(self, parser):  # pylint: disable=unused-argument
        """Initializes a parser backend object."""

    def __call__(self, parse_obj, key=None):
        raise NotImplementedError

    def apply(self, parse_obj):
        """Apply the parser function to a parse object."""
        raise NotImplementedError
