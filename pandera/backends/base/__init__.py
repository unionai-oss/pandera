"""Base classes for parsing, validation, and error Reporting Backends.

These classes implement a common interface of operations needed for
data validation. These operations are exposed as methods that are composed
together to implement the pandera schema specification.
"""

from abc import ABC
from collections.abc import Iterable
from typing import Any, NamedTuple, Optional, Union

# from pandera.api.base.checks import BaseCheck
from pandera.errors import SchemaError, SchemaErrorReason


class CoreCheckResult(NamedTuple):
    """Namedtuple for holding results of core checks."""

    passed: bool
    check: Union[str, "BaseCheck"] | None = None  # type: ignore
    check_index: int | None = None
    check_output: Any | None = None
    reason_code: SchemaErrorReason | None = None
    message: str | None = None
    failure_cases: Any | None = None
    schema_error: SchemaError | None = None
    original_exc: Exception | None = None


class ColumnInfo(NamedTuple):
    """Column metadata used during validation."""

    sorted_column_names: Iterable
    expanded_column_names: frozenset
    destuttered_column_names: list
    absent_column_names: list
    regex_match_patterns: list | None = None
    lazy_exclude_column_names: list | None = None


class CoreParserResult(NamedTuple):
    """Namedtuple for holding core parser results."""

    passed: bool
    parser: Union[str, "BaseParser"] | None = None  # type: ignore
    parser_index: int | None = None
    parser_output: Any | None = None
    reason_code: SchemaErrorReason | None = None
    message: str | None = None
    failure_cases: Any | None = None
    schema_error: SchemaError | None = None
    original_exc: Exception | None = None


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
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
    ):
        """Subsamples a check object before applying check functions."""
        raise NotImplementedError

    def validate(
        self,
        check_obj,
        schema,
        *,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
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
        schema,
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
        self, schema_name: str, schema_errors: list[SchemaError]
    ):
        """Get failure cases metadata for lazy validation."""
        raise NotImplementedError

    def drop_invalid_rows(self, check_obj, error_handler):
        """Remove invalid elements in a `check_obj` according to failures in caught by the `error_handler`"""
        raise NotImplementedError


class BaseCheckBackend(ABC):
    """Abstract base class for a check backend implementation."""

    def __init__(self, check):
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

    def __init__(self, parser):
        """Initializes a parser backend object."""

    def __call__(self, parse_obj, key=None):
        raise NotImplementedError

    def apply(self, parse_obj):
        """Apply the parser function to a parse object."""
        raise NotImplementedError
