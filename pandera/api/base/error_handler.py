"""Handle schema errors."""

from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Union

from pandera.api.checks import Check
from pandera.validation_depth import validation_type
from pandera.errors import SchemaError, SchemaErrorReason


class ErrorCategory(Enum):
    """Error category codes"""

    DATA = "data-failures"
    SCHEMA = "schema-failures"
    DTYPE_COERCION = "dtype-coercion-failures"


class ErrorHandler:
    """Handler for Schema & Data level errors during validation."""

    def __init__(self, lazy: bool = True) -> None:
        """Initialize ErrorHandler.

        :param lazy: if True, lazily evaluates all checks before raising the exception.
        Defaults to True.
        """
        self._lazy = lazy
        self._collected_errors: List[Dict[str, Any]] = []
        self._schema_errors: List[SchemaError] = []
        self._summarized_errors = defaultdict(lambda: defaultdict(list))  # type: ignore

    @property
    def lazy(self) -> bool:
        """Whether or not the schema error handler raises errors immediately."""
        return self._lazy

    def collect_error(
        self,
        error_type: ErrorCategory,
        reason_code: SchemaErrorReason,
        schema_error: SchemaError,
        original_exc: Union[BaseException, None] = None,
    ):
        """Collect schema error, raising exception if lazy is False.

        :param error_type: type of error
        :param reason_code: string representing reason for error
        :param schema_error: ``SchemaError`` object.
        """
        if not self._lazy:
            raise schema_error from original_exc

        # delete data of validated object from SchemaError object to prevent
        # storing copies of the validated DataFrame/Series for every
        # SchemaError collected.
        if hasattr(schema_error, "data"):
            del schema_error.data

        schema_error.data = None

        self._schema_errors.append(schema_error)

        failure_cases_count = len(getattr(schema_error, "failure_cases", []))

        self._collected_errors.append(
            {
                "type": error_type,
                "column": schema_error.schema.name,
                "check": schema_error.check,
                "reason_code": reason_code,
                "error": schema_error,
                "failure_cases_count": failure_cases_count,
            }
        )

    def collect_errors(
        self,
        schema_errors: List[SchemaError],
        original_exc: Union[BaseException, None] = None,
    ):
        """Collect schema errors from a SchemaErrors exception.

        :param reason_code: string representing reason for error.
        :param schema_error: ``SchemaError`` object.
        :param original_exc: original exception associated with the SchemaError.
        """
        for schema_error in schema_errors:
            self.collect_error(
                validation_type(schema_error.reason_code),
                schema_error.reason_code,
                schema_error,
                original_exc or schema_error,
            )

    @property
    def collected_errors(self) -> List[Dict[str, Any]]:
        """Retrieve error objects collected during lazy validation."""
        return self._collected_errors

    @collected_errors.setter
    def collected_errors(self, value: List[Dict[str, Any]]):
        """Set the list of collected errors."""
        if not isinstance(value, list):
            raise ValueError("collected_errors must be a list")
        self._collected_errors = value

    @property
    def schema_errors(self) -> List[SchemaError]:
        """Retrieve SchemaError objects collected during lazy validation."""
        return self._schema_errors

    def summarize(self, schema_name):
        """Collect schema error, raising exception if lazy is False.

        :param schema: schema object
        """

        for e in self._collected_errors:
            category = e["type"].name
            subcategory = e["reason_code"].name
            error = e["error"]

            if isinstance(error.check, Check):
                check = error.check.error
            else:
                check = error.check

            self._summarized_errors[category][subcategory].append(
                {
                    "schema": schema_name,
                    "column": e["column"],
                    "check": check,
                    "error": error.__str__().replace("\n", ""),
                }
            )

        return self._summarized_errors
