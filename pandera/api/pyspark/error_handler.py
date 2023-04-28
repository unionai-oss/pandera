"""Handle schema errors."""

from enum import Enum
from typing import Dict, List, Union
from collections import defaultdict

from pandera.errors import SchemaError, SchemaErrorReason


class ErrorCategory(Enum):
    """Error category codes"""

    DATA = "data-failures"
    SCHEMA = "schema-failures"
    DTYPE_COERCION = "dtype-coercion-failures"


class ErrorHandler:
    """Handler for Schema & Data level errors during validation."""

    def __init__(self, lazy: bool) -> None:
        """Initialize ErrorHandler.

        :param lazy: if True, lazily evaluates all checks before raising the exception.
        """
        self._lazy = lazy
        self._collected_errors = []  # type: ignore
        self._summarized_errors = defaultdict(lambda: defaultdict(list))

    @property
    def lazy(self) -> bool:
        """Whether or not the schema error handler raises errors immediately."""
        return self._lazy

    def collect_error(
        self,
        type: ErrorCategory,
        reason_code: SchemaErrorReason,
        schema_error: SchemaError,
        original_exc: BaseException = None,
    ):
        """Collect schema error, raising exception if lazy is False.

        :param type: type of error
        :param reason_code: string representing reason for error
        :param schema_error: ``SchemaError`` object.
        """
        if not self._lazy:
            raise schema_error from original_exc

        # delete data of validated object from SchemaError object to prevent
        # storing copies of the validated DataFrame/Series for every
        # SchemaError collected.
        del schema_error.data
        schema_error.data = None

        self._collected_errors.append(
            {
                "type": type,
                "column": schema_error.schema.name,
                "check": schema_error.check,
                "reason_code": reason_code,
                "error": schema_error,
            }
        )

    @property
    def collected_errors(self) -> List[Dict[str, Union[SchemaError, str]]]:
        """Retrieve SchemaError objects collected during lazy validation."""
        return self._collected_errors

    def summarize(self, schema):
        """Collect schema error, raising exception if lazy is False.

        :param schema: schema object
        """

        for error in self._collected_errors:
            err_cat = error["reason_code"].name
            err_type = error["type"].name
            self._summarized_errors[err_type][err_cat].append(
                {
                    "schema": schema.name,
                    "column": error["column"],
                    "check": error["error"].check,
                    "error": error["error"].__str__(),
                }
            )

        return self._summarized_errors
