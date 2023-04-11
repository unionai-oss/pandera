"""Handle schema errors."""

from typing import List, Optional

from pandera.errors import SchemaError, SchemaErrors, SchemaErrorReason


class SchemaErrorHandler:
    """Handler for SchemaError objects during validation."""

    def __init__(self, lazy: bool) -> None:
        """Initialize SchemaErrorHandler.

        :param lazy: if True, lazily evaluates schema checks and stores
            SchemaError objects. Otherwise raise a SchemaError immediately.
        """
        self._lazy = lazy
        self._collected_errors: List[SchemaError] = []  # type: ignore

    @property
    def lazy(self) -> bool:
        """Whether or not the schema error handler raises errors immediately."""
        return self._lazy

    def collect_error(
        self,
        reason_code: Optional[SchemaErrorReason],
        schema_error: SchemaError,
        original_exc: BaseException = None,
    ):
        """Collect schema error, raising exception if lazy is False.

        :param reason_code: string representing reason for error.
        :param schema_error: ``SchemaError`` object.
        :param original_exc: original exception associated with the SchemaError.
        """
        if not self._lazy:
            raise schema_error from original_exc

        # delete data of validated object from SchemaError object to prevent
        # storing copies of the validated DataFrame/Series for every
        # SchemaError collected.
        del schema_error.data
        schema_error.data = None

        if reason_code is not None:
            schema_error.reason_code = reason_code

        self._collected_errors.append(schema_error)

    def collect_errors(
        self,
        schema_errors: SchemaErrors,
        original_exc: BaseException = None,
    ):
        """Collect schema errors from a SchemaErrors exception.

        :param reason_code: string representing reason for error.
        :param schema_error: ``SchemaError`` object.
        :param original_exc: original exception associated with the SchemaError.
        """
        for schema_error in schema_errors.schema_errors:
            self.collect_error(
                schema_error.reason_code,
                schema_error,
                original_exc or schema_errors,
            )

    @property
    def collected_errors(self) -> List[SchemaError]:
        """Retrieve SchemaError objects collected during lazy validation."""
        return self._collected_errors
