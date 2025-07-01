"""Backend implementation for Ibis schema components."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Iterable, List, Optional, cast

import ibis
import ibis.selectors as s

from pandera.api.base.error_handler import ErrorHandler
from pandera.backends.base import CoreCheckResult
from pandera.backends.ibis.base import IbisSchemaBackend
from pandera.config import ValidationScope
from pandera.engines.ibis_engine import Engine
from pandera.errors import (
    SchemaDefinitionError,
    SchemaError,
    SchemaErrorReason,
    SchemaErrors,
)
from pandera.validation_depth import validate_scope, validation_type

if TYPE_CHECKING:
    from pandera.api.ibis.components import Column


class ColumnBackend(IbisSchemaBackend):
    """Backend implementation for Ibis table columns."""

    def validate(
        self,
        check_obj: ibis.Table,
        schema: Column,
        *,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> ibis.Table:
        """Validation backend implementation for Ibis table columns."""
        error_handler = ErrorHandler(lazy)

        if inplace:
            warnings.warn("setting inplace=True will have no effect.")

        if schema.name is None:
            raise SchemaDefinitionError(
                "Column schema must have a name specified."
            )

        def validate_column(check_obj, column_name):
            # make sure the schema component mutations are reverted after
            # validation
            _orig_name = schema.name
            _orig_regex = schema.regex

            try:
                # set the column name and regex flag for a single column
                schema.name = column_name
                schema.regex = False

                # TODO(deepyaman): subsample the check object if head, tail, or sample are specified
                sample = check_obj[column_name]

                # run the checks
                core_checks = [
                    self.check_dtype,
                    self.run_checks,
                ]

                args = (sample, schema)
                for check in core_checks:
                    results = check(*args)
                    if isinstance(results, CoreCheckResult):
                        results = [results]

                    for result in results:
                        if result.passed:
                            continue
                        # Why cast `results` only in components.py, not in container.py?
                        results = cast(List[CoreCheckResult], results)
                        if result.schema_error is not None:
                            error = result.schema_error
                        else:
                            error = SchemaError(
                                schema=schema,
                                data=check_obj,
                                message=result.message,
                                failure_cases=result.failure_cases,
                                check=result.check,
                                check_index=result.check_index,
                                check_output=result.check_output,
                                reason_code=result.reason_code,
                            )
                            error_handler.collect_error(  # Why indent (unlike in container.py)?
                                validation_type(result.reason_code),
                                result.reason_code,
                                error,
                                original_exc=result.original_exc,
                            )

            finally:
                # revert the schema component mutations
                schema.name = _orig_name
                schema.regex = _orig_regex

        column_keys_to_check = (
            self.get_regex_columns(schema, check_obj)
            if schema.regex
            else [schema.name]
        )

        for column_name in column_keys_to_check:
            validate_column(check_obj, column_name)

        if lazy and error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.schema_errors,
                data=check_obj,
            )

        return check_obj

    def get_regex_columns(self, schema, check_obj) -> Iterable:
        return check_obj.select(s.matches(schema.selector)).columns

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_dtype(
        self, check_obj: ibis.Column, schema: Column
    ) -> CoreCheckResult:
        passed = True
        failure_cases = None
        msg = None

        if schema.dtype is not None:
            dtype_check_results = schema.dtype.check(
                Engine.dtype(check_obj.type()),
                check_obj,
            )
            if isinstance(dtype_check_results, bool):
                passed = dtype_check_results
                failure_cases = str(check_obj.type())
                msg = (
                    f"expected column '{check_obj.get_name()}' to have type "
                    f"{schema.dtype}, got {check_obj.type()}"
                )
            else:
                raise NotImplementedError

        return CoreCheckResult(
            passed=passed,
            check=f"dtype('{schema.dtype}')",
            reason_code=SchemaErrorReason.WRONG_DATATYPE,
            message=msg,
            failure_cases=failure_cases,
        )

    @validate_scope(scope=ValidationScope.DATA)
    def run_checks(self, check_obj, schema) -> List[CoreCheckResult]:
        check_results: List[CoreCheckResult] = []
        for check_index, check in enumerate(schema.checks):
            try:
                check_results.append(
                    self.run_check(
                        check_obj,
                        schema,
                        check,
                        check_index,
                        schema.selector,
                    )
                )
            except Exception as err:  # pylint: disable=broad-except
                # catch other exceptions that may occur when executing the Check
                err_msg = f'"{err.args[0]}"' if err.args else ""
                msg = f"{err.__class__.__name__}({err_msg})"
                check_results.append(
                    CoreCheckResult(
                        passed=False,
                        check=check,
                        check_index=check_index,
                        reason_code=SchemaErrorReason.CHECK_ERROR,
                        message=msg,
                        failure_cases=msg,
                        original_exc=err,
                    )
                )
        return check_results
