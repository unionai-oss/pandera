"""Backend implementation for Ibis schema components."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional, cast

import ibis
import ibis.expr.operations as ops
import ibis.selectors as s

from pandera.api.base.error_handler import get_error_category
from pandera.api.ibis.error_handler import ErrorHandler
from pandera.backends.base import CoreCheckResult
from pandera.backends.ibis.base import IbisSchemaBackend
from pandera.config import ValidationScope
from pandera.constants import CHECK_OUTPUT_KEY
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
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
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
                    self.check_nullable,
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
                        results = cast(list[CoreCheckResult], results)
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
                        error_handler.collect_error(
                            get_error_category(result.reason_code),
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

    @validate_scope(scope=ValidationScope.DATA)
    def check_nullable(
        self, check_obj: ibis.Column, schema: Column
    ) -> CoreCheckResult:
        """Check if a column is nullable.

        This check considers nulls and nan values as effectively equivalent.
        """
        if schema.nullable:
            return CoreCheckResult(
                passed=True,
                check="not_nullable",
                reason_code=SchemaErrorReason.SERIES_CONTAINS_NULLS,
            )

        isna = check_obj.isnull()
        if check_obj.type().is_floating() and ibis.get_backend(
            check_obj
        ).has_operation(ops.IsNan):
            isna |= check_obj.isnan()

        passed = (~isna).all().to_pyarrow().as_py()
        check_output = isna.name(CHECK_OUTPUT_KEY)
        failure_cases = check_obj.as_table().filter(check_output)
        return CoreCheckResult(
            passed=passed,
            check_output=check_output,
            check="not_nullable",
            reason_code=SchemaErrorReason.SERIES_CONTAINS_NULLS,
            message=f"non-nullable column '{schema.name}' contains null values",
            failure_cases=failure_cases,
        )

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
    def run_checks(self, check_obj, schema) -> list[CoreCheckResult]:
        check_results: list[CoreCheckResult] = []
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
            except Exception as err:
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
