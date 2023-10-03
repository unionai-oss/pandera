"""Validation backend for polars components."""

from typing import List, Optional, cast

import polars as pl

from pandera.api.polars.components import Column
from pandera.backends.base import BaseSchemaBackend, CoreCheckResult
from pandera.backends.polars.base import PolarsSchemaBackend
from pandera.error_handlers import SchemaErrorHandler
from pandera.errors import (
    SchemaError,
    SchemaErrors,
    SchemaErrorReason,
    FailureCaseMetadata,
)


class ColumnBackend(PolarsSchemaBackend):
    def validate(
        self,
        check_obj: pl.LazyFrame,
        schema: Column,
        *,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> pl.LazyFrame:

        error_handler = SchemaErrorHandler(lazy)

        core_checks = [
            (self.check_dtype, (check_obj, schema)),
            (self.run_checks, (check_obj, schema)),
        ]

        for core_check, args in core_checks:
            results = core_check(*args)
            if isinstance(results, CoreCheckResult):
                results = [results]
            results = cast(List[CoreCheckResult], results)
            for result in results:
                if result.passed:
                    continue

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
                        result.reason_code,
                        error,
                        original_exc=result.original_exc,
                    )

        if lazy and error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.collected_errors,
                data=check_obj,
            )

        return check_obj

    def check_dtype(
        self,
        check_obj: pl.LazyFrame,
        schema: Column,
    ) -> CoreCheckResult:

        passed = True
        failure_cases = None
        msg = None

        if schema.dtype is not None:
            obj_dtype = check_obj.schema[schema.name]
            passed = obj_dtype.is_(schema.dtype)

        if not passed:
            failure_cases = str(obj_dtype)
            msg = (
                f"expected column '{schema.name}' to have type "
                f"{schema.dtype}, got {obj_dtype}"
            )

        return CoreCheckResult(
            passed=passed,
            check=f"dtype('{schema.dtype}')",
            reason_code=SchemaErrorReason.WRONG_DATATYPE,
            message=msg,
            failure_cases=failure_cases,
        )

    # pylint: disable=unused-argument
    def run_checks(self, check_obj, schema) -> List[CoreCheckResult]:
        check_results: List[CoreCheckResult] = []
        for check_index, check in enumerate(schema.checks):
            check_args = [schema.name]  # pass in column key
            try:
                check_results.append(
                    self.run_check(
                        check_obj,
                        schema,
                        check,
                        check_index,
                        *check_args,
                    )
                )
            except Exception as err:  # pylint: disable=broad-except
                # catch other exceptions that may occur when executing the Check
                err_msg = f'"{err.args[0]}"' if len(err.args) > 0 else ""
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
