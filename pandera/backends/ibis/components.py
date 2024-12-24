"""Backend implementation for Ibis schema components."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, cast

import ibis.expr.types as ir

from pandera.api.base.error_handler import ErrorHandler
from pandera.backends.base import CoreCheckResult
from pandera.backends.ibis.base import IbisSchemaBackend
from pandera.backends.pandas.error_formatters import scalar_failure_case
from pandera.config import ValidationScope
from pandera.engines.ibis_engine import Engine
from pandera.errors import SchemaError, SchemaErrorReason, SchemaErrors
from pandera.validation_depth import validate_scope, validation_type

if TYPE_CHECKING:
    from pandera.api.ibis.components import Column


class ColumnBackend(IbisSchemaBackend):
    """Backend implementation for Ibis table columns."""

    def validate(
        self,
        check_obj: ir.Table,
        schema: Column,
        *,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> ir.Table:
        """Validation backend implementation for Ibis table columns."""
        error_handler = ErrorHandler(lazy)

        # TODO(deepyaman): subsample the check object if head, tail, or sample are specified
        sample = check_obj[schema.name]

        # run the checks
        core_checks = [
            (self.check_dtype, (sample, schema)),
        ]

        for check, args in core_checks:
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

        if lazy and error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.collected_errors,
                data=check_obj,
            )

        return check_obj

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_dtype(
        self, check_obj: ir.Column, schema: Column
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
                failure_cases = scalar_failure_case(str(check_obj.type()))
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
