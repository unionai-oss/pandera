"""Ibis parsing, validation, and error-reporting backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import ibis.expr.types as ir

from pandera.backends.base import CoreCheckResult
from pandera.backends.ibis.base import IbisSchemaBackend
from pandera.error_handlers import SchemaErrorHandler
from pandera.errors import SchemaError, SchemaErrorReason, SchemaErrors

if TYPE_CHECKING:
    from pandera.api.ibis.container import DataFrameSchema


class DataFrameSchemaBackend(IbisSchemaBackend):
    """Backend for Ibis DataFrameSchema."""

    def validate(
        self,
        check_obj: ir.Table,
        schema: DataFrameSchema,
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
        error_handler = SchemaErrorHandler(lazy)

        # collect schema components
        components = schema.columns.values()

        # TODO(deepyaman): subsample the check object if head, tail, or sample are specified
        sample = check_obj

        # run the checks
        core_checks = [
            (self.run_schema_component_checks, (sample, components, lazy)),
        ]

        for check, args in core_checks:
            results = check(*args)
            if isinstance(results, CoreCheckResult):
                results = [results]

            for result in results:
                if result.passed:
                    continue

                if result.schema_error is not None:
                    error = result.schema_error
                else:
                    error = SchemaError(
                        schema,
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

        if error_handler.collected_errors:
            if getattr(schema, "drop_invalid_rows", False):
                check_obj = self.drop_invalid_rows(check_obj, error_handler)
                return check_obj
            else:
                raise SchemaErrors(
                    schema=schema,
                    schema_errors=error_handler.collected_errors,
                    data=check_obj,
                )

        return check_obj

    def run_schema_component_checks(
        self,
        check_obj: ir.Table,
        schema_components: List,
        lazy: bool,
    ) -> List[CoreCheckResult]:
        """Run checks for all schema components."""
        check_results = []
        check_passed = []
        # schema-component-level checks
        for schema_component in schema_components:
            try:
                result = schema_component.validate(
                    check_obj, lazy=lazy, inplace=True
                )
                check_passed.append(isinstance(result, ir.Table))
            except SchemaError as err:
                check_results.append(
                    CoreCheckResult(
                        passed=False,
                        check="schema_component_checks",
                        reason_code=SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                        schema_error=err,
                    )
                )
            except SchemaErrors as err:
                check_results.extend(
                    [
                        CoreCheckResult(
                            passed=False,
                            check="schema_component_checks",
                            reason_code=SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                            schema_error=schema_error,
                        )
                        for schema_error in err.schema_errors
                    ]
                )
        assert all(check_passed)
        return check_results
