"""Ibis parsing, validation, and error-reporting backends."""

import warnings
from typing import List

import ibis.expr.types as ir

from pandera.api.ibis.types import CheckResult
from pandera.backends.base import BaseSchemaBackend, CoreCheckResult
from pandera.backends.pandas.error_formatters import (
    consolidate_failure_cases,
    format_generic_error_message,
    format_vectorized_error_message,
    reshape_failure_cases,
    scalar_failure_case,
    summarize_failure_cases,
)
from pandera.errors import (
    FailureCaseMetadata,
    SchemaError,
    SchemaErrorReason,
    SchemaWarning,
)


class IbisSchemaBackend(BaseSchemaBackend):
    """Base backend for Ibis schemas."""

    def run_check(
        self,
        check_obj: ir.Table,
        schema,
        check,
        check_index: int,
        *args,
    ) -> CoreCheckResult:
        """Handle check results, raising SchemaError on check failure.

        :param check_obj: data object to be validated.
        :param schema: pandera schema object.
        :param check: Check object used to validate Ibis object.
        :param check_index: index of check in the schema component check list.
        :param args: arguments to pass into check object.
        :returns: True if check results pass or check.raise_warning=True, otherwise
            False.
        """
        check_result: CheckResult = check(check_obj, *args)

        passed = check_result.check_passed.execute()
        failure_cases = None
        message = None

        if not passed:
            if check_result.failure_cases is None:
                # encode scalar False values explicitly
                failure_cases = scalar_failure_case(check_result.check_passed)
                message = format_generic_error_message(
                    schema, check, check_index
                )
            else:
                failure_cases = reshape_failure_cases(
                    check_result.failure_cases.to_pandas(), check.ignore_na
                )
                message = format_vectorized_error_message(
                    schema, check, check_index, failure_cases
                )

            # raise a warning without exiting if the check is specified to do so
            # but make sure the check passes
            if check.raise_warning:
                warnings.warn(
                    message,
                    SchemaWarning,
                )
                return CoreCheckResult(
                    passed=True,
                    check=check,
                    reason_code=SchemaErrorReason.DATAFRAME_CHECK,
                )

        return CoreCheckResult(
            passed=passed,
            check=check,
            check_index=check_index,
            check_output=check_result.check_output,
            reason_code=SchemaErrorReason.DATAFRAME_CHECK,
            message=message,
            failure_cases=failure_cases,
        )

    def failure_cases_metadata(
        self,
        schema_name: str,
        schema_errors: List[SchemaError],
    ) -> FailureCaseMetadata:
        """Create failure cases metadata required for SchemaErrors exception."""
        failure_cases = consolidate_failure_cases(schema_errors)
        message, error_counts = summarize_failure_cases(
            schema_name, schema_errors, failure_cases
        )
        return FailureCaseMetadata(
            failure_cases=failure_cases,
            message=message,
            error_counts=error_counts,
        )
