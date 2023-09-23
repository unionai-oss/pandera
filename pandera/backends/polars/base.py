"""Polars Parsing, Validation, and Error Reporting Backends."""

import warnings
from collections import defaultdict
from typing import List, Dict

import polars as pl
from pandera.api.polars.types import CheckResult
from pandera.backends.base import BaseSchemaBackend, CoreCheckResult
from pandera.backends.pandas.error_formatters import (
    format_generic_error_message,
    format_vectorized_error_message,
)
from pandera.errors import (
    SchemaError,
    FailureCaseMetadata,
    SchemaErrorReason,
    SchemaWarning,
)


class PolarsSchemaBackend(BaseSchemaBackend):
    def run_check(
        self,
        check_obj: pl.LazyFrame,
        schema,
        check,
        check_index: int,
        *args,
    ) -> CoreCheckResult:
        """Handle check results, raising SchemaError on check failure.

        :param check_obj: data object to be validated.
        :param schema: pandera schema object
        :param check: Check object used to validate pandas object.
        :param check_index: index of check in the schema component check list.
        :param args: arguments to pass into check object.
        :returns: True if check results pass or check.raise_warning=True, otherwise
            False.
        """
        check_result: CheckResult = check(check_obj, *args)

        passed = check_result.check_passed.collect().item()
        failure_cases = None
        message = None

        # TODO: this needs to collect the actual values
        if not passed:
            if check_result.failure_cases is None:
                # encode scalar False values explicitly
                failure_cases = passed
                message = format_generic_error_message(
                    schema, check, check_index
                )
            else:
                # use check_result
                failure_cases = check_result.failure_cases.collect()
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
            check_output=check_result.check_output.collect(),
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
        error_counts: Dict[str, int] = defaultdict(int)

        failure_case_collection = []

        for err in schema_errors:

            error_counts[err.reason_code] += 1

            check_identifier = (
                None
                if err.check is None
                else err.check
                if isinstance(err.check, str)
                else err.check.error
                if err.check.error is not None
                else err.check.name
                if err.check.name is not None
                else str(err.check)
            )

            if isinstance(err.failure_cases, pl.LazyFrame):
                raise NotImplementedError

            elif isinstance(err.failure_cases, pl.DataFrame):
                err_failure_cases = err.failure_cases.with_columns(
                    schema_context=pl.lit(err.schema.__class__.__name__),
                    column=pl.lit(err.schema.name),
                    check=pl.lit(check_identifier),
                    check_number=pl.lit(err.check_index),
                )

            else:
                scalar_failure_cases = defaultdict(list)
                scalar_failure_cases["schema_context"].append(
                    err.schema.__class__.__name__
                )
                scalar_failure_cases["column"].append(err.schema.name)
                scalar_failure_cases["check"].append(check_identifier)
                scalar_failure_cases["check_number"].append(err.check_index)
                scalar_failure_cases["failure_case"].append(err.failure_cases)
                scalar_failure_cases["index"].append(None)
                err_failure_cases = pl.DataFrame(scalar_failure_cases)

            failure_case_collection.append(err_failure_cases)

        failure_cases = pl.concat(failure_case_collection)

        return FailureCaseMetadata(
            failure_cases=failure_cases,
            message=FAILURE_CASE_TEMPLATE.format(
                schema_name=schema_name,
                error_count=sum(error_counts.values()),
                failure_cases=str(failure_cases),
            ),
            error_counts=error_counts,
        )


FAILURE_CASE_TEMPLATE = """
Schema {schema_name}: A total of {error_count} errors were found.

{failure_cases}
""".strip()
