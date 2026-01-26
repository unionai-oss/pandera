"""Ibis parsing, validation, and error-reporting backends."""

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING

import ibis
import ibis.selectors as s
import pandas as pd

from pandera.api.checks import Check
from pandera.api.ibis.error_handler import ErrorHandler
from pandera.api.ibis.types import CheckResult
from pandera.backends.base import BaseSchemaBackend, CoreCheckResult
from pandera.backends.ibis.constants import POSITIONAL_JOIN_BACKENDS
from pandera.backends.pandas.error_formatters import (
    consolidate_failure_cases,
    format_generic_error_message,
    format_vectorized_error_message,
    reshape_failure_cases,
)
from pandera.constants import CHECK_OUTPUT_KEY, CHECK_OUTPUT_SUFFIX
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
        check_obj: ibis.Table,
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
                failure_cases = check_result.check_passed
                message = format_generic_error_message(
                    schema, check, check_index
                )
            else:
                import pandas as pd

                from pandera.api.pandas.types import is_table

                check_failure_cases = check_result.failure_cases.to_pandas()
                if is_table(check_failure_cases):
                    check_failure_cases = (
                        pd.Series(check_failure_cases.to_dict("records"))
                        .rename("failure_case")
                        .to_frame()
                    )

                failure_cases = reshape_failure_cases(
                    check_failure_cases, check.ignore_na
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
        schema_errors: list[SchemaError],
    ) -> FailureCaseMetadata:
        """Create failure cases metadata required for SchemaErrors exception."""
        failure_cases = consolidate_failure_cases(schema_errors)

        error_handler = ErrorHandler()
        error_handler.collect_errors(schema_errors)
        error_dicts = {}

        def defaultdict_to_dict(d):
            if isinstance(d, defaultdict):
                d = {k: defaultdict_to_dict(v) for k, v in d.items()}
            return d

        if error_handler.collected_errors:
            error_dicts = error_handler.summarize(schema_name=schema_name)
            error_dicts = defaultdict_to_dict(error_dicts)

        error_counts = defaultdict(int)  # type: ignore
        for error in error_handler.collected_errors:
            error_counts[error["reason_code"].name] += 1

        return FailureCaseMetadata(
            failure_cases=failure_cases,
            message=error_dicts,
            error_counts=error_counts,
        )

    def drop_invalid_rows(
        self, check_obj: ibis.Table, error_handler: ErrorHandler
    ) -> ibis.Table:
        """Remove invalid elements in a check obj according to failures caught by the error handler."""
        import ibis.expr.types as ir

        if (
            positional_join := check_obj.get_backend().name
            in POSITIONAL_JOIN_BACKENDS
        ):
            out = check_obj
            join = lambda left, right: left.join(right, how="positional")
        else:
            # For backends that do not support positional joins:
            # https://github.com/ibis-project/ibis/issues/9486
            index_col = "__idx__"
            out = check_obj.mutate(**{index_col: ibis.row_number().over()})

            def join(left, right):
                return left.join(
                    right.mutate(**{index_col: ibis.row_number().over()}),
                    index_col,
                )

        for i, error in enumerate(error_handler.schema_errors):
            check_output = error.check_output
            if isinstance(check_output, ir.BooleanColumn):
                check_output = (
                    (~check_output).name(CHECK_OUTPUT_KEY).as_table()
                )

            out = join(
                out,
                check_output.rename(
                    {f"{i}{CHECK_OUTPUT_SUFFIX}": CHECK_OUTPUT_KEY}
                ),
            )

        if not positional_join:
            out = out.drop(index_col)

        acc = ibis.literal(True)
        for col in out.columns:
            if col.endswith(CHECK_OUTPUT_SUFFIX):
                acc = acc & out[col]

        return out.filter(acc).drop(s.endswith(CHECK_OUTPUT_SUFFIX))

    def _extract_check_results(
        self,
        original_table: ibis.Table,
        wide_executed: pd.DataFrame,
        checks_applied: list[tuple[int, Check]],
        schema,
    ) -> list[CoreCheckResult]:
        """Extract individual CoreCheckResult from executed wide table.

        :param original_table: The original Ibis table being validated.
        :param wide_executed: The executed wide table as a pandas DataFrame.
        :param checks_applied: List of (check_index, check) tuples for applied checks.
        :param schema: The schema being validated against.
        :returns: List of CoreCheckResult objects.
        """
        results = []

        for check_index, check in checks_applied:
            # Find check columns by prefix pattern: {check_index}_{col}{CHECK_OUTPUT_SUFFIX}
            prefix = f"{check_index}_"
            check_cols = [
                c
                for c in wide_executed.columns
                if c.startswith(prefix) and c.endswith(CHECK_OUTPUT_SUFFIX)
            ]

            if not check_cols:
                results.append(
                    CoreCheckResult(
                        passed=True,
                        check=check,
                        check_index=check_index,
                        reason_code=SchemaErrorReason.DATAFRAME_CHECK,
                    )
                )
                continue

            # Compute passed: all check columns must be True for all rows
            passed = wide_executed[check_cols].all(axis=None)

            if passed:
                results.append(
                    CoreCheckResult(
                        passed=True,
                        check=check,
                        check_index=check_index,
                        reason_code=SchemaErrorReason.DATAFRAME_CHECK,
                    )
                )
            else:
                # Extract failure cases: rows where any check column is False
                failure_mask = ~wide_executed[check_cols].all(axis=1)
                original_cols = [
                    c
                    for c in wide_executed.columns
                    if not c.endswith(CHECK_OUTPUT_SUFFIX)
                ]
                failure_cases = wide_executed.loc[failure_mask, original_cols]

                # Apply n_failure_cases limit
                if check.n_failure_cases is not None:
                    failure_cases = failure_cases.head(check.n_failure_cases)

                failure_cases = reshape_failure_cases(
                    failure_cases, check.ignore_na
                )
                message = format_vectorized_error_message(
                    schema, check, check_index, failure_cases
                )

                if check.raise_warning:
                    warnings.warn(message, SchemaWarning)
                    results.append(
                        CoreCheckResult(
                            passed=True,
                            check=check,
                            check_index=check_index,
                            reason_code=SchemaErrorReason.DATAFRAME_CHECK,
                        )
                    )
                else:
                    results.append(
                        CoreCheckResult(
                            passed=False,
                            check=check,
                            check_index=check_index,
                            reason_code=SchemaErrorReason.DATAFRAME_CHECK,
                            message=message,
                            failure_cases=failure_cases,
                        )
                    )

        return results
