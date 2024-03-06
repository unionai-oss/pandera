"""Polars Parsing, Validation, and Error Reporting Backends."""

import warnings
from collections import defaultdict
from typing import List, Dict, Optional

import polars as pl
from pandera.api.polars.types import CheckResult
from pandera.backends.base import BaseSchemaBackend, CoreCheckResult
from pandera.backends.pandas.error_formatters import (
    format_generic_error_message,
    format_vectorized_error_message,
)
from pandera.error_handlers import SchemaErrorHandler
from pandera.errors import (
    SchemaError,
    FailureCaseMetadata,
    SchemaErrorReason,
    SchemaWarning,
)


def is_float_dtype(check_obj: pl.LazyFrame, name):
    """Check if a column/selector is a float."""
    return all(
        dtype in pl.FLOAT_DTYPES
        for dtype in check_obj.select(pl.col(name)).schema.values()
    )


class PolarsSchemaBackend(BaseSchemaBackend):
    """Backend for polars LazyFrame schema."""

    def subsample(
        self,
        check_obj: pl.LazyFrame,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        obj_subsample = []
        if head is not None:
            obj_subsample.append(check_obj.head(head))
        if tail is not None:
            obj_subsample.append(check_obj.tail(tail))
        if sample is not None:
            obj_subsample.append(
                check_obj.sample(sample, random_state=random_state)
            )
        return (
            check_obj
            if not obj_subsample
            else pl.concat(obj_subsample).unique()
        )

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

            if isinstance(err.failure_cases, pl.DataFrame):
                failure_cases_df = err.failure_cases

                if len(err.failure_cases) > 1:
                    # for boolean dataframe check results, reduce failure cases
                    # to a struct column
                    failure_cases_df = err.failure_cases.with_columns(
                        failure_case=pl.Series(
                            err.failure_cases.rows(named=True)
                        )
                    ).select(pl.col.failure_case)

                failure_cases_df = failure_cases_df.with_columns(
                    schema_context=pl.lit(err.schema.__class__.__name__),
                    column=pl.lit(err.schema.name),
                    check=pl.lit(check_identifier),
                    check_number=pl.lit(err.check_index),
                    index=pl.lit(None),
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
                failure_cases_df = pl.DataFrame(scalar_failure_cases)

            failure_case_collection.append(failure_cases_df)

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

    def drop_invalid_rows(
        self,
        check_obj: pl.LazyFrame,
        error_handler: SchemaErrorHandler,
    ) -> pl.LazyFrame:
        """Remove invalid elements in a check obj according to failures in caught by the error handler."""
        errors = error_handler.collected_errors
        check_outputs = pl.DataFrame(
            {str(i): err.check_output for i, err in enumerate(errors)}
        )
        valid_rows = check_outputs.select(
            valid_rows=pl.fold(
                acc=pl.lit(True),
                function=lambda acc, x: acc & x,
                exprs=pl.col(pl.Boolean),
            )
        )["valid_rows"]
        return check_obj.filter(valid_rows)


FAILURE_CASE_TEMPLATE = """
Schema {schema_name}: A total of {error_count} errors were found.

{failure_cases}
""".strip()
