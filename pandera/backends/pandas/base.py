"""Pandas Parsing, Validation, and Error Reporting Backends."""

import warnings
from typing import (
    FrozenSet,
    Iterable,
    List,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
)

import pandas as pd

from pandera.api.base.checks import CheckResult
from pandera.backends.base import BaseSchemaBackend, CoreCheckResult
from pandera.backends.pandas.error_formatters import (
    format_generic_error_message,
    format_vectorized_error_message,
    consolidate_failure_cases,
    summarize_failure_cases,
    reshape_failure_cases,
    scalar_failure_case,
)
from pandera.errors import FailureCaseMetadata, SchemaError, SchemaErrorReason


class ColumnInfo(NamedTuple):
    """Column metadata used during validation."""

    sorted_column_names: Iterable
    expanded_column_names: FrozenSet
    destuttered_column_names: List
    absent_column_names: List
    lazy_exclude_column_names: List


FieldCheckObj = Union[pd.Series, pd.DataFrame]

T = TypeVar(
    "T",
    pd.Series,
    pd.DataFrame,
    FieldCheckObj,
    covariant=True,
)


class PandasSchemaBackend(BaseSchemaBackend):
    """Base backend for pandas schemas."""

    def subsample(
        self,
        check_obj,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        pandas_obj_subsample = []
        if head is not None:
            pandas_obj_subsample.append(check_obj.head(head))
        if tail is not None:
            pandas_obj_subsample.append(check_obj.tail(tail))
        if sample is not None:
            pandas_obj_subsample.append(
                check_obj.sample(sample, random_state=random_state)
            )
        return (
            check_obj
            if not pandas_obj_subsample
            else pd.concat(pandas_obj_subsample).pipe(
                lambda x: x[~x.index.duplicated()]
            )
        )

    def run_check(
        self,
        check_obj,
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

        passed = check_result.check_passed
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
                    check_result.failure_cases, check.ignore_na
                )
                message = format_vectorized_error_message(
                    schema, check, check_index, failure_cases
                )

            # raise a warning without exiting if the check is specified to do so
            # but make sure the check passes
            if check.raise_warning:
                warnings.warn(message, UserWarning)
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
