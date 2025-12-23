"""PySpark parsing, validation, and error-reporting backends."""

import warnings
from collections.abc import Iterable
from typing import (
    Any,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
)

from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from pandera.api.checks import CheckResult
from pandera.api.pyspark.types import PySparkDataFrameTypes
from pandera.backends.base import BaseSchemaBackend, CoreCheckResult
from pandera.backends.pyspark.error_formatters import (
    format_generic_error_message,
    scalar_failure_case,
)
from pandera.errors import (
    FailureCaseMetadata,
    SchemaError,
    SchemaErrorReason,
    SchemaWarning,
)


class ColumnInfo(NamedTuple):
    """Column metadata used during validation."""

    sorted_column_names: Iterable
    expanded_column_names: frozenset
    destuttered_column_names: list
    absent_column_names: list
    lazy_exclude_column_names: list


FieldCheckObj = Union[col, PySparkDataFrameTypes]

T = TypeVar(
    "T",
    col,  # type: ignore
    DataFrame,
    FieldCheckObj,  # type: ignore
    covariant=True,
)


class PysparkSchemaBackend(BaseSchemaBackend):
    """Base backend for PySpark schemas."""

    def subsample(
        self,
        check_obj: PySparkDataFrameTypes,
        head: int | None = None,
        tail: int | None = None,
        sample: float | None = None,
        random_state: int | None = None,
    ):
        if sample is not None:
            return check_obj.sample(
                withReplacement=False,
                fraction=sample,
                seed=random_state,  # type: ignore
            )
        return check_obj

    def run_check(
        self,
        check_obj,
        schema,
        check,
        check_index: int,
        *args,
    ) -> CoreCheckResult:
        """Handle check results, raising SchemaError on check failure.

        :param check_obj: pyspark dataframe object
        :param schema: schema information of the column in the dataframe that needs to be validated
        :param check: Check object used to validate pyspark object.
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
            # encode scalar False values explicitly
            failure_cases = scalar_failure_case(passed)
            message = format_generic_error_message(schema, check)

            # raise a warning without exiting if the check is specified to do so
            if check.raise_warning:  # pragma: no cover
                warnings.warn(
                    message=message,
                    category=SchemaWarning,
                )
                return CoreCheckResult(
                    passed=True,
                    check=check,
                    reason_code=SchemaErrorReason.CHECK_ERROR,
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
        schema_errors: list[dict[str, Any]],
    ) -> FailureCaseMetadata:
        """Create failure cases metadata required for SchemaErrors exception."""

        return FailureCaseMetadata(
            failure_cases=None,
            message=schema_errors,
            error_counts={},
        )
