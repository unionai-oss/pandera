"""Pandas parsing, validation, and error-reporting backends."""

import warnings
from collections import defaultdict
from typing import Optional, TypeVar, Union

import pandas as pd

from pandera.api.base.checks import CheckResult
from pandera.api.base.error_handler import ErrorHandler
from pandera.api.parsers import Parser
from pandera.backends.base import (
    BaseSchemaBackend,
    CoreCheckResult,
    CoreParserResult,
)
from pandera.backends.pandas.error_formatters import (
    consolidate_failure_cases,
    format_generic_error_message,
    format_vectorized_error_message,
    reshape_failure_cases,
)
from pandera.errors import (
    FailureCaseMetadata,
    SchemaError,
    SchemaErrorReason,
    SchemaWarning,
)

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

    def run_parser(
        self,
        check_obj,
        parser: Parser,
        parser_index: int,
        *args,
    ) -> CoreParserResult:
        """Handle parser results.

        :param check_obj: data object to be validated.
        :param schema: pandera schema object
        :param parser: Parser object used to validate pandas object.
        :param parser_index: index of parser in the schema component parser list.
        :param args: arguments to pass into parser object.
        :returns:  ParserResult
        """
        parser_result = parser(check_obj, *args)

        return CoreParserResult(
            passed=True,
            parser=parser,
            parser_index=parser_index,
            parser_output=parser_result.parser_output,
            reason_code=SchemaErrorReason.DATAFRAME_PARSER,
            failure_cases=None,
            message=None,
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
        :param schema: pandera schema object.
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
                failure_cases = check_result.check_passed
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

    def drop_invalid_rows(self, check_obj, error_handler: ErrorHandler):
        """Remove invalid elements in a check obj according to failures caught by the error handler."""
        for err in error_handler.schema_errors:
            if isinstance(err.failure_cases, str):
                # if the failure cases are a string, it means the error is
                # a schema-level error.
                continue
            if isinstance(check_obj.index, pd.MultiIndex):
                index_tuples = err.failure_cases["index"].apply(eval)
                index_values = pd.MultiIndex.from_tuples(index_tuples)
            else:
                index_values = err.failure_cases["index"]

            mask = ~check_obj.index.isin(index_values)
            check_obj = check_obj.loc[mask]

        return check_obj
