"""Pandera array backends."""

import traceback
from typing import cast, Iterable, NamedTuple, Optional

import pandas as pd
from multimethod import DispatchError

from pandera.backends.pyspark.base import PysparkSchemaBackend
from pandera.backends.pandas.error_formatters import (
    reshape_failure_cases,
    scalar_failure_case,
)

from pandera.engines.pyspark_engine import Engine
from pandera.error_handlers import SchemaErrorHandler
from pandera.errors import (
    ParserError,
    SchemaError,
    SchemaErrors,
    SchemaErrorReason,
)
from pyspark.sql import DataFrame


class CoreCheckResult(NamedTuple):
    """Namedtuple for holding results of core checks."""

    check: str
    reason_code: SchemaErrorReason
    passed: bool
    message: Optional[str] = None
    failure_cases: Optional[Iterable] = None


class ArraySchemaBackend(PysparkSchemaBackend):
    """Backend for pyspark arrays."""

    def preprocess(self, check_obj, inplace: bool = False):
        return check_obj

    def validate(
        self,
        check_obj,
        schema,
        *,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ):
        # pylint: disable=too-many-locals
        error_handler = SchemaErrorHandler(lazy)
        check_obj = self.preprocess(check_obj, inplace)

        if schema.coerce:
            try:
                check_obj = self.coerce_dtype(
                    check_obj, schema=schema, error_handler=error_handler
                )
            except SchemaError as exc:
                error_handler.collect_error(exc.reason_code, exc)

        check_obj_subsample = self.subsample(
            check_obj,
            sample,
            random_state,
        )

        # run the core checks
        for core_check in (
            self.check_name,
            # self.check_nullable,
            # self.check_unique,
            self.check_dtype,
        ):
            check_result = core_check(check_obj_subsample, schema)
            print(check_result)
            breakpoint()
            if not check_result.passed:
                error_handler.collect_error(
                    check_result.reason_code,
                    SchemaError(
                        schema=schema,
                        data=check_obj,
                        message=check_result.message,
                        failure_cases=check_result.failure_cases,
                        check=check_result.check,
                        reason_code=check_result.reason_code,
                    ),
                )
        breakpoint()
        check_results = self.run_checks(
            check_obj_subsample, schema, error_handler, lazy
        )
        breakpoint()
        assert all(check_results)

        if lazy and error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.collected_errors,
                data=check_obj,
            )
        return check_obj

    def coerce_dtype(
        self,
        check_obj,
        *,
        schema=None,
        # pylint: disable=unused-argument
        error_handler: SchemaErrorHandler = None,
    ):
        """Coerce type of a pd.Series by type specified in dtype.

        :param pd.Series series: One-dimensional ndarray with axis labels
            (including time series).
        :returns: ``Series`` with coerced data type
        """
        assert schema is not None, "The `schema` argument must be provided."
        if schema.dtype is None or not schema.coerce:
            return check_obj

        try:
            return schema.dtype.try_coerce(check_obj)
        except ParserError as exc:
            raise SchemaError(
                schema=schema,
                data=check_obj,
                message=(
                    f"Error while coercing '{schema.name}' to type "
                    f"{schema.dtype}: {exc}:\n{exc.failure_cases}"
                ),
                failure_cases=exc.failure_cases,
                check=f"coerce_dtype('{schema.dtype}')",
            ) from exc

    def check_name(self, check_obj: DataFrame, schema):
        column_found = not (schema.name is None or schema.name not in check_obj.columns)
        breakpoint()
        return CoreCheckResult(
            check=f"field_name('{schema.name}')",
            reason_code=SchemaErrorReason.WRONG_FIELD_NAME
            if not column_found
            else SchemaErrorReason.NO_ERROR,
            passed=column_found,
            message=(
                f"Expected {type(check_obj)} to have column named: '{schema.name}', "
                f"but found columns '{check_obj.columns}'"
                if not column_found
                else f"column check_name validation passed."
            ),
            failure_cases=scalar_failure_case(schema.name)
            if not column_found
            else None,
        )

    def check_nullable(self, check_obj: DataFrame, schema):
        isna = check_obj.isna()
        passed = schema.nullable or not isna.any()
        return CoreCheckResult(
            check="not_nullable",
            reason_code=SchemaErrorReason.SERIES_CONTAINS_NULLS,
            passed=cast(bool, passed),
            message=(
                f"non-nullable series '{check_obj.name}' contains "
                f"null values:\n{check_obj[isna]}"
            ),
            failure_cases=reshape_failure_cases(check_obj[isna], ignore_na=False),
        )

    def check_unique(self, check_obj: DataFrame, schema):
        passed = True
        failure_cases = None
        message = None

        if schema.unique:
            # Todo  Add Failure Cases

            if check_obj.count() != check_obj.drop_duplicates().count:
                passed = False
                failure_cases = None  # reshape_failure_cases(failed)
                message = (
                    f"Column '{schema.name}' contains duplicate "
                    # f"values:\n{failed}"
                )

        return CoreCheckResult(
            check="field_uniqueness",
            reason_code=SchemaErrorReason.SERIES_CONTAINS_DUPLICATES,
            passed=passed,
            message=message,
            failure_cases=failure_cases,
        )

    def check_dtype(self, check_obj: DataFrame, schema):
        passed = True
        failure_cases = None
        msg = None

        if schema.dtype is not None:
            breakpoint()
            dtype_check_results = schema.dtype.check(
                Engine.dtype(check_obj.schema[schema.name].dataType),
            )

            if isinstance(dtype_check_results, bool):
                passed = dtype_check_results
                failure_cases = scalar_failure_case(
                    str(Engine.dtype(check_obj.schema[schema.name].dataType))
                )
                msg = (
                    f"expected column '{schema.name}' to have type "
                    f"{schema.dtype}, got {Engine.dtype(check_obj.schema[schema.name].dataType)}"
                    if not passed
                    else f"column type matched with expected '{schema.dtype}'"
                )
            else:
                passed = dtype_check_results.all()
                failure_cases = reshape_failure_cases(
                    check_obj[~dtype_check_results.astype(bool)],
                    ignore_na=False,
                )
                msg = (
                    f"expected series '{check_obj.name}' to have type "
                    f"{schema.dtype}:\nfailure cases:\n{failure_cases}"
                )
            reason_code = (
                SchemaErrorReason.WRONG_DATATYPE
                if dtype_check_results
                else SchemaErrorReason.NO_ERROR
            )

        return CoreCheckResult(
            check=f"dtype('{schema.dtype}')",
            reason_code=reason_code,
            passed=passed,
            message=msg,
            failure_cases=failure_cases,
        )

    # pylint: disable=unused-argument
    def run_checks(self, check_obj, schema, error_handler, lazy):
        check_results = []
        breakpoint()
        for check_index, check in enumerate(schema.checks):
            check_args = [schema.name]
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
            except SchemaError as err:
                breakpoint()
                error_handler.collect_error(
                    SchemaErrorReason.DATAFRAME_CHECK,
                    err,
                )
            except Exception as err:  # pylint: disable=broad-except
                # catch other exceptions that may occur when executing the Check
                if isinstance(err, DispatchError):
                    # if the error was raised by a check registered via
                    # multimethod, get the underlying __cause__
                    err = err.__cause__
                err_msg = f'"{err.args[0]}"' if len(err.args) > 0 else ""
                err_str = f"{err.__class__.__name__}({ err_msg})"
                error_handler.collect_error(
                    SchemaErrorReason.CHECK_ERROR,
                    SchemaError(
                        schema=schema,
                        data=check_obj,
                        message=(
                            f"Error while executing check function: {err_str}\n"
                            + traceback.format_exc()
                        ),
                        failure_cases=scalar_failure_case(err_str),
                        check=check,
                        check_index=check_index,
                    ),
                    original_exc=err,
                )
        return check_results