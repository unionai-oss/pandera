"""Pandera array backends."""

import traceback
from typing import cast

from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from pandera.api.base.error_handler import ErrorCategory, ErrorHandler
from pandera.backends.base import CoreCheckResult
from pandera.backends.pyspark.base import PysparkSchemaBackend
from pandera.backends.pyspark.error_formatters import scalar_failure_case
from pandera.engines.pyspark_engine import Engine
from pandera.errors import (
    ParserError,
    SchemaError,
    SchemaErrorReason,
    SchemaErrors,
)
from pandera.validation_depth import ValidationScope, validate_scope


class ColumnSchemaBackend(PysparkSchemaBackend):
    """Backend for pyspark arrays."""

    def preprocess(self, check_obj, inplace: bool = False):
        return check_obj

    def _core_checks(
        self, check_obj, schema, error_handler: ErrorHandler
    ) -> ErrorHandler:
        """This function runs the core checks"""
        # run the core checks
        for check in (
            self.check_name,
            self.check_dtype,
            self.check_nullable,
            self.run_checks,
        ):
            results = check(check_obj, schema)
            if isinstance(results, CoreCheckResult):
                results = [results]

            for result in results:
                if result.passed:
                    continue

                if result.schema_error is not None:
                    error = result.schema_error
                else:
                    error = SchemaError(
                        schema=schema,
                        data=check_obj,
                        message=result.message,
                        failure_cases=result.failure_cases,
                        check=result.check,
                        reason_code=result.reason_code,
                    )
                error_handler.collect_error(
                    ErrorCategory.SCHEMA,
                    result.reason_code,
                    error,
                    result.original_exc,
                )

        return error_handler

    def validate(
        self,
        check_obj,
        schema,
        *,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ):
        check_obj = self.preprocess(check_obj, inplace)

        error_handler = ErrorHandler(lazy=lazy)

        if schema.coerce:
            try:
                check_obj = self.coerce_dtype(check_obj, schema=schema)
            except SchemaError as exc:
                error_handler.collect_error(
                    ErrorCategory.SCHEMA, exc.reason_code, exc
                )

        error_handler = self._core_checks(check_obj, schema, error_handler)

        if lazy and error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.schema_errors,
                data=check_obj,
            )
        return check_obj

    @validate_scope(scope=ValidationScope.SCHEMA)
    def coerce_dtype(
        self,
        check_obj: DataFrame,
        schema,
    ) -> DataFrame:
        """Coerce type of a pyspark.sql.function.col by type specified in dtype.

        :param check_obj: Pyspark DataFrame
        :returns: ``DataFrame`` with coerced data type
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

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_nullable(self, check_obj: DataFrame, schema) -> CoreCheckResult:
        passed = True

        # Use schema level information to optimize execution of the `nullable` check:
        # ignore this check if Pandera Field's `nullable` property is True
        # (check not necessary) or if df column's `nullable` property is False
        # (PySpark's nullable ensures the presence of values when creating the df)
        if (not schema.nullable) and (check_obj.schema[schema.name].nullable):
            passed = (
                check_obj.filter(col(schema.name).isNull()).limit(1).count()
                == 0
            )

        return CoreCheckResult(
            check="not_nullable",
            reason_code=SchemaErrorReason.SERIES_CONTAINS_NULLS,
            passed=cast(bool, passed),
            message=(f"non-nullable column '{schema.name}' contains null"),
            failure_cases=scalar_failure_case(schema.name),
        )

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_name(self, check_obj: DataFrame, schema) -> CoreCheckResult:
        column_found = not (
            schema.name is None or schema.name not in check_obj.columns
        )
        return CoreCheckResult(
            check=f"field_name('{schema.name}')",
            reason_code=(
                SchemaErrorReason.WRONG_FIELD_NAME
                if not column_found
                else SchemaErrorReason.NO_ERROR
            ),
            passed=column_found,
            message=(
                f"Expected {type(check_obj)} to have column named: '{schema.name}', "
                f"but found columns '{check_obj.columns}'"
                if not column_found
                else "column check_name validation passed."
            ),
            failure_cases=(
                scalar_failure_case(schema.name) if not column_found else None
            ),
        )

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_dtype(self, check_obj: DataFrame, schema) -> CoreCheckResult:
        passed = True
        failure_cases = None
        msg = None
        reason_code = None

        if schema.dtype is not None:
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
            reason_code = (
                SchemaErrorReason.WRONG_DATATYPE
                if not dtype_check_results
                else SchemaErrorReason.NO_ERROR
            )

        return CoreCheckResult(
            check=f"dtype('{schema.dtype}')",
            reason_code=reason_code,
            passed=passed,
            message=msg,
            failure_cases=failure_cases,
        )

    @validate_scope(scope=ValidationScope.DATA)
    def run_checks(self, check_obj, schema) -> list[CoreCheckResult]:
        check_results = []
        for check_index, check in enumerate(schema.checks):
            check_args = [schema.name]
            try:
                check_results.append(
                    self.run_check(
                        check_obj, schema, check, check_index, *check_args
                    )
                )
            # except SchemaError as err:
            #     check_results.append(
            #         CoreCheckResult(
            #             passed=False,
            #             check=check,
            #             check_index=check_index,
            #             reason_code=SchemaErrorReason.CHECK_ERROR,
            #             message=str(err),
            #             failure_cases=err.failure_cases,
            #             original_exc=err,
            #         )
            #     )
            except TypeError as err:
                raise err
            except Exception as err:
                # catch other exceptions that may occur when executing the Check
                err_msg = f'"{err.args[0]}"' if err.args else ""
                msg = f"{err.__class__.__name__}({err_msg})"
                check_results.append(
                    CoreCheckResult(
                        passed=False,
                        check=check,
                        check_index=check_index,
                        reason_code=SchemaErrorReason.CHECK_ERROR,
                        message=msg,
                        failure_cases=msg,
                        original_exc=err,
                    )
                )
        return check_results
