"""Column backend for narwhals — per-column validation layer."""

import warnings
from collections.abc import Iterable
from typing import cast

import narwhals.stable.v1 as nw

from pandera.api.base.error_handler import ErrorHandler, get_error_category
from pandera.api.narwhals.utils import _to_native
from pandera.backends.base import CoreCheckResult
from pandera.backends.narwhals.base import NarwhalsSchemaBackend, _materialize
from pandera.config import ValidationScope
from pandera.constants import CHECK_OUTPUT_KEY
from pandera.errors import SchemaError, SchemaErrorReason, SchemaWarning
from pandera.validation_depth import validate_scope


class ColumnBackend(NarwhalsSchemaBackend):
    """Per-column validation backend for narwhals-backed DataFrames.

    Implements check_nullable, check_unique, check_dtype, run_checks, and
    run_checks_and_handle_errors — mirroring pandera/backends/polars/components.py
    but using narwhals APIs throughout.
    """

    @validate_scope(scope=ValidationScope.DATA)
    def check_nullable(self, check_obj, schema) -> list[CoreCheckResult]:
        """Check that no null (or NaN for float columns) values exist.

        :param check_obj: narwhals LazyFrame containing the column.
        :param schema: Schema object with .nullable and .selector attributes.
        :returns: List of CoreCheckResult — one entry per column selector.
        """
        if schema.nullable:
            return [
                CoreCheckResult(
                    passed=True,
                    check="not_nullable",
                    reason_code=SchemaErrorReason.SERIES_CONTAINS_NULLS,
                )
            ]

        col = schema.selector
        null_expr = nw.col(col).is_null()
        if self.is_float_dtype(check_obj, col):
            null_expr = null_expr | nw.col(col).is_nan()

        is_null_lf = check_obj.select(null_expr)
        # Materialize both — narwhals does NOT support lazy horizontal concat
        data_df = _materialize(check_obj)
        is_null_df = _materialize(is_null_lf)

        results = []
        for column in is_null_df.collect_schema().names():
            if not is_null_df[column].any():
                continue
            combined = nw.concat(
                [data_df, is_null_df.rename({column: CHECK_OUTPUT_KEY})],
                how="horizontal",
            )
            failure_cases = _to_native(
                combined.filter(nw.col(CHECK_OUTPUT_KEY)).select(column)
            )
            results.append(
                CoreCheckResult(
                    passed=False,
                    check_output=is_null_df.rename({column: CHECK_OUTPUT_KEY}),
                    check="not_nullable",
                    reason_code=SchemaErrorReason.SERIES_CONTAINS_NULLS,
                    message=f"non-nullable column '{schema.selector}' contains null values",
                    failure_cases=failure_cases,
                )
            )
        return results

    @validate_scope(scope=ValidationScope.DATA)
    def check_unique(self, check_obj, schema) -> list[CoreCheckResult]:
        """Check that column values are unique (no duplicates).

        :param check_obj: narwhals LazyFrame containing the column.
        :param schema: Schema object with .unique and .selector attributes.
        :returns: List of CoreCheckResult — one entry per column selector.
        """
        check_name = "field_uniqueness"
        if not schema.unique:
            return [
                CoreCheckResult(
                    passed=True,
                    check=check_name,
                    reason_code=SchemaErrorReason.SERIES_CONTAINS_DUPLICATES,
                )
            ]

        # COLUMN-02: force collection before is_duplicated() — is_duplicated() is a
        # window function that requires full data visibility. _materialize() handles both
        # Polars LazyFrame (.collect()) and Ibis nw.DataFrame (.execute()).
        collected = _materialize(check_obj.select(schema.selector))
        duplicates = collected.select(nw.col(schema.selector).is_duplicated())

        results = []
        for column in duplicates.collect_schema().names():
            if not duplicates[column].any():
                results.append(
                    CoreCheckResult(
                        passed=True,
                        check=check_name,
                        reason_code=SchemaErrorReason.SERIES_CONTAINS_DUPLICATES,
                    )
                )
                continue
            combined = nw.concat(
                [collected, duplicates.rename({column: "_duplicated"})],
                how="horizontal",
            )
            failure_cases = _to_native(
                combined.filter(nw.col("_duplicated")).select(column)
            )
            results.append(
                CoreCheckResult(
                    passed=False,
                    check=check_name,
                    check_output=duplicates.select(
                        (~nw.col(column)).alias(CHECK_OUTPUT_KEY)
                    ),
                    reason_code=SchemaErrorReason.SERIES_CONTAINS_DUPLICATES,
                    message=f"column '{schema.selector}' not unique:\n{failure_cases}",
                    failure_cases=failure_cases,
                )
            )
        return results

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_dtype(self, check_obj, schema) -> list[CoreCheckResult]:
        """Check that column dtype matches schema.dtype.

        :param check_obj: narwhals LazyFrame containing the column.
        :param schema: Schema object with .dtype and .selector attributes.
        :returns: List of CoreCheckResult — one entry per column selector.
        """
        if schema.dtype is None:
            return [
                CoreCheckResult(
                    passed=True,
                    check=f"dtype('{schema.dtype}')",
                    reason_code=SchemaErrorReason.WRONG_DATATYPE,
                )
            ]

        # Import inside method to avoid circular import chains
        from pandera.engines import narwhals_engine

        results = []
        schema_obj = check_obj.select(schema.selector).collect_schema()
        for column, nw_dtype in zip(schema_obj.names(), schema_obj.dtypes()):
            try:
                col_pandera_dtype = narwhals_engine.Engine.dtype(nw_dtype)
            except TypeError:
                col_pandera_dtype = nw_dtype  # fallback: .check() will return False
            passed = schema.dtype.check(col_pandera_dtype)
            results.append(
                CoreCheckResult(
                    passed=bool(passed),
                    check=f"dtype('{schema.dtype}')",
                    reason_code=SchemaErrorReason.WRONG_DATATYPE,
                    message=(
                        f"expected column '{column}' to have type {schema.dtype}, "
                        f"got {nw_dtype}"
                        if not passed
                        else None
                    ),
                    failure_cases=str(nw_dtype) if not passed else None,
                )
            )
        return results

    @validate_scope(scope=ValidationScope.DATA)
    def run_checks(self, check_obj, schema) -> list[CoreCheckResult]:
        """Execute all Check objects attached to schema and return results.

        :param check_obj: narwhals LazyFrame containing the column.
        :param schema: Schema object with .checks list and .selector attribute.
        :returns: List of CoreCheckResult, one per check.
        """
        check_results: list[CoreCheckResult] = []
        for check_index, check in enumerate(schema.checks):
            try:
                check_results.append(
                    self.run_check(
                        check_obj, schema, check, check_index, schema.selector
                    )
                )
            except Exception as err:
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

    def run_checks_and_handle_errors(
        self, error_handler: ErrorHandler, schema, check_obj, **subsample_kwargs
    ) -> ErrorHandler:
        """Orchestrate all column checks, collecting errors via ErrorHandler.

        :param error_handler: ErrorHandler instance to collect schema errors.
        :param schema: Schema object defining column constraints.
        :param check_obj: narwhals LazyFrame containing the column.
        :param subsample_kwargs: Keyword arguments forwarded to subsample().
        :returns: The error_handler with all errors collected.
        """
        check_obj_subsample = self.subsample(check_obj, **subsample_kwargs)
        core_checks = [
            self.check_nullable,
            self.check_unique,
            self.check_dtype,
            self.run_checks,
        ]
        args = (check_obj_subsample, schema)
        for core_check in core_checks:
            results = core_check(*args)
            if isinstance(results, CoreCheckResult):
                results = [results]
            results = cast(list[CoreCheckResult], results)
            for result in results:
                if result.passed:
                    continue
                if result.schema_error is not None:
                    error = result.schema_error
                else:
                    assert result.reason_code is not None
                    error = SchemaError(
                        schema=schema,
                        data=check_obj,
                        message=result.message,
                        failure_cases=result.failure_cases,
                        check=result.check,
                        check_index=result.check_index,
                        check_output=result.check_output,
                        reason_code=result.reason_code,
                    )
                error_handler.collect_error(
                    get_error_category(result.reason_code),
                    result.reason_code,
                    error,
                    original_exc=result.original_exc,
                )
        return error_handler
