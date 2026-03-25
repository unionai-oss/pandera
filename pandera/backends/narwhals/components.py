"""Column backend for Narwhals — per-column validation layer."""

import warnings
from collections.abc import Iterable
from typing import cast

import narwhals.stable.v1 as nw

from pandera.api.base.error_handler import get_error_category
from pandera.api.narwhals.error_handler import ErrorHandler
from pandera.api.narwhals.utils import _to_native
from pandera.backends.base import CoreCheckResult
from pandera.backends.narwhals.base import NarwhalsSchemaBackend, _materialize
from pandera.config import ValidationScope
from pandera.constants import CHECK_OUTPUT_KEY
from pandera.errors import (
    SchemaDefinitionError,
    SchemaError,
    SchemaErrorReason,
    SchemaErrors,
    SchemaWarning,
)
from pandera.validation_depth import validate_scope


class ColumnBackend(NarwhalsSchemaBackend):
    """Per-column validation backend for Narwhals-backed DataFrames.

    Implements validate, check_nullable, check_unique, check_dtype, run_checks,
    and run_checks_and_handle_errors — mirroring pandera/backends/polars/components.py
    but using Narwhals APIs throughout.
    """

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
        """Validate a Narwhals-backed frame against a column schema.

        :param check_obj: native pl.LazyFrame (or other Narwhals-supported frame)
        :param schema: Column schema with checks and constraints
        :returns: validated frame (same type as input)
        """
        if inplace:
            warnings.warn("setting inplace=True will have no effect.")

        if schema.name is None:
            raise SchemaDefinitionError(
                "Column schema must have a name specified."
            )

        # Wrap to Narwhals for uniform handling
        check_lf = nw.from_native(check_obj, eager_or_interchange_only=False)
        if isinstance(check_lf, nw.DataFrame):
            check_lf = check_lf.lazy()

        error_handler = ErrorHandler(lazy)

        if getattr(schema, "drop_invalid_rows", False) and not lazy:
            raise SchemaDefinitionError(
                "When drop_invalid_rows is True, lazy must be set to True."
            )

        error_handler = self.run_checks_and_handle_errors(
            error_handler,
            schema,
            check_lf,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
        )

        if lazy and error_handler.collected_errors:
            if getattr(schema, "drop_invalid_rows", False):
                check_lf = self.drop_invalid_rows(check_lf, error_handler)
            else:
                # Use native frame for data in SchemaErrors — SchemaErrors.__init__
                # calls schema.get_backend(data), which requires a registered native type.
                raise SchemaErrors(
                    schema=schema,
                    schema_errors=error_handler.schema_errors,
                    data=_to_native(check_lf),
                )
        elif not lazy and error_handler.collected_errors:
            # Non-lazy mode: raise the first collected error
            raise error_handler.schema_errors[0]

        return check_lf

    def get_regex_columns(self, schema, check_obj) -> Iterable:
        """Get column names matching a regex pattern."""
        frame_cols = check_obj.collect_schema().names()
        import re
        return [c for c in frame_cols if re.search(schema.selector, c)]

    @validate_scope(scope=ValidationScope.DATA)
    def check_nullable(self, check_obj, schema) -> list[CoreCheckResult]:
        """Check that no null (or NaN for float columns) values exist.

        :param check_obj: Narwhals LazyFrame containing the column.
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

        # Add null indicator inline — single LazyFrame op, no separate frame materialization.
        combined_lf = check_obj.with_columns(null_expr.alias(CHECK_OUTPUT_KEY))

        # Materialize ONE ROW to evaluate the scalar bool — not the full frame.
        # _materialize handles both nw.LazyFrame (collect) and SQL-lazy DataFrame (execute).
        has_nulls_df = _materialize(combined_lf.select(nw.col(CHECK_OUTPUT_KEY).any()))
        has_nulls = bool(has_nulls_df[CHECK_OUTPUT_KEY][0])

        if not has_nulls:
            return [
                CoreCheckResult(
                    passed=True,
                    check="not_nullable",
                    reason_code=SchemaErrorReason.SERIES_CONTAINS_NULLS,
                )
            ]

        # failure_cases and check_output stay lazy — narwhals wrappers, not native.
        failure_cases = combined_lf.filter(nw.col(CHECK_OUTPUT_KEY)).select(col)
        return [
            CoreCheckResult(
                passed=False,
                check_output=combined_lf,
                check="not_nullable",
                reason_code=SchemaErrorReason.SERIES_CONTAINS_NULLS,
                message=f"non-nullable column '{schema.selector}' contains null values",
                failure_cases=failure_cases,
            )
        ]

    @validate_scope(scope=ValidationScope.DATA)
    def check_unique(self, check_obj, schema) -> list[CoreCheckResult]:
        """Check that column values are unique (no duplicates).

        :param check_obj: Narwhals LazyFrame containing the column.
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

        # SQL-lazy safe: group_by().agg(nw.len()) works on ibis without materializing full frame.
        # Supersedes COLUMN-02 collect()+is_duplicated() approach for SQL-lazy backends.
        col = schema.selector
        grouped = (
            check_obj
            .select(nw.col(col))
            .group_by(nw.col(col))
            .agg(nw.len().alias("_count"))
        )
        dup_values = grouped.filter(nw.col("_count") > 1).select(col)
        native_dups = nw.to_native(_materialize(dup_values))

        results = []
        if len(native_dups) == 0:
            results.append(
                CoreCheckResult(
                    passed=True,
                    check=check_name,
                    reason_code=SchemaErrorReason.SERIES_CONTAINS_DUPLICATES,
                )
            )
        else:
            results.append(
                CoreCheckResult(
                    passed=False,
                    check=check_name,
                    check_output=None,  # group_by approach does not produce per-row booleans
                    reason_code=SchemaErrorReason.SERIES_CONTAINS_DUPLICATES,
                    message=f"column '{col}' not unique:\n{native_dups}",
                    failure_cases=native_dups,
                )
            )
        return results

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_dtype(self, check_obj, schema) -> list[CoreCheckResult]:
        """Check that column dtype matches schema.dtype.

        :param check_obj: Narwhals LazyFrame containing the column.
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
                col_pandera_dtype = nw_dtype

            # Use narwhals_engine for comparison — Engine.dtype() now accepts
            # cross-engine dtypes (polars_engine, ibis_engine) by re-interpreting
            # through the shared abstract pandera base class. Parametric types
            # (List, Struct) fall back to a direct check, which will report
            # WRONG_DATATYPE for cross-engine schemas. TODO: root fix is in schema
            # construction — pandera.polars/pandera.ibis should produce narwhals
            # engine dtypes when the Narwhals backend is active.
            try:
                schema_nw_dtype = narwhals_engine.Engine.dtype(schema.dtype)
                passed = schema_nw_dtype.check(col_pandera_dtype)
            except TypeError:
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

        :param check_obj: Narwhals LazyFrame containing the column.
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
        :param check_obj: Narwhals LazyFrame containing the column.
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
                    # Convert narwhals failure_cases to native for SchemaError public API.
                    # CoreCheckResult carries narwhals wrappers; SchemaError.failure_cases
                    # is the public API and must be native (pl.DataFrame, ibis.Table, etc.)
                    # so callers can use the result without narwhals knowledge.
                    # For SQL-lazy backends (ibis): nw.to_native(LazyFrame) returns ibis.Table
                    #   directly (no execution) — hasattr(native, 'execute') detects this case.
                    # For polars LazyFrame: must collect() first, then to_native → pl.DataFrame.
                    # For nw.DataFrame: to_native directly (polars → pl.DataFrame, ibis → ibis.Table).
                    fc = result.failure_cases
                    if isinstance(fc, nw.LazyFrame):
                        native_fc = nw.to_native(fc)
                        if hasattr(native_fc, "execute"):
                            # SQL-lazy backend (ibis): native is already ibis.Table
                            fc = native_fc
                        else:
                            # Polars lazy: collect to eager then unwrap
                            fc = nw.to_native(_materialize(fc))
                    elif isinstance(fc, nw.DataFrame):
                        fc = nw.to_native(fc)
                    error = SchemaError(
                        schema=schema,
                        data=check_obj,
                        message=result.message,
                        failure_cases=fc,
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
