"""Validation backend for Narwhals DataFrameSchema."""

from __future__ import annotations

import copy
import re
import traceback
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

import narwhals.stable.v1 as nw

from pandera.api.base.error_handler import get_error_category
from pandera.api.narwhals.error_handler import ErrorHandler
from pandera.api.narwhals.utils import _is_lazy, _to_native

if TYPE_CHECKING:
    from pandera.api.polars.container import DataFrameSchema

from pandera.backends.base import ColumnInfo, CoreCheckResult
from pandera.backends.narwhals.base import NarwhalsSchemaBackend, _materialize
from pandera.config import ValidationDepth, ValidationScope, config_context, get_config_context
from pandera.errors import (
    ParserError,
    SchemaDefinitionError,
    SchemaError,
    SchemaErrorReason,
    SchemaErrors,
)
from pandera.utils import is_regex
from pandera.validation_depth import validate_scope, validation_type


def _to_lazy_nw(check_obj) -> nw.LazyFrame:
    """Wrap any supported native frame as a Narwhals LazyFrame."""
    wrapped = nw.from_native(check_obj, eager_or_interchange_only=False)
    if isinstance(wrapped, nw.DataFrame):
        return wrapped.lazy()
    return wrapped  # already nw.LazyFrame


def _to_frame_kind_nw(lf: nw.LazyFrame, return_type: type):
    """Unwrap Narwhals LazyFrame to the original native frame type."""
    native = nw.to_native(lf)
    # If the caller originally passed an eager frame, materialise by calling
    # .collect() on the native lazy result.  Use duck-typing on the *return_type
    # class* rather than importing polars: a lazy class exposes .collect on the
    # type itself, an eager class or ibis.Table does not.
    if not hasattr(return_type, "collect"):
        if hasattr(native, "collect"):
            # Acceptable: full-frame collect only at the final validation return boundary.
            # The caller originally passed an eager frame (e.g. pl.DataFrame) and expects
            # an eager result back.  This is a user-visible materialization at schema exit,
            # not an internal hot-path collect.
            return native.collect()
    return native


class DataFrameSchemaBackend(NarwhalsSchemaBackend):
    def validate(
        self,
        check_obj,
        schema: DataFrameSchema,
        *,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ):
        # Capture the input type so we can return the same type
        return_type = type(check_obj)

        # Convert to Narwhals LazyFrame — all parsers operate on LazyFrame
        check_lf = _to_lazy_nw(check_obj)

        if inplace:
            warnings.warn("setting inplace=True will have no effect.")

        error_handler = ErrorHandler(lazy)

        column_info = self.collect_column_info(check_lf, schema)

        if getattr(schema, "drop_invalid_rows", False) and not lazy:
            raise SchemaDefinitionError(
                "When drop_invalid_rows is True, lazy must be set to True."
            )

        # Phase 4: parsers list contains ONLY strict_filter_columns.
        # coerce_dtype, set_default, add_missing_columns are deferred to later phases.
        core_parsers: list[tuple[Callable[..., Any], tuple[Any, ...]]] = [
            (self.strict_filter_columns, (schema, column_info)),
        ]

        for parser, args in core_parsers:
            try:
                check_lf = parser(check_lf, *args)
            except SchemaError as exc:
                error_handler.collect_error(
                    get_error_category(exc.reason_code),
                    exc.reason_code,
                    exc,
                )
            except SchemaErrors as exc:
                error_handler.collect_errors(exc.schema_errors)

        # collect schema components
        components = self.collect_schema_components(
            check_lf, schema, column_info
        )

        # subsample on the Narwhals LazyFrame — no native round-trip before checks
        sample_obj = self.subsample(
            check_lf,
            head,
            tail,
            sample,
            random_state,
        )
        # subsample() returns nw.LazyFrame (unchanged) or nw.DataFrame (if head/tail used);
        # normalize to LazyFrame for uniform check execution
        if isinstance(sample_obj, nw.DataFrame):
            sample_lf = sample_obj.lazy()
        else:
            sample_lf = sample_obj  # already nw.LazyFrame

        core_checks = [
            (self.check_column_presence, (check_lf, schema, column_info)),
            (self.check_column_values_are_unique, (sample_lf, schema)),
            (
                self.run_schema_component_checks,
                (sample_lf, schema, components, lazy),
            ),
            (self.run_checks, (sample_lf, schema)),
        ]

        # When drop_invalid_rows=True, data checks must run even for lazy/SQL
        # backends that default to SCHEMA_ONLY validation depth. Force
        # SCHEMA_AND_DATA so @validate_scope(DATA) checks are not skipped.
        _check_ctx = (
            config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA)
            if getattr(schema, "drop_invalid_rows", False)
            else config_context()
        )

        with _check_ctx:
            for check, args in core_checks:
                results = check(*args)  # type: ignore[operator]
                if isinstance(results, CoreCheckResult):
                    results = [results]

                for result in results:
                    if result.passed:
                        continue

                    if result.schema_error is not None:
                        error = result.schema_error
                    else:
                        # Unwrap Narwhals failure_cases to native at the SchemaError boundary.
                        # CoreCheckResult carries Narwhals wrappers; SchemaError.failure_cases
                        # is the public API and must be native.
                        fc = result.failure_cases
                        if isinstance(fc, nw.LazyFrame):
                            if hasattr(nw.to_native(fc), "execute"):
                                # SQL-lazy backend (ibis): nw.to_native returns ibis.Table directly
                                fc = nw.to_native(fc)
                            else:
                                # Error path: collect failure_cases LazyFrame to eager.
                                # Bounded: fc contains only failing rows, not the full frame.
                                fc = nw.to_native(_materialize(fc))
                        elif isinstance(fc, nw.DataFrame):
                            fc = nw.to_native(fc)
                        error = SchemaError(
                            schema,
                            data=check_lf,
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

        if error_handler.collected_errors:
            if getattr(schema, "drop_invalid_rows", False):
                check_obj_parsed = _to_frame_kind_nw(check_lf, return_type)
                check_obj_parsed = self.drop_invalid_rows(
                    check_obj_parsed, error_handler
                )
                return check_obj_parsed
            else:
                raise SchemaErrors(
                    schema=schema,
                    schema_errors=error_handler.schema_errors,
                    data=_to_frame_kind_nw(check_lf, return_type),
                )

        return _to_frame_kind_nw(check_lf, return_type)

    @validate_scope(scope=ValidationScope.DATA)
    def run_checks(
        self,
        check_obj,
        schema,
    ) -> list[CoreCheckResult]:
        """Run a list of checks on the check object."""
        # dataframe-level checks
        check_results: list[CoreCheckResult] = []
        for check_index, check in enumerate(schema.checks):
            try:
                check_results.append(
                    self.run_check(check_obj, schema, check, check_index)
                )
            except SchemaDefinitionError:
                raise
            except Exception as err:
                # catch other exceptions that may occur when executing the check
                err_msg = f'"{err.args[0]}"' if err.args else ""
                err_str = f"{err.__class__.__name__}({err_msg})"
                msg = (
                    f"Error while executing check function: {err_str}\n"
                    + traceback.format_exc()
                )
                check_results.append(
                    CoreCheckResult(
                        passed=False,
                        check=check,
                        check_index=check_index,
                        reason_code=SchemaErrorReason.CHECK_ERROR,
                        message=msg,
                        failure_cases=err_str,
                        original_exc=err,
                    )
                )
        return check_results

    def run_schema_component_checks(
        self,
        check_obj,
        schema,
        schema_components: list,
        lazy: bool,
    ) -> list[CoreCheckResult]:
        """Run checks for all schema components."""
        check_results = []
        check_passed = []
        # Convert to native frame for column component dispatch.
        # Column.validate() calls get_backend(check_obj) which looks up by native
        # type — native polars LazyFrame for polars schemas, ibis.Table for ibis schemas.
        native_obj = _to_native(check_obj)
        # schema-component-level checks
        for schema_component in schema_components:
            try:
                result = schema_component.validate(native_obj, lazy=lazy)
                # Narwhals backend returns a Narwhals frame, not pl.LazyFrame.
                # The component validate() not raising is the success signal.
                check_passed.append(result is not None)
            except SchemaError as err:
                check_results.append(
                    CoreCheckResult(
                        passed=False,
                        check="schema_component_checks",
                        reason_code=SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                        schema_error=err,
                    )
                )
            except SchemaErrors as err:
                check_results.extend(
                    [
                        CoreCheckResult(
                            passed=False,
                            check="schema_component_checks",
                            reason_code=SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                            schema_error=schema_error,
                        )
                        for schema_error in err.schema_errors
                    ]
                )
        assert all(check_passed)
        return check_results

    def collect_column_info(self, check_obj, schema):
        """Collect column metadata for the dataframe."""
        # Use collect_schema().names() — lazy-safe Narwhals equivalent of
        # get_lazyframe_column_names()
        frame_column_names = check_obj.collect_schema().names()

        column_names: list[Any] = []
        absent_column_names: list[Any] = []
        regex_match_patterns: list[Any] = []

        for col_name, col_schema in schema.columns.items():
            if (
                not col_schema.regex
                and col_name not in frame_column_names
                and col_schema.required
            ):
                absent_column_names.append(col_name)

            if col_schema.regex:
                try:
                    column_names.extend(
                        col_schema.get_backend(check_obj).get_regex_columns(
                            col_schema, check_obj
                        )
                    )
                    regex_match_patterns.append(col_schema.selector)
                except SchemaError:
                    pass
            elif col_name in frame_column_names:
                column_names.append(col_name)

        # drop adjacent duplicated column names
        destuttered_column_names = list(frame_column_names)

        return ColumnInfo(
            sorted_column_names=dict.fromkeys(column_names),
            expanded_column_names=frozenset(column_names),
            destuttered_column_names=destuttered_column_names,
            absent_column_names=absent_column_names,
            regex_match_patterns=regex_match_patterns,
        )

    def collect_schema_components(
        self,
        check_obj,
        schema,
        column_info: ColumnInfo,
    ):
        """Collects all schema components to use for validation."""

        columns: dict = schema.columns
        frame_column_names = check_obj.collect_schema().names()

        if not schema.columns and schema.dtype is not None:
            # set schema components to dataframe dtype if columns are not
            # specified but the dataframe-level dtype is specified.
            columns = {
                col_name: col
                for col_name, col in zip(
                    frame_column_names,
                    schema.infer_columns(frame_column_names),
                )
            }

        schema_components = []
        for col_name, col in columns.items():
            if (
                col.required  # type: ignore
                or col_name in frame_column_names
                or (
                    column_info.regex_match_patterns is not None
                    and col.selector in column_info.regex_match_patterns
                )
            ) and col_name not in column_info.absent_column_names:
                col = copy.deepcopy(col)
                if schema.dtype is not None:
                    # override column dtype with dataframe dtype
                    col.dtype = schema.dtype  # type: ignore

                # disable coercion at the schema component level since the
                # dataframe-level schema already coerced it.
                col.coerce = False  # type: ignore
                schema_components.append(col)

        return schema_components

    ###########
    # Parsers #
    ###########

    def strict_filter_columns(
        self,
        check_obj,
        schema,
        column_info: ColumnInfo,
    ):
        """Filter columns that aren't specified in the schema."""
        # dataframe strictness check makes sure all columns in the dataframe
        # are specified in the dataframe schema
        if not (schema.strict or schema.ordered):
            return check_obj

        filter_out_columns = []
        sorted_column_names = iter(column_info.sorted_column_names)
        for column in column_info.destuttered_column_names:
            is_schema_col = column in column_info.expanded_column_names
            if schema.strict is True and not is_schema_col:
                raise SchemaError(
                    schema=schema,
                    data=check_obj,
                    message=(
                        f"column '{column}' not in {schema.__class__.__name__}"
                        f" {schema.columns}"
                    ),
                    failure_cases=column,
                    check="column_in_schema",
                    reason_code=SchemaErrorReason.COLUMN_NOT_IN_SCHEMA,
                )
            if schema.strict == "filter" and not is_schema_col:
                filter_out_columns.append(column)
            if schema.ordered and is_schema_col:
                try:
                    next_ordered_col = next(sorted_column_names)
                except StopIteration:
                    pass
                if next_ordered_col != column:
                    raise SchemaError(
                        schema=schema,
                        data=check_obj,
                        message=f"column '{column}' out-of-order",
                        failure_cases=column,
                        check="column_ordered",
                        reason_code=SchemaErrorReason.COLUMN_NOT_ORDERED,
                    )

        if schema.strict == "filter":
            check_obj = check_obj.drop(filter_out_columns)

        return check_obj

    ##########
    # Checks #
    ##########

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_column_presence(
        self,
        check_obj,
        schema,
        column_info: Any,
    ) -> list[CoreCheckResult]:
        """Check that all columns in the schema are present in the dataframe."""
        results = []
        if column_info.absent_column_names and not schema.add_missing_columns:
            for colname in column_info.absent_column_names:
                if is_regex(colname):
                    # don't raise an error if the column schema name is a
                    # regex pattern — try to select using regex expression
                    try:
                        frame_cols = check_obj.collect_schema().names()
                        matching = [c for c in frame_cols if re.search(colname, c)]
                        if matching:
                            continue
                    except Exception:
                        pass
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="column_in_dataframe",
                        reason_code=SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
                        message=(
                            f"column '{colname}' not in dataframe"
                            f"\n{_to_native(check_obj.head())}"
                        ),
                        failure_cases=colname,
                    )
                )
        return results

    @validate_scope(scope=ValidationScope.DATA)
    def check_column_values_are_unique(
        self,
        check_obj,
        schema,
    ) -> CoreCheckResult:
        """Check that column values are unique."""

        passed = True
        message = None
        failure_cases = None

        if not schema.unique:
            return CoreCheckResult(
                passed=passed,
                check="multiple_fields_uniqueness",
            )

        temp_unique: list[list] = (
            [schema.unique]
            if all(isinstance(x, str) for x in schema.unique)
            else schema.unique
        )
        frame_column_names = check_obj.collect_schema().names()
        check_output = None
        for lst in temp_unique:
            subset = [x for x in lst if x in frame_column_names]
            grouped = (
                check_obj
                .select(subset)
                .group_by(*[nw.col(c) for c in subset])
                .agg(nw.len().alias("_count"))
            )
            dup_rows = grouped.filter(nw.col("_count") > 1).drop("_count")
            # Bounded: dup_rows contains only rows with duplicate key values — not the full frame.
            # Materialization is required here to evaluate len() and produce failure_cases.
            native_dups = nw.to_native(_materialize(dup_rows))

            if len(native_dups) > 0:
                failure_cases = native_dups
                passed = False
                message = (
                    f"columns '{(*subset,)}' not unique:\n{failure_cases}"
                )
                break
        return CoreCheckResult(
            passed=passed,
            check="multiple_fields_uniqueness",
            reason_code=SchemaErrorReason.DUPLICATES,
            message=message,
            failure_cases=failure_cases,
            check_output=check_output,
        )
