"""Ibis parsing, validation, and error-reporting backends."""

from __future__ import annotations

import copy
import traceback
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Optional

import ibis
from ibis import _
from ibis import selectors as s
from ibis.common.exceptions import IbisError

from pandera.api.base.error_handler import get_error_category
from pandera.api.ibis.error_handler import ErrorHandler
from pandera.backends.base import ColumnInfo, CoreCheckResult
from pandera.backends.ibis.base import IbisSchemaBackend
from pandera.backends.utils import convert_uniquesettings
from pandera.config import ValidationScope
from pandera.errors import (
    ParserError,
    SchemaDefinitionError,
    SchemaError,
    SchemaErrorReason,
    SchemaErrors,
)
from pandera.utils import is_regex
from pandera.validation_depth import validate_scope, validation_type

if TYPE_CHECKING:
    from pandera.api.ibis.container import DataFrameSchema


class DataFrameSchemaBackend(IbisSchemaBackend):
    """Backend for Ibis DataFrameSchema."""

    def validate(
        self,
        check_obj: ibis.Table,
        schema: DataFrameSchema,
        *,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ):
        """
        Parse and validate a check object, returning type-coerced and validated
        object.
        """
        error_handler = ErrorHandler(lazy)

        column_info = self.collect_column_info(check_obj, schema)

        core_parsers: list[tuple[Callable[..., Any], tuple[Any, ...]]] = [
            (self.strict_filter_columns, (schema, column_info)),
        ]

        for parser, args in core_parsers:
            try:
                check_obj = parser(check_obj, *args)
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
            check_obj, schema, column_info
        )

        # TODO(deepyaman): subsample the check object if head, tail, or sample are specified
        sample = check_obj

        # run the checks
        core_checks = [
            (self.check_column_presence, (check_obj, schema, column_info)),
            (self.check_column_values_are_unique, (check_obj, schema)),
            (
                self.run_schema_component_checks,
                (sample, schema, components, lazy),
            ),
            (self.run_checks, (sample, schema)),
        ]

        for check, args in core_checks:
            results = check(*args)
            if isinstance(results, CoreCheckResult):
                results = [results]

            for result in results:
                if result.passed:
                    continue

                if result.schema_error is not None:
                    error = result.schema_error
                else:
                    error = SchemaError(
                        schema,
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

        if error_handler.collected_errors:
            if getattr(schema, "drop_invalid_rows", False):
                check_obj = self.drop_invalid_rows(check_obj, error_handler)
                return check_obj
            else:
                raise SchemaErrors(
                    schema=schema,
                    schema_errors=error_handler.schema_errors,
                    data=check_obj,
                )

        return check_obj

    @validate_scope(scope=ValidationScope.DATA)
    def run_checks(
        self,
        check_obj: ibis.Table,
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
        check_obj: ibis.Table,
        schema,
        schema_components: Iterable,
        lazy: bool,
    ) -> list[CoreCheckResult]:
        """Run checks for all schema components."""
        check_results = []
        check_passed = []
        # schema-component-level checks
        for schema_component in schema_components:
            try:
                result = schema_component.validate(check_obj, lazy=lazy)
                check_passed.append(isinstance(result, ibis.Table))
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

    def collect_column_info(
        self, check_obj: ibis.Table, schema: DataFrameSchema
    ) -> ColumnInfo:
        """Collect column metadata for the table."""
        column_names: list[Any] = []
        absent_column_names: list[Any] = []
        regex_match_patterns: list[Any] = []

        for col_name, col_schema in schema.columns.items():
            if (
                not col_schema.regex
                and col_name not in check_obj
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
                    regex_match_patterns.append(col_schema.name)
                except SchemaError:
                    pass
            elif col_name in check_obj:
                column_names.append(col_name)

        # Ibis tables cannot have duplicated column names
        destuttered_column_names = check_obj.columns

        return ColumnInfo(
            sorted_column_names=dict.fromkeys(column_names),
            expanded_column_names=frozenset(column_names),
            destuttered_column_names=destuttered_column_names,
            absent_column_names=absent_column_names,
            regex_match_patterns=regex_match_patterns,
        )

    def collect_schema_components(
        self,
        check_obj: ibis.Table,
        schema: DataFrameSchema,
        column_info: ColumnInfo,
    ):
        """Collects all schema components to use for validation."""

        columns = schema.columns

        if not schema.columns and schema.dtype is not None:
            # set schema components to dataframe dtype if columns are not
            # specified but the dataframe-level dtype is specified.
            from pandera.api.ibis.components import Column

            columns = {}
            for col in check_obj.columns:
                columns[col] = Column(schema.dtype, name=str(col))

        schema_components = []
        for col_name, col in columns.items():
            if (
                col.required  # type: ignore
                or col_name in check_obj
                or (
                    column_info.regex_match_patterns is not None
                    and col_name in column_info.regex_match_patterns
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
        check_obj: ibis.Table,
        schema: DataFrameSchema,
        column_info: ColumnInfo,
    ) -> ibis.Table:
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

    def check_column_names_are_unique(
        self,
        check_obj: ibis.Table,
        schema,
    ) -> CoreCheckResult:
        """Check that column names are unique."""
        raise NotImplementedError(
            "Ibis does not support duplicate column names"
        )

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_column_presence(
        self,
        check_obj: ibis.Table,
        schema: DataFrameSchema,
        column_info: ColumnInfo,
    ) -> list[CoreCheckResult]:
        """Check that all columns in the schema are present in the table."""
        results = []
        if column_info.absent_column_names and not schema.add_missing_columns:
            for colname in column_info.absent_column_names:
                if is_regex(colname):
                    try:
                        # don't raise an error if the column schema name is a
                        # regex pattern
                        check_obj.select(s.matches(colname))
                        continue
                    except IbisError:
                        # regex pattern didn't match any columns
                        pass
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="column_in_dataframe",
                        reason_code=SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
                        message=(
                            f"column '{colname}' not in table. "
                            f"Columns in table: {check_obj.columns}"
                        ),
                        failure_cases=colname,
                    )
                )
        return results

    @validate_scope(scope=ValidationScope.DATA)
    def check_column_values_are_unique(
        self,
        check_obj: ibis.Table,
        schema: DataFrameSchema,
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

        keep_setting = convert_uniquesettings(schema.report_duplicates)
        temp_unique: list[list] = (
            [schema.unique]
            if all(isinstance(x, str) for x in schema.unique)
            else schema.unique
        )
        for lst in temp_unique:
            subset = [x for x in lst if x in check_obj]
            if keep_setting == "first":
                duplicated = ibis.row_number().over(group_by=subset) > 0
            elif keep_setting == "last":
                duplicated = (_.count() - ibis.row_number()).over(
                    group_by=subset
                ) > 1
            else:
                duplicated = _.count().over(group_by=subset) > 1
            duplicates = check_obj.select(duplicated=duplicated).duplicated
            if duplicates.any().execute():
                failure_cases = check_obj.filter(duplicated)
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
        )
