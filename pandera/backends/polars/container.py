"""Validation backend for polars DataFrameSchema."""

import copy
import traceback
import warnings
from typing import Any, Callable, List, Optional, Tuple

import polars as pl

from pandera.api.base.error_handler import ErrorHandler
from pandera.api.polars.container import DataFrameSchema
from pandera.api.polars.types import PolarsData
from pandera.backends.base import ColumnInfo, CoreCheckResult
from pandera.backends.polars.base import PolarsSchemaBackend
from pandera.config import ValidationDepth, ValidationScope, get_config_context
from pandera.errors import (
    ParserError,
    SchemaDefinitionError,
    SchemaError,
    SchemaErrorReason,
    SchemaErrors,
)
from pandera.utils import is_regex
from pandera.validation_depth import validate_scope, validation_type


class DataFrameSchemaBackend(PolarsSchemaBackend):
    # pylint: disable=too-many-branches
    def validate(
        self,
        check_obj: pl.LazyFrame,
        schema: DataFrameSchema,
        *,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ):
        if inplace:
            warnings.warn("setting inplace=True will have no effect.")

        error_handler = ErrorHandler(lazy)

        column_info = self.collect_column_info(check_obj, schema)

        if getattr(schema, "drop_invalid_rows", False) and not lazy:
            raise SchemaDefinitionError(
                "When drop_invalid_rows is True, lazy must be set to True."
            )

        core_parsers: List[Tuple[Callable[..., Any], Tuple[Any, ...]]] = [
            (self.add_missing_columns, (schema, column_info)),
            (self.strict_filter_columns, (schema, column_info)),
            (self.coerce_dtype, (schema,)),
            (self.set_default, (schema,)),
        ]

        for parser, args in core_parsers:
            try:
                check_obj = parser(check_obj, *args)
            except SchemaError as exc:
                error_handler.collect_error(
                    validation_type(exc.reason_code),
                    exc.reason_code,
                    exc,
                )
            except SchemaErrors as exc:
                error_handler.collect_errors(exc.schema_errors)

        components = self.collect_schema_components(
            check_obj,
            schema,
            column_info,
        )

        # subsample the check object if head, tail, or sample are specified
        sample = self.subsample(check_obj, head, tail, sample, random_state)

        core_checks = [
            (self.check_column_presence, (check_obj, schema, column_info)),
            (self.check_column_values_are_unique, (sample, schema)),
            (self.run_schema_component_checks, (sample, components, lazy)),
            (self.run_checks, (sample, schema)),
        ]

        for check, args in core_checks:
            results = check(*args)  # type: ignore[operator]
            if isinstance(results, CoreCheckResult):
                results = [results]

            # pylint: disable=no-member
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
                    validation_type(result.reason_code),
                    result.reason_code,
                    error,
                    original_exc=result.original_exc,
                )

        if error_handler.collected_errors:
            if getattr(schema, "drop_invalid_rows", False):
                check_obj = self.drop_invalid_rows(check_obj, error_handler)
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
        check_obj: pl.LazyFrame,
        schema,
    ) -> List[CoreCheckResult]:
        """Run a list of checks on the check object."""
        # dataframe-level checks
        check_results: List[CoreCheckResult] = []
        for check_index, check in enumerate(schema.checks):
            try:
                check_results.append(
                    self.run_check(check_obj, schema, check, check_index)
                )
            except SchemaDefinitionError:
                raise
            except Exception as err:  # pylint: disable=broad-except
                # catch other exceptions that may occur when executing the check
                err_msg = f'"{err.args[0]}"' if len(err.args) > 0 else ""
                err_str = f"{err.__class__.__name__}({ err_msg})"
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
        check_obj: pl.LazyFrame,
        schema_components: List,
        lazy: bool,
    ) -> List[CoreCheckResult]:
        """Run checks for all schema components."""
        check_results = []
        check_passed = []
        # schema-component-level checks
        for schema_component in schema_components:
            try:
                result = schema_component.validate(check_obj, lazy=lazy)
                check_passed.append(isinstance(result, pl.LazyFrame))
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

    def collect_column_info(self, check_obj: pl.LazyFrame, schema):
        """Collect column metadata for the dataframe."""
        column_names: List[Any] = []
        absent_column_names: List[Any] = []
        regex_match_patterns: List[Any] = []

        for col_name, col_schema in schema.columns.items():
            if (
                not col_schema.regex
                and col_name not in check_obj.columns
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
            elif col_name in check_obj.columns:
                column_names.append(col_name)

        # drop adjacent duplicated column names
        destuttered_column_names = [*check_obj.columns]

        return ColumnInfo(
            sorted_column_names=dict.fromkeys(column_names),
            expanded_column_names=frozenset(column_names),
            destuttered_column_names=destuttered_column_names,
            absent_column_names=absent_column_names,
            regex_match_patterns=regex_match_patterns,
        )

    def collect_schema_components(
        self,
        check_obj: pl.LazyFrame,
        schema,
        column_info: ColumnInfo,
    ):
        """Collects all schema components to use for validation."""

        columns = schema.columns

        if not schema.columns and schema.dtype is not None:
            # set schema components to dataframe dtype if columns are not
            # specified by the dataframe-level dtype is specified.
            from pandera.api.polars.components import Column

            columns = {}
            for col in check_obj.columns:
                columns[col] = Column(schema.dtype, name=str(col))

        schema_components = []
        for col_name, col in columns.items():
            if (
                col.required  # type: ignore
                or col_name in check_obj
                or col_name in column_info.regex_match_patterns
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

    def add_missing_columns(
        self,
        check_obj: pl.LazyFrame,
        schema,
        column_info: ColumnInfo,
    ):
        """Add columns that aren't in the dataframe."""
        # Add missing columns to dataframe based on 'add_missing_columns'
        # schema property

        if not (
            column_info.absent_column_names and schema.add_missing_columns
        ):
            return check_obj

        # Absent columns are required to have a default value or be nullable
        for col_name in column_info.absent_column_names:
            col_schema = schema.columns[col_name]
            if col_schema.default is None and not col_schema.nullable:
                raise SchemaError(
                    schema=schema,
                    data=check_obj,
                    message=(
                        f"column '{col_name}' in {schema.__class__.__name__}"
                        f" {schema.columns} requires a default value "
                        f"when non-nullable add_missing_columns is enabled"
                    ),
                    failure_cases=col_name,
                    check="add_missing_has_default",
                    reason_code=SchemaErrorReason.ADD_MISSING_COLUMN_NO_DEFAULT,
                )

        # Create companion dataframe of default values for missing columns
        missing_cols_schema = {
            k: v
            for k, v in schema.columns.items()
            if k in column_info.absent_column_names
        }

        # Append missing columns
        check_obj = check_obj.with_columns(
            **{k: v.default for k, v in missing_cols_schema.items()}
        ).cast({k: v.dtype.type for k, v in missing_cols_schema.items()})

        # Set column order
        check_obj = check_obj.select([*schema.columns])
        return check_obj

    def strict_filter_columns(
        self,
        check_obj: pl.LazyFrame,
        schema,
        column_info: ColumnInfo,
    ) -> pl.LazyFrame:
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

    def coerce_dtype(self, check_obj: pl.LazyFrame, schema=None):
        """Coerce dataframe columns to the correct dtype."""
        assert schema is not None, "The `schema` argument must be provided."

        error_handler = ErrorHandler(lazy=True)

        if not (
            schema.coerce or any(col.coerce for col in schema.columns.values())
        ):
            return check_obj

        try:
            check_obj = self._coerce_dtype_helper(check_obj, schema)
        except SchemaErrors as err:
            for schema_error in err.schema_errors:
                error_handler.collect_error(
                    validation_type(SchemaErrorReason.SCHEMA_COMPONENT_CHECK),
                    SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                    schema_error,
                )
        except SchemaError as err:
            error_handler.collect_error(
                validation_type(SchemaErrorReason.SCHEMA_COMPONENT_CHECK),
                SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                err,
            )

        if error_handler.collected_errors:
            # raise SchemaErrors if this method is called without an
            # error_handler
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.schema_errors,
                data=check_obj,
            )

        return check_obj

    def _coerce_dtype_helper(
        self,
        obj: pl.LazyFrame,
        schema,
    ) -> pl.LazyFrame:
        """Coerce dataframe to the type specified in dtype.

        :param obj: dataframe to coerce.
        :returns: dataframe with coerced dtypes
        """
        error_handler = ErrorHandler(lazy=True)

        config_ctx = get_config_context(validation_depth_default=None)

        # If validation depth involves validating data, use try_coerce since we
        # want to check actual data values. Otherwise, coerce simply detects
        # datatype mismatches.
        coerce_fn: str = (
            "try_coerce"
            if config_ctx.validation_depth
            in (
                ValidationDepth.SCHEMA_AND_DATA,
                ValidationDepth.DATA_ONLY,
            )
            else "coerce"
        )

        try:
            if schema.dtype is not None:
                obj = getattr(schema.dtype, coerce_fn)(obj)
            else:
                for col_schema in schema.columns.values():
                    if schema.coerce or col_schema.coerce:
                        obj = getattr(col_schema.dtype, coerce_fn)(
                            PolarsData(obj, col_schema.selector)
                        )
        except ParserError as exc:
            error_handler.collect_error(
                validation_type(SchemaErrorReason.DATATYPE_COERCION),
                SchemaErrorReason.DATATYPE_COERCION,
                SchemaError(
                    schema=schema,
                    data=obj,
                    message=exc.args[0],
                    check=f"coerce_dtype('{schema.dtypes}')",
                    reason_code=SchemaErrorReason.DATATYPE_COERCION,
                    failure_cases=exc.failure_cases,
                    check_output=exc.parser_output,
                ),
            )
        except pl.ComputeError as exc:
            error_handler.collect_error(
                validation_type(SchemaErrorReason.DATATYPE_COERCION),
                SchemaErrorReason.DATATYPE_COERCION,
                SchemaError(
                    schema=schema,
                    data=obj,
                    message=(
                        f"Error while coercing '{schema.name}' to type "
                        f"{schema.dtype}: {exc}"
                    ),
                    check=f"coerce_dtype('{schema.dtypes}')",
                    reason_code=SchemaErrorReason.DATATYPE_COERCION,
                ),
            )

        if error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.schema_errors,
                data=obj,
            )

        return obj

    def set_default(self, check_obj: pl.LazyFrame, schema) -> pl.LazyFrame:
        """Set default values for columns with missing values."""

        for col_schema in [
            s
            for s in schema.columns.values()
            if hasattr(s, "default") and s.default is not None
        ]:
            backend = col_schema.get_backend(check_obj)
            check_obj = backend.set_default(check_obj, col_schema)

        return check_obj

    ##########
    # Checks #
    ##########

    def check_column_names_are_unique(
        self,
        check_obj: pl.LazyFrame,
        schema,
    ) -> CoreCheckResult:
        """Check that column names are unique."""
        raise NotImplementedError(
            "polars does not support duplicate column names"
        )

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_column_presence(
        self,
        check_obj: pl.LazyFrame,
        schema,
        column_info: Any,
    ) -> List[CoreCheckResult]:
        """Check that all columns in the schema are present in the dataframe."""
        results = []
        if column_info.absent_column_names and not schema.add_missing_columns:
            for colname in column_info.absent_column_names:
                if (
                    is_regex(colname)
                    and check_obj.select(pl.col(colname)).columns
                ):
                    # don't raise an error if the column schema name is a
                    # regex pattern
                    continue
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="column_in_dataframe",
                        reason_code=SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
                        message=(
                            f"column '{colname}' not in dataframe"
                            f"\n{check_obj.head()}"
                        ),
                        failure_cases=colname,
                    )
                )
        return results

    @validate_scope(scope=ValidationScope.DATA)
    def check_column_values_are_unique(
        self,
        check_obj: pl.LazyFrame,
        schema,
    ) -> CoreCheckResult:
        """Check that column values are unique."""

        passed = True
        message = None
        failure_cases = None

        if not schema.unique:
            return CoreCheckResult(
                passed=passed,
                check="dataframe_column_labels_unique",
            )

        # NOTE: fix this pylint error
        # pylint: disable=not-an-iterable
        temp_unique: List[List] = (
            [schema.unique]
            if all(isinstance(x, str) for x in schema.unique)
            else schema.unique
        )

        for lst in temp_unique:
            subset = [x for x in lst if x in check_obj.columns]
            duplicates = check_obj.select(subset).collect().is_duplicated()
            if duplicates.any():
                failure_cases = check_obj.filter(duplicates)

                passed = False
                message = f"columns '{*subset,}' not unique:\n{failure_cases}"
                break
        return CoreCheckResult(
            passed=passed,
            check="multiple_fields_uniqueness",
            reason_code=SchemaErrorReason.DUPLICATES,
            message=message,
            failure_cases=failure_cases,
        )
