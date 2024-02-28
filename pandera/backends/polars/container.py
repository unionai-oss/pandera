"""Validation backend for polars DataFrameSchema."""

import itertools
from typing import Any, Optional, List, Callable, Tuple

import polars as pl

from pandera.api.polars.container import DataFrameSchema
from pandera.backends.base import CoreCheckResult, ColumnInfo
from pandera.backends.polars.base import PolarsSchemaBackend
from pandera.error_handlers import SchemaErrorHandler
from pandera.errors import (
    SchemaError,
    SchemaErrors,
    SchemaErrorReason,
)


class DataFrameSchemaBackend(PolarsSchemaBackend):
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
        error_handler = SchemaErrorHandler(lazy)

        column_info = self.collect_column_info(check_obj, schema)

        core_parsers: List[Tuple[Callable[..., Any], Tuple[Any, ...]]] = [
            (self.add_missing_columns, (schema, column_info)),
            (self.strict_filter_columns, (schema, column_info)),
            (self.coerce_dtype, (schema,)),
        ]

        for parser, args in core_parsers:
            try:
                check_obj = parser(check_obj, *args)
            except SchemaError as exc:
                error_handler.collect_error(exc.reason_code, exc)
            except SchemaErrors as exc:
                error_handler.collect_errors(exc)

        components = [v for _, v in schema.columns.items()]

        core_checks = [
            # (self.check_column_names_are_unique, (check_obj, schema)),  # TODO
            # (self.check_column_presence, (check_obj, schema, column_info)),  # TODO
            # (self.check_column_values_are_unique, (sample, schema)),  # TODO
            (self.run_schema_component_checks, (check_obj, components, lazy))
            # (self.run_checks, (sample, schema)),  # TODO
        ]

        for check, args in core_checks:
            results = check(*args)
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
                    result.reason_code,
                    error,
                    original_exc=result.original_exc,
                )

        if error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.collected_errors,
                data=check_obj,
            )

        return check_obj

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
                result = schema_component.validate(
                    check_obj, lazy=lazy, inplace=True
                )
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
                    # TODO: implement get_regex_columns
                    column_names.extend(
                        col_schema.get_backend(check_obj).get_regex_columns(
                            col_schema, check_obj.columns
                        )
                    )
                    regex_match_patterns.append(col_schema.name)
                except SchemaError:
                    pass
            elif col_name in check_obj.columns:
                column_names.append(col_name)

        # drop adjacent duplicated column names
        destuttered_column_names = [*check_obj.columns]
        if len(check_obj.columns) != len(set(check_obj.columns)):
            destuttered_column_names = [
                k for k, _ in itertools.groupby(check_obj.columns)
            ]

        return ColumnInfo(
            sorted_column_names=dict.fromkeys(column_names),
            expanded_column_names=frozenset(column_names),
            destuttered_column_names=destuttered_column_names,
            absent_column_names=absent_column_names,
            regex_match_patterns=regex_match_patterns,
        )

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

        if not column_info.absent_column_names and schema.add_missing_columns:
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

        error_handler = SchemaErrorHandler(lazy=True)

        if not (
            schema.coerce or any(col.coerce for col in schema.columns.values())
        ):
            return check_obj

        try:
            check_obj = self._coerce_dtype_helper(check_obj, schema)
        except SchemaErrors as err:
            for schema_error in err.schema_errors:
                error_handler.collect_error(
                    SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                    schema_error,
                )
        except SchemaError as err:
            error_handler.collect_error(
                SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                err,
            )

        if error_handler.collected_errors:
            # raise SchemaErrors if this method is called without an
            # error_handler
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.collected_errors,
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
        error_handler = SchemaErrorHandler(lazy=True)

        if schema.dtype is not None:
            obj = obj.cast(schema.dtype.type)
        else:
            # TODO: support coercion of regex columns
            obj = obj.cast({k: v.type for k, v in schema.dtypes.items()})

        try:
            obj = obj.collect().lazy()
        except pl.exceptions.ComputeError as exc:
            error_handler.collect_error(
                SchemaErrorReason.DATATYPE_COERCION,
                SchemaError(
                    schema=schema,
                    data=obj,
                    message=(
                        f"Error while coercing '{schema.name}' to type "
                        f"{schema.dtype}: {exc}"
                    ),
                    check=f"coerce_dtype('{schema.dtypes}')",
                ),
            )

        if error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.collected_errors,
                data=obj,
            )

        return obj

    ##########
    # Checks #
    ##########

    def check_column_names_are_unique(
        self,
        check_obj: pl.LazyFrame,
        schema,
    ) -> CoreCheckResult:
        """Check that column names are unique."""
        raise NotImplementedError

    def check_column_presence(
        self,
        check_obj: pl.LazyFrame,
        schema,
        column_info: Any,
    ) -> List[CoreCheckResult]:
        """Check that all columns in the schema are present in the dataframe."""
        raise NotImplementedError

    def check_column_values_are_unique(
        self,
        check_obj: pl.LazyFrame,
        schema,
    ) -> CoreCheckResult:
        """Check that column values are unique."""
        raise NotImplementedError
