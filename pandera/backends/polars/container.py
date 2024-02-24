"""Validation backend for polars DataFrameSchema."""

from typing import Any, Optional, List, Callable, Tuple

import polars as pl

from pandera.api.polars.container import DataFrameSchema
from pandera.backends.base import CoreCheckResult
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

        column_info = self.collect_column_info(check_obj, schema)  # TODO

        core_parsers: List[Tuple[Callable[..., Any], Tuple[Any, ...]]] = [
            ((self.add_missing_columns), (schema, column_info)),  # TODO
            (self.strict_filter_columns, (schema, column_info)),  # TODO
            (self.coerce_dtype, (schema,)),  # TODO
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
        raise NotImplementedError

    ###########
    # Parsers #
    ###########

    def add_missing_columns(
        self,
        check_obj: pl.LazyFrame,
        schema,
        column_info: Any,
    ):
        """Add missing columns to the dataframe."""
        raise NotImplementedError

    def strict_filter_columns(
        self,
        check_obj: pl.LazyFrame,
        schema,
        column_info: Any,
    ):
        """Filter columns based on schema."""
        raise NotImplementedError

    def coerce_dtype(self, check_obj: pl.LazyFrame, schema=None):
        """Coerce dataframe columns to the correct dtype."""
        raise NotImplementedError

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
