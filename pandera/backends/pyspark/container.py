"""PySpark parsing, validation, and error-reporting backends."""

import copy
import traceback
import warnings
from collections.abc import Callable
from typing import Any, Optional

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count

from pandera.api.base.error_handler import (
    ErrorCategory,
    ErrorHandler,
    get_error_category,
)
from pandera.api.pyspark.types import is_table
from pandera.backends.base import ColumnInfo, CoreCheckResult
from pandera.backends.pyspark.base import PysparkSchemaBackend
from pandera.backends.pyspark.decorators import cache_check_obj
from pandera.backends.pyspark.error_formatters import scalar_failure_case
from pandera.config import get_config_context
from pandera.errors import (
    SchemaDefinitionError,
    SchemaError,
    SchemaErrorReason,
    SchemaErrors,
)
from pandera.validation_depth import ValidationScope, validate_scope


class DataFrameSchemaBackend(PysparkSchemaBackend):
    """Backend for PySpark DataFrameSchema."""

    def preprocess(self, check_obj: DataFrame, inplace: bool = False):
        """Preprocesses a check object before applying check functions."""
        return check_obj

    @cache_check_obj()
    def validate(
        self,
        check_obj: DataFrame,
        schema,
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
        error_handler = ErrorHandler(lazy=lazy)

        if not get_config_context().validation_enabled:
            warnings.warn(
                "Skipping the validation checks as validation is disabled"
            )
            return check_obj
        if not is_table(check_obj):
            raise TypeError(
                f"expected a pyspark DataFrame, got {type(check_obj)}"
            )

        check_obj = self.preprocess(check_obj, inplace=inplace)
        if hasattr(check_obj, "pandera"):
            check_obj = check_obj.pandera.add_schema(schema)  # type: ignore

        column_info = self.collect_column_info(check_obj, schema, lazy)

        core_parsers: list[tuple[Callable[..., Any], tuple[Any, ...]]] = [
            (self.strict_filter_columns, (schema, column_info)),
            (self.coerce_dtype, (schema,)),
        ]

        for parser, args in core_parsers:
            try:
                check_obj = parser(check_obj, *args)
            except SchemaError as exc:
                error_handler.collect_error(
                    get_error_category(exc.reason_code), exc.reason_code, exc
                )
            except SchemaErrors as exc:
                error_handler.collect_errors(exc.schema_errors)

        # We may have modified columns, for example by
        # add_missing_columns, so regenerate column info
        column_info = self.collect_column_info(check_obj, schema, lazy)

        # subsample the check object if sample is specified
        if sample is not None:
            check_obj_sample = self.subsample(
                check_obj, sample=sample, random_state=random_state
            )
        else:
            check_obj_sample = check_obj

        # collect schema components
        schema_components = self.collect_schema_components(
            check_obj, schema, column_info
        )

        # check the container metadata, e.g. field names
        core_checks = [
            (self.check_column_names_are_unique, (check_obj, schema)),
            (self.check_column_presence, (check_obj, schema, column_info)),
            (self.check_column_values_are_unique, (check_obj_sample, schema)),
            (
                self.run_schema_component_checks,
                (check_obj_sample, schema, schema_components, lazy),
            ),
            (self.run_checks, (check_obj_sample, schema)),
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
                    result.original_exc,
                )

        error_dicts = {}
        if error_handler.collected_errors:
            error_dicts = error_handler.summarize(schema_name=schema.name)

        check_obj.pandera.errors = error_dicts  # type: ignore
        return check_obj

    def run_schema_component_checks(
        self,
        check_obj: DataFrame,
        schema,
        schema_components: list,
        lazy: bool,
    ) -> list[CoreCheckResult]:
        """Run checks for all schema components."""
        check_results: list[CoreCheckResult] = []
        check_passed: list[bool] = []

        # schema-component-level checks
        for schema_component in schema_components:
            # make sure the schema component mutations are reverted after
            # validation
            _orig_dtype = schema_component.dtype
            _orig_coerce = schema_component.coerce

            try:
                result = schema_component.validate(
                    check_obj=check_obj,
                    lazy=lazy,
                    inplace=True,
                )
                passed = is_table(result)
                check_passed.append(passed)
                check_results.append(
                    CoreCheckResult(
                        passed=passed,
                        check="schema_component_checks",
                    )
                )
            except SchemaError as err:
                check_results.append(
                    CoreCheckResult(
                        passed=False,
                        check="schema_component_checks",
                        reason_code=err.reason_code
                        or SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                        schema_error=err,
                    )
                )
            except SchemaErrors as err:
                check_results.extend(
                    [
                        CoreCheckResult(
                            passed=False,
                            check="schema_component_checks",
                            reason_code=schema_error.reason_code
                            or SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                            schema_error=schema_error,
                        )
                        for schema_error in err.schema_errors
                    ]
                )
            finally:
                # revert the schema component mutations
                schema_component.dtype = _orig_dtype
                schema_component.coerce = _orig_coerce

        assert all(check_results)
        return check_results

    @validate_scope(scope=ValidationScope.DATA)
    def run_checks(
        self, check_obj: DataFrame, schema
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

    def collect_column_info(
        self,
        check_obj: DataFrame,
        schema,
        lazy: bool,
    ) -> ColumnInfo:
        """Collect column metadata."""
        column_names: list[Any] = []
        absent_column_names: list[Any] = []
        lazy_exclude_column_names: list[Any] = []

        for col_name, col_schema in schema.columns.items():
            if (
                not col_schema.regex
                and col_name not in check_obj.columns
                and col_schema.required
            ):
                absent_column_names.append(col_name)
                if lazy:
                    # NOTE: remove this since we can just use
                    # absent_column_names in the collect_schema_components
                    # method
                    lazy_exclude_column_names.append(col_name)

            if col_schema.regex:
                try:
                    column_names.extend(
                        col_schema.get_backend(check_obj).get_regex_columns(
                            col_schema, check_obj
                        )
                    )
                except SchemaError:
                    pass
            elif col_name in check_obj.columns:
                column_names.append(col_name)

        # drop adjacent duplicated column names

        destuttered_column_names = list(set(check_obj.columns))

        return ColumnInfo(
            sorted_column_names=dict.fromkeys(column_names),
            expanded_column_names=frozenset(column_names),
            destuttered_column_names=destuttered_column_names,
            absent_column_names=absent_column_names,
            lazy_exclude_column_names=lazy_exclude_column_names,
        )

    def collect_schema_components(
        self,
        check_obj: DataFrame,
        schema,
        column_info: ColumnInfo,
    ):
        """Collects all schema components to use for validation."""
        schema_components = []
        for col_name, column in schema.columns.items():
            if (
                column.required or col_name in check_obj.columns
            ) and col_name not in column_info.lazy_exclude_column_names:
                column = copy.deepcopy(column)
                if schema.dtype is not None:
                    # override column dtype with dataframe dtype
                    column.dtype = schema.dtype

                # disable coercion at the schema component level since the
                # dataframe-level schema already coerced it.
                column.coerce = False
                schema_components.append(column)

        return schema_components

    ###########
    # Parsers #
    ###########

    def strict_filter_columns(
        self,
        check_obj: DataFrame,
        schema,
        column_info: ColumnInfo,
    ):
        """Filters columns that aren't specified in the schema."""
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
                    failure_cases=scalar_failure_case(column),
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
                        failure_cases=scalar_failure_case(column),
                        check="column_ordered",
                        reason_code=SchemaErrorReason.COLUMN_NOT_ORDERED,
                    )

        if schema.strict == "filter":
            schema = check_obj.pandera.schema
            check_obj = check_obj.drop(*filter_out_columns)
            check_obj.pandera.add_schema(schema)  # type: ignore

        return check_obj

    def coerce_dtype(
        self,
        check_obj: DataFrame,
        schema=None,
    ):
        """Coerces check object to the expected type."""
        assert schema is not None, "The `schema` argument must be provided."
        error_handler = ErrorHandler()

        if not (
            schema.coerce or any(col.coerce for col in schema.columns.values())
        ):
            return check_obj

        try:
            check_obj = self._coerce_dtype(check_obj, schema)

        except SchemaErrors as err:
            for schema_error in err.schema_errors:
                error_handler.collect_error(
                    ErrorCategory.DTYPE_COERCION,
                    SchemaErrorReason.CHECK_ERROR,
                    schema_error,
                )
        except SchemaError as err:
            error_handler.collect_error(
                ErrorCategory.SCHEMA, err.reason_code, err
            )

        if error_handler.collected_errors and not error_handler.lazy:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.schema_errors,
                data=check_obj,
            )

        return check_obj

    def _coerce_dtype(
        self,
        obj: DataFrame,
        schema,
    ) -> DataFrame:
        """Coerce dataframe to the type specified in dtype.

        :param obj: dataframe to coerce.
        :param schema: schema object
        :returns: dataframe with coerced dtypes
        """
        # NOTE: clean up the error handling!
        error_handler = ErrorHandler(lazy=True)

        def _try_coercion(obj, colname, col_schema):
            try:
                schema = obj.pandera.schema

                obj = obj.withColumn(
                    colname, col(colname).cast(col_schema.dtype.type)
                )
                obj.pandera.add_schema(schema)
                return obj

            except SchemaError as exc:
                error_handler.collect_error(
                    ErrorCategory.DTYPE_COERCION, exc.reason_code, exc
                )
                return obj

        for colname, col_schema in schema.columns.items():
            if col_schema.regex:
                try:
                    matched_columns = col_schema.get_backend(
                        obj
                    ).get_regex_columns(col_schema, obj.columns)
                except SchemaError:
                    matched_columns = []

                for matched_colname in matched_columns:
                    if col_schema.coerce or schema.coerce:
                        obj = _try_coercion(
                            obj,
                            matched_colname,
                            col_schema,
                            # col_schema.coerce_dtype, obj[matched_colname]
                        )

            elif (
                (col_schema.coerce or schema.coerce)
                and schema.dtype is None
                and colname in obj.columns
            ):
                _col_schema = copy.deepcopy(col_schema)
                _col_schema.coerce = True
                obj = _try_coercion(obj, colname, col_schema)

        if error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.collected_errors,  # type: ignore
                data=obj,
            )

        return obj

    @validate_scope(scope=ValidationScope.DATA)
    def check_column_values_are_unique(
        self,
        check_obj: DataFrame,
        schema,
    ) -> CoreCheckResult:
        """Check uniqueness in the check object."""
        assert schema is not None, "The `schema` argument must be provided."

        passed = True
        failure_cases = None
        message = None

        if not schema.unique:
            return CoreCheckResult(
                passed=True,
                check="unique",
            )

        # Determine unique columns based on schema's config
        unique_columns = (
            [schema.unique]
            if isinstance(schema.unique, str)
            else schema.unique
        )

        # Check if values belong to the dataframe columns
        missing_unique_columns = set(unique_columns) - set(check_obj.columns)
        if missing_unique_columns:
            return CoreCheckResult(
                passed=False,
                check="unique",
                reason_code=SchemaErrorReason.DUPLICATES,
                message=(
                    f"Specified `unique` columns are missing in the dataframe: "
                    f"{list(missing_unique_columns)}"
                ),
            )

        # Filter out empty column names
        unique_columns = [col for col in unique_columns if col]
        if not unique_columns:
            return CoreCheckResult(
                passed=True,
                check="unique",
            )

        duplicates_count = (
            check_obj.select(*unique_columns)  # ignore other cols
            .groupby(*unique_columns)
            .agg(count("*").alias("pandera_duplicate_counts"))
            .filter(
                col("pandera_duplicate_counts") > 1
            )  # long name to avoid collisions
            .count()
        )

        if duplicates_count > 0:
            passed = False
            message = (
                f"Duplicated rows [{duplicates_count}] were found "
                f"for columns {unique_columns}"
            )
            failure_cases = unique_columns

        return CoreCheckResult(
            passed=passed,
            check="unique",
            reason_code=SchemaErrorReason.DUPLICATES,
            message=message,
            failure_cases=failure_cases,
        )

    ##########
    # Checks #
    ##########

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_column_names_are_unique(
        self,
        check_obj: DataFrame,
        schema,
    ) -> CoreCheckResult:
        """Check for column name uniqueness."""

        passed = True
        failure_cases = None
        message = None

        if not schema.unique_column_names:
            return CoreCheckResult(
                passed=True,
                check="dataframe_column_labels_unique",
            )

        column_count_dict: dict[Any, Any] = {}
        failed = []
        for column_name in check_obj.columns:
            if column_count_dict.get(column_name):
                # Insert to the list only once
                if column_count_dict[column_name] == 1:
                    failed.append(column_name)
                column_count_dict[column_name] += 1

            else:
                column_count_dict[column_name] = 0

        if failed:
            passed = False
            message = (
                f"dataframe contains multiple columns with label(s): {failed}"
            )
            failure_cases = failed

        return CoreCheckResult(
            passed=passed,
            check="dataframe_column_labels_unique",
            reason_code=SchemaErrorReason.DUPLICATE_COLUMN_LABELS,
            message=message,
            failure_cases=failure_cases,
        )

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_column_presence(
        self,
        check_obj: DataFrame,
        schema,
        column_info: ColumnInfo,
    ) -> list[CoreCheckResult]:
        """Check that all columns in the schema are present in the dataframe."""
        results = []
        if column_info.absent_column_names:
            for colname in column_info.absent_column_names:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="column_in_dataframe",
                        reason_code=SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
                        message=(
                            f"column '{colname}' not in dataframe"
                            f" {check_obj.head()}"
                        ),
                        failure_cases=scalar_failure_case(colname),
                    )
                )
        else:
            results.append(
                CoreCheckResult(
                    passed=True,
                    check="column_in_dataframe",
                )
            )
        return results
