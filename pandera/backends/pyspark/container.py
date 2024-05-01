"""Pyspark Parsing, Validation, and Error Reporting Backends."""

import copy
import traceback
import warnings
from typing import Any, Dict, List, Optional

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count

from pandera.api.base.error_handler import ErrorCategory, ErrorHandler
from pandera.api.pyspark.types import is_table
from pandera.backends.pyspark.base import ColumnInfo, PysparkSchemaBackend
from pandera.backends.pyspark.decorators import cache_check_obj, validate_scope
from pandera.backends.pyspark.error_formatters import scalar_failure_case
from pandera.config import get_config_context
from pandera.errors import (
    SchemaDefinitionError,
    SchemaError,
    SchemaErrorReason,
    SchemaErrors,
)
from pandera.validation_depth import ValidationScope


class DataFrameSchemaBackend(PysparkSchemaBackend):
    """Backend for pyspark DataFrameSchema."""

    def preprocess(self, check_obj: DataFrame, inplace: bool = False):
        """Preprocesses a check object before applying check functions."""
        return check_obj

    @validate_scope(scope=ValidationScope.SCHEMA)
    def _schema_checks(
        self,
        check_obj: DataFrame,
        schema,
        column_info: ColumnInfo,
        error_handler: ErrorHandler,
    ):
        """run the checks related to columns presence, strictness and filter column if neccesary"""

        # check the container metadata, e.g. field names
        try:
            self.check_column_names_are_unique(check_obj, schema)
        except SchemaError as exc:
            error_handler.collect_error(
                error_type=ErrorCategory.SCHEMA,
                reason_code=exc.reason_code,
                schema_error=exc,
            )

        try:
            self.check_column_presence(check_obj, schema, column_info)
        except SchemaErrors as exc:
            for schema_error in exc.schema_errors:
                error_handler.collect_error(
                    error_type=ErrorCategory.SCHEMA,
                    reason_code=schema_error["reason_code"],
                    schema_error=schema_error["error"],
                )

        # strictness check and filter
        try:
            check_obj = self.strict_filter_columns(
                check_obj, schema, column_info, error_handler
            )
        except SchemaError as exc:
            error_handler.collect_error(
                error_type=ErrorCategory.SCHEMA,
                reason_code=exc.reason_code,
                schema_error=exc,
            )

        # try to coerce datatypes
        check_obj = self.coerce_dtype(
            check_obj,
            schema=schema,
            error_handler=error_handler,
        )

        return check_obj

    @validate_scope(scope=ValidationScope.DATA)
    def _data_checks(
        self,
        check_obj: DataFrame,
        schema,
        column_info: ColumnInfo,  # pylint: disable=unused-argument
        error_handler: ErrorHandler,
    ):
        """Run the checks related to data validation and uniqueness."""

        # uniqueness of values
        try:
            check_obj = self.unique(
                check_obj, schema=schema, error_handler=error_handler
            )
        except SchemaError as err:
            error_handler.collect_error(
                ErrorCategory.DATA, err.reason_code, err
            )

        return check_obj

    @cache_check_obj()
    def validate(
        self,
        check_obj: DataFrame,
        schema,
        *,
        head: Optional[int] = None,  # pylint: disable=unused-argument
        tail: Optional[int] = None,  # pylint: disable=unused-argument
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
        error_handler: ErrorHandler = None,
    ):
        """
        Parse and validate a check object, returning type-coerced and validated
        object.
        """
        assert (
            error_handler is not None
        ), "The `error_handler` argument must be provided."
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
            check_obj = check_obj.pandera.add_schema(schema)
        column_info = self.collect_column_info(check_obj, schema, lazy)

        # validate the columns (schema) of the dataframe
        check_obj = self._schema_checks(
            check_obj, schema, column_info, error_handler
        )

        # validate the rows (data) of the dataframe
        check_obj = self._data_checks(
            check_obj, schema, column_info, error_handler
        )

        # collect schema components and prepare check object to be validated
        schema_components = self.collect_schema_components(
            check_obj, schema, column_info
        )
        check_obj_subsample = self.subsample(
            check_obj, sample=sample, random_state=random_state
        )
        try:
            self.run_schema_component_checks(
                check_obj_subsample, schema_components, lazy, error_handler
            )
        except SchemaError as exc:
            error_handler.collect_error(
                error_type=ErrorCategory.SCHEMA,
                reason_code=exc.reason_code,
                schema_error=exc,
            )
        try:
            self.run_checks(check_obj_subsample, schema, error_handler)
        except SchemaError as exc:
            error_handler.collect_error(
                error_type=ErrorCategory.DATA,
                reason_code=exc.reason_code,
                schema_error=exc,
            )

        error_dicts = {}

        if error_handler.collected_errors:
            error_dicts = error_handler.summarize(schema_name=schema.name)

        check_obj.pandera.errors = error_dicts
        return check_obj

    def run_schema_component_checks(
        self,
        check_obj: DataFrame,
        schema_components: List,
        lazy: bool,
        error_handler: Optional[ErrorHandler],
    ):
        """Run checks for all schema components."""
        assert (
            error_handler is not None
        ), "The `error_handler` argument must be provided."
        check_results = []
        # schema-component-level checks
        for schema_component in schema_components:
            try:
                result = schema_component.validate(
                    check_obj=check_obj,
                    lazy=lazy,
                    inplace=True,
                    error_handler=error_handler,
                )
                check_results.append(is_table(result))
            except SchemaError as err:
                error_handler.collect_error(
                    ErrorCategory.SCHEMA,
                    err.reason_code,
                    err,
                )
        assert all(check_results)

    @validate_scope(scope=ValidationScope.DATA)
    def run_checks(self, check_obj: DataFrame, schema, error_handler):
        """Run a list of checks on the check object."""
        # dataframe-level checks
        check_results = []
        for check_index, check in enumerate(
            schema.checks
        ):  # schema.checks is null
            try:
                check_results.append(
                    self.run_check(check_obj, schema, check, check_index)
                )
            except SchemaError as err:
                error_handler.collect_error(
                    ErrorCategory.DATA, err.reason_code, err
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

                error_handler.collect_error(
                    ErrorCategory.DATA,
                    SchemaErrorReason.CHECK_ERROR,
                    SchemaError(
                        self,
                        check_obj,
                        msg,
                        failure_cases=scalar_failure_case(err_str),
                        check=check,
                        check_index=check_index,
                    ),
                    original_exc=err,
                )

    def collect_column_info(
        self,
        check_obj: DataFrame,
        schema,
        lazy: bool,
    ) -> ColumnInfo:
        """Collect column metadata."""
        column_names: List[Any] = []
        absent_column_names: List[Any] = []
        lazy_exclude_column_names: List[Any] = []

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
    @validate_scope(scope=ValidationScope.SCHEMA)
    def strict_filter_columns(
        self,
        check_obj: DataFrame,
        schema,
        column_info: ColumnInfo,
        error_handler: ErrorHandler,
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
                error_handler.collect_error(
                    ErrorCategory.SCHEMA,
                    SchemaErrorReason.COLUMN_NOT_IN_SCHEMA,
                    SchemaError(
                        schema=schema,
                        data=check_obj,
                        message=(
                            f"column '{column}' not in {schema.__class__.__name__}"
                            f" {schema.columns}"
                        ),
                        failure_cases=scalar_failure_case(column),
                        check="column_in_schema",
                        reason_code=SchemaErrorReason.COLUMN_NOT_IN_SCHEMA,
                    ),
                )
            if schema.strict == "filter" and not is_schema_col:
                filter_out_columns.append(column)
            if schema.ordered and is_schema_col:
                try:
                    next_ordered_col = next(sorted_column_names)
                except StopIteration:
                    pass
                if next_ordered_col != column:
                    error_handler.collect_error(
                        ErrorCategory.SCHEMA,
                        SchemaErrorReason.COLUMN_NOT_ORDERED,
                        SchemaError(
                            schema=schema,
                            data=check_obj,
                            message=f"column '{column}' out-of-order",
                            failure_cases=scalar_failure_case(column),
                            check="column_ordered",
                            reason_code=SchemaErrorReason.COLUMN_NOT_ORDERED,
                        ),
                    )

        if schema.strict == "filter":
            check_obj = check_obj.drop(*filter_out_columns)

        return check_obj

    @validate_scope(scope=ValidationScope.SCHEMA)
    def coerce_dtype(
        self,
        check_obj: DataFrame,
        *,
        schema=None,
        error_handler: ErrorHandler = None,
    ):
        """Coerces check object to the expected type."""
        assert schema is not None, "The `schema` argument must be provided."
        assert (
            error_handler is not None
        ), "The `error_handler` argument must be provided."

        if not (
            schema.coerce or any(col.coerce for col in schema.columns.values())
        ):
            return check_obj

        try:
            check_obj = self._coerce_dtype(check_obj, schema)

        except SchemaErrors as err:
            for schema_error_dict in err.schema_errors:
                if not error_handler.lazy:
                    # raise the first error immediately if not doing lazy validation
                    raise schema_error_dict["error"]
                error_handler.collect_error(
                    ErrorCategory.DTYPE_COERCION,
                    SchemaErrorReason.CHECK_ERROR,
                    schema_error_dict["error"],
                )
        except SchemaError as err:
            if not error_handler.lazy:
                raise err
            error_handler.collect_error(
                ErrorCategory.SCHEMA, err.reason_code, err
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
                    matched_columns = None

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
    def unique(
        self,
        check_obj: DataFrame,
        *,
        schema=None,
        error_handler: ErrorHandler = None,
    ):
        """Check uniqueness in the check object."""
        assert schema is not None, "The `schema` argument must be provided."
        assert (
            error_handler is not None
        ), "The `error_handler` argument must be provided."

        if not schema.unique:
            return check_obj

        # Determine unique columns based on schema's config
        unique_columns = (
            [schema.unique]
            if isinstance(schema.unique, str)
            else schema.unique
        )

        # Check if values belong to the dataframe columns
        missing_unique_columns = set(unique_columns) - set(check_obj.columns)
        if missing_unique_columns:
            raise SchemaDefinitionError(
                "Specified `unique` columns are missing in the dataframe: "
                f"{list(missing_unique_columns)}"
            )

        duplicates_count = (
            check_obj.select(*unique_columns)  # ignore other cols
            .groupby(*unique_columns)
            .agg(count("*").alias("pandera_duplicate_counts"))
            .filter(
                col("pandera_duplicate_counts") > 1
            )  # long name to avoid colisions
            .count()
        )

        if duplicates_count > 0:
            raise SchemaError(
                schema=schema,
                data=check_obj,
                message=(
                    f"Duplicated rows [{duplicates_count}] were found "
                    f"for columns {unique_columns}"
                ),
                check="unique",
                reason_code=SchemaErrorReason.DUPLICATES,
            )

        return check_obj

    def _check_uniqueness(
        self,
        obj: DataFrame,
        schema,
    ) -> DataFrame:
        """Ensure uniqueness in dataframe columns.

        :param obj: dataframe to check.
        :param schema: schema object.
        :returns: dataframe checked.
        """

    ##########
    # Checks #
    ##########

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_column_names_are_unique(self, check_obj: DataFrame, schema):
        """Check for column name uniquness."""
        if not schema.unique_column_names:
            return
        column_count_dict: Dict[Any, Any] = {}
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
            raise SchemaError(
                schema=schema,
                data=check_obj,
                message=(
                    f"dataframe contains multiple columns with label(s): {failed}"
                ),
                failure_cases=scalar_failure_case(failed),
                check="dataframe_column_labels_unique",
                reason_code=SchemaErrorReason.DUPLICATE_COLUMN_LABELS,
            )

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_column_presence(
        self, check_obj: DataFrame, schema, column_info: ColumnInfo
    ):
        """Check for presence of specified columns in the data object."""
        if column_info.absent_column_names:
            reason_code = SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME
            raise SchemaErrors(
                schema=schema,
                schema_errors=[  # type: ignore
                    {
                        "reason_code": reason_code,
                        "error": SchemaError(
                            schema=schema,
                            data=check_obj,
                            message=(
                                f"column '{colname}' not in dataframe"
                                f" {check_obj.head()}"
                            ),
                            failure_cases=scalar_failure_case(colname),
                            check="column_in_dataframe",
                            reason_code=reason_code,
                        ),
                    }
                    for colname in column_info.absent_column_names
                ],
                data=check_obj,
            )
