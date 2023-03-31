"""Pyspark Parsing, Validation, and Error Reporting Backends."""

import copy
import itertools
import traceback
from typing import Any, List, Optional

from pandera.backends.pyspark.base import ColumnInfo, PysparkSchemaBackend
from pandera.backends.pandas.utils import convert_uniquesettings
from pandera.api.pyspark.types import is_table
from pandera.backends.pandas.error_formatters import (
    reshape_failure_cases,
    scalar_failure_case,
)
from pandera.error_handlers import SchemaErrorHandler
from pandera.errors import (
    ParserError,
    SchemaError,
    SchemaErrors,
    SchemaDefinitionError,
)
from pyspark.sql import DataFrame
from pyspark.sql.functions import cast, col


class DataFrameSchemaBackend(PysparkSchemaBackend):
    """Backend for pyspark DataFrameSchema."""

    def preprocess(self, check_obj: DataFrame, inplace: bool = False):
        """Preprocesses a check object before applying check functions."""
        return check_obj

    def validate(
        self,
        check_obj: DataFrame,
        schema,
        *,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ):
        """
        Parse and validate a check object, returning type-coerced and validated
        object.
        """
        # Todo To be done by Neeraj
        if not is_table(check_obj):
            raise TypeError(f"expected a pyspark DataFrame, got {type(check_obj)}")

        # Todo Error handling .. pending PS discussion
        error_handler = SchemaErrorHandler(lazy)

        check_obj = self.preprocess(check_obj, inplace=inplace)
        if hasattr(check_obj, "pandera"):
            check_obj = check_obj.pandera.add_schema(schema)

        column_info = self.collect_column_info(check_obj, schema, lazy)

        # check the container metadata, e.g. field names
        try:
            self.check_column_names_are_unique(check_obj, schema)
        except SchemaError as exc:
            error_handler.collect_error(exc.reason_code, exc)

        try:
            self.check_column_presence(check_obj, schema, column_info)
        except SchemaErrors as exc:
            for schema_error in exc.schema_errors:
                error_handler.collect_error(
                    schema_error["reason_code"],
                    schema_error["error"],
                )

        # strictness check and filter
        try:
            check_obj = self.strict_filter_columns(check_obj, schema, column_info)
        except SchemaError as exc:
            error_handler.collect_error(exc.reason_code, exc)
        # try to coerce datatypes
        check_obj = self.coerce_dtype(
            check_obj,
            schema=schema,
            error_handler=error_handler,
        )
        # collect schema components and prepare check object to be validated
        schema_components = self.collect_schema_components(
            check_obj, schema, column_info
        )
        check_obj_subsample = self.subsample(check_obj, sample, random_state)
        try:
            # TODO: need to create apply at column level
            self.run_schema_component_checks(
                check_obj_subsample, schema_components, lazy, error_handler
            )
        except SchemaError as exc:
            error_handler.collect_error(exc.reason_code, exc)
        breakpoint()
        try:
            self.run_checks(check_obj_subsample, schema, error_handler)
        except SchemaError as exc:
            error_handler.collect_error(exc.reason_code, exc)
        breakpoint()
        if error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.collected_errors,
                data=check_obj,
            )
        breakpoint()
        return check_obj

    def run_schema_component_checks(
        self,
        check_obj: DataFrame,
        schema_components: List,
        lazy: bool,
        error_handler: SchemaErrorHandler,
    ):
        """Run checks for all schema components."""
        check_results = []
        # schema-component-level checks
        for schema_component in schema_components:
            try:
                breakpoint()
                result = schema_component.validate(check_obj, lazy=lazy, inplace=True)
                check_results.append(is_table(result))
            except SchemaError as err:
                error_handler.collect_error("schema_component_check", err)
            except SchemaErrors as err:
                for schema_error_dict in err.schema_errors:
                    error_handler.collect_error(
                        "schema_component_check", schema_error_dict["error"]
                    )
        breakpoint()
        assert all(check_results)

    def run_checks(self, check_obj: DataFrame, schema, error_handler):
        """Run a list of checks on the check object."""
        # dataframe-level checks
        check_results = []
        breakpoint()
        for check_index, check in enumerate(schema.checks):  # schama.checks is null
            try:
                check_results.append(  # TODO: looping over cols
                    self.run_check(check_obj, schema, check, check_index)
                )
            except SchemaError as err:
                error_handler.collect_error("dataframe_check", err)
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
                    "check_error",
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
                        col_schema.BACKEND.get_regex_columns(
                            col_schema, check_obj.columns
                        )
                    )
                except SchemaError:
                    pass
            elif col_name in check_obj.columns:
                column_names.append(col_name)

        # drop adjacent duplicated column names

        destuttered_column_names = list(set(check_obj.columns))
        # if check_obj.columns != destuttered_column_names:
        #     destuttered_column_names = [
        #         k for k, _ in itertools.groupby(check_obj.columns)
        #     ]

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
        for col_name, col in schema.columns.items():
            if (
                col.required or col_name in check_obj
            ) and col_name not in column_info.lazy_exclude_column_names:
                col = copy.deepcopy(col)
                if schema.dtype is not None:
                    # override column dtype with dataframe dtype
                    col.dtype = schema.dtype

                # disable coercion at the schema component level since the
                # dataframe-level schema already coerced it.
                col.coerce = False
                schema_components.append(col)

        return schema_components

    ###########
    # Parsers #
    ###########

    def strict_filter_columns(
        self, check_obj: DataFrame, schema, column_info: ColumnInfo
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
                    reason_code="column_not_in_schema",
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
                        reason_code="column_not_ordered",
                    )

        if schema.strict == "filter":
            if type(check_obj).__module__.startswith("pyspark.pandas"):
                # NOTE: remove this when we have a seperate backend for pyspark
                # pandas.
                check_obj = check_obj.drop(labels=filter_out_columns, axis=1)
            else:
                check_obj.drop(labels=filter_out_columns, inplace=True, axis=1)

        return check_obj

    def coerce_dtype(
        self,
        check_obj: DataFrame,
        *,
        schema=None,
        error_handler: Optional[SchemaErrorHandler] = None,
    ):
        """Coerces check object to the expected type."""
        assert schema is not None, "The `schema` argument must be provided."

        _error_handler = error_handler or SchemaErrorHandler(lazy=True)

        if not (schema.coerce or any(col.coerce for col in schema.columns.values())):
            return check_obj

        try:
            check_obj = self._coerce_dtype(check_obj, schema)

        except SchemaErrors as err:
            for schema_error_dict in err.schema_errors:
                if not _error_handler.lazy:
                    # raise the first error immediately if not doing lazy
                    # validation
                    raise schema_error_dict["error"]
                _error_handler.collect_error(
                    "schema_component_check", schema_error_dict["error"]
                )
        except SchemaError as err:
            if not _error_handler.lazy:
                raise err
            _error_handler.collect_error("schema_component_check", err)

        if error_handler is None and _error_handler.collected_errors:
            # raise SchemaErrors if this method is called without an
            # error_handler
            raise SchemaErrors(
                schema=schema,
                schema_errors=_error_handler.collected_errors,
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
        :returns: dataframe with coerced dtypes
        """
        # NOTE: clean up the error handling!
        error_handler = SchemaErrorHandler(lazy=True)

        def _coerce_df_dtype(obj: DataFrame) -> DataFrame:
            if schema.dtype is None:
                raise ValueError(
                    "dtype argument is None. Must specify this argument "
                    "to coerce dtype"
                )

            try:
                return schema.dtype.try_coerce(obj)
            except ParserError as exc:
                raise SchemaError(
                    schema=schema,
                    data=obj,
                    message=(
                        f"Error while coercing '{schema.name}' to type "
                        f"{schema.dtype}: {exc}\n{exc.failure_cases}"
                    ),
                    failure_cases=exc.failure_cases,
                    check=f"coerce_dtype('{schema.dtype}')",
                ) from exc

        def _try_coercion(obj, colname, col_schema):
            try:
                schema = obj.pandera.schema
                breakpoint()
                obj = obj.withColumn(colname, col(colname).cast(col_schema.dtype.type))
                obj.pandera.add_schema(schema)
                return obj

            except SchemaError as exc:
                error_handler.collect_error("dtype_coercion_error", exc)
                return obj

        for colname, col_schema in schema.columns.items():

            if col_schema.regex:
                try:
                    matched_columns = col_schema.BACKEND.get_regex_columns(
                        col_schema, obj.columns
                    )
                except SchemaError:
                    matched_columns = None

                for matched_colname in matched_columns:
                    if col_schema.coerce or schema.coerce:
                        obj = _try_coercion(
                            obj,
                            matched_colname,
                            col_schema
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

        if schema.dtype is not None:
            obj = _try_coercion(_coerce_df_dtype, obj)

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

    def check_column_names_are_unique(self, check_obj: DataFrame, schema):
        """Check for column name uniquness."""
        if not schema.unique_column_names:
            return
        column_count_dict = {}
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
                    "dataframe contains multiple columns with label(s): " f"{failed}"
                ),
                # Todo change the scalar_failure_case
                failure_cases=scalar_failure_case(failed),
                check="dataframe_column_labels_unique",
                reason_code="duplicate_dataframe_column_labels",
            )

    def check_column_presence(
        self, check_obj: DataFrame, schema, column_info: ColumnInfo
    ):
        """Check for presence of specified columns in the data object."""
        if column_info.absent_column_names:
            reason_code = "column_not_in_dataframe"
            raise SchemaErrors(
                schema=schema,
                schema_errors=[
                    {
                        "reason_code": reason_code,
                        "error": SchemaError(
                            schema=schema,
                            data=check_obj,
                            message=(
                                f"column '{colname}' not in dataframe"
                                f"\n{check_obj.head()}"
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

    def check_column_values_are_unique(self, check_obj: DataFrame, schema):
        """Check that column values are unique."""
        if not schema.unique:
            return

        # NOTE: fix this pylint error
        # pylint: disable=not-an-iterable

        # Todo Not needed for spark discuss and remove convert_uniquesettings
        # keep_setting = convert_uniquesettings(schema.report_duplicates)
        temp_unique: List[List] = (
            [schema.unique]
            if all(isinstance(x, str) for x in schema.unique)
            else schema.unique
        )
        for lst in temp_unique:
            subset = [x for x in lst if x in check_obj]
            original_count = check_obj.count()
            drop_count = check_obj.drop_duplicates(  # type: ignore
                subset=subset  # type: ignore
            ).count()
            if drop_count != original_count:
                # NOTE: this is a hack to support pyspark.pandas, need to
                # figure out a workaround to error: "Cannot combine the
                # series or dataframe because it comes from a different
                # dataframe."
                # Todo How to handle the error
                # failure_cases = check_obj.loc[duplicates, subset]

                # failure_cases = reshape_failure_cases(failure_cases)
                failure_cases = None
                raise SchemaError(
                    schema=schema,
                    data=check_obj,
                    message=f"columns '{*subset,}' not unique:\n{failure_cases}",
                    failure_cases=failure_cases,
                    check="multiple_fields_uniqueness",
                    reason_code="duplicates",
                )