"""Pandas Parsing, Validation, and Error Reporting Backends."""

import copy
import itertools
import traceback
from typing import Any, List, Optional, Union

import pandas as pd

from pandera.backends.pandas.base import ColumnInfo, PandasSchemaBackend
from pandera.backends.pandas.utils import convert_uniquesettings
from pandera.core.pandas.types import is_table
from pandera.error_formatters import reshape_failure_cases, scalar_failure_case
from pandera.error_handlers import SchemaErrorHandler
from pandera.errors import (
    ParserError,
    SchemaError,
    SchemaErrors,
    SchemaDefinitionError,
)


class DataFrameSchemaBackend(PandasSchemaBackend):
    def preprocess(self, check_obj: pd.DataFrame, inplace: bool = False):
        if not inplace:
            check_obj = check_obj.copy()
        return check_obj

    def validate(
        self,
        check_obj: pd.DataFrame,
        schema,
        *,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ):
        if not is_table(check_obj):
            raise TypeError(f"expected pd.DataFrame, got {type(check_obj)}")

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
        except SchemaError as exc:
            error_handler.collect_error(exc.reason_code, exc)

        # strictness check and filter
        try:
            check_obj = self.strict_filter_columns(
                check_obj, schema, column_info
            )
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
        check_obj_subsample = self.subsample(
            check_obj, head, tail, sample, random_state
        )
        try:
            self.run_schema_component_checks(
                check_obj_subsample, schema_components, lazy, error_handler
            )
        except SchemaError as exc:
            error_handler.collect_error(exc.reason_code, exc)

        try:
            self.run_checks(check_obj_subsample, schema, error_handler)
        except SchemaError as exc:
            error_handler.collect_error(exc.reason_code, exc)

        try:
            self.check_column_values_are_unique(check_obj_subsample, schema)
        except SchemaError as exc:
            error_handler.collect_error(exc.reason_code, exc)

        if error_handler.collected_errors:
            raise SchemaErrors(
                schema,
                error_handler.collected_errors,
                check_obj,
            )

        return check_obj

    def run_schema_component_checks(
        self,
        check_obj: pd.DataFrame,
        schema_components: List,
        lazy: bool,
        error_handler: SchemaErrorHandler,
    ):
        check_results = []
        # schema-component-level checks
        for schema_component in schema_components:
            try:
                result = schema_component.validate(
                    check_obj, lazy=lazy, inplace=True
                )
                check_results.append(is_table(result))
            except SchemaError as err:
                error_handler.collect_error("schema_component_check", err)
            except SchemaErrors as err:
                for schema_error_dict in err.schema_errors:
                    error_handler.collect_error(
                        "schema_component_check", schema_error_dict["error"]
                    )
        assert all(check_results)

    def run_checks(self, check_obj: pd.DataFrame, schema, error_handler):
        # dataframe-level checks
        check_results = []
        for check_index, check in enumerate(schema.checks):
            try:
                check_results.append(
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
        check_obj: pd.DataFrame,
        schema,
        lazy: bool,
    ) -> ColumnInfo:
        column_names: List[Any] = []
        absent_column_names: List[Any] = []
        lazy_exclude_column_names: List[Any] = []

        for col_name, col_schema in schema.columns.items():
            if (
                not col_schema.regex
                and col_name not in check_obj
                and col_schema.required
            ):
                absent_column_names.append(col_name)
                if lazy:
                    # TODO: remove this since we can just use
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
        destuttered_column_names = [*check_obj.columns]
        if check_obj.columns.has_duplicates:
            destuttered_column_names = [
                k for k, _ in itertools.groupby(check_obj.columns)
            ]

        return ColumnInfo(
            sorted_column_names=dict.fromkeys(column_names),
            expanded_column_names=frozenset(column_names),
            destuttered_column_names=destuttered_column_names,
            absent_column_names=absent_column_names,
            lazy_exclude_column_names=lazy_exclude_column_names,
        )

    def collect_schema_components(
        self,
        check_obj: pd.DataFrame,
        schema,
        column_info: ColumnInfo,
    ):
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

        if schema.index is not None:
            schema_components.append(schema.index)
        return schema_components

    ###########
    # Parsers #
    ###########

    def strict_filter_columns(
        self, check_obj: pd.DataFrame, schema, column_info: ColumnInfo
    ):
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
            check_obj.drop(labels=filter_out_columns, inplace=True, axis=1)

        return check_obj

    def coerce_dtype(
        self,
        check_obj: pd.DataFrame,
        *,
        schema=None,
        error_handler: Optional[SchemaErrorHandler] = None,
    ):
        assert schema is not None, "The `schema` argument must be provided."

        _error_handler = error_handler or SchemaErrorHandler(lazy=True)

        if not (
            schema.coerce
            or (schema.index is not None and schema.index.coerce)
            or any(col.coerce for col in schema.columns.values())
        ):
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

        if error_handler is None:
            # raise SchemaErrors if this method is called without an
            # error_handler
            raise SchemaErrors(
                schema, _error_handler.collected_errors, check_obj
            )

        return check_obj

    def _coerce_dtype(
        self,
        obj: pd.DataFrame,
        schema,
    ) -> pd.DataFrame:
        """Coerce dataframe to the type specified in dtype.

        :param obj: dataframe to coerce.
        :returns: dataframe with coerced dtypes
        """
        # TODO: clean up the error handling!
        error_handler = SchemaErrorHandler(lazy=True)

        def _coerce_df_dtype(obj: pd.DataFrame) -> pd.DataFrame:
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

        def _try_coercion(coerce_fn, obj):
            try:
                return coerce_fn(obj)
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
                    matched_columns = pd.Index([])

                for matched_colname in matched_columns:
                    if col_schema.coerce or schema.coerce:
                        obj[matched_colname] = _try_coercion(
                            col_schema.coerce_dtype, obj[matched_colname]
                        )
            elif (
                (col_schema.coerce or schema.coerce)
                and schema.dtype is None
                and colname in obj
            ):
                _col_schema = copy.deepcopy(col_schema)
                _col_schema.coerce = True
                obj[colname] = _try_coercion(
                    _col_schema.coerce_dtype, obj[colname]
                )

        if schema.dtype is not None:
            obj = _try_coercion(_coerce_df_dtype, obj)
        if schema.index is not None and (schema.index.coerce or schema.coerce):
            index_schema = copy.deepcopy(schema.index)
            if schema.coerce:
                # coercing at the dataframe-level should apply index coercion
                # for both single- and multi-indexes.
                index_schema._coerce = True
            coerced_index = _try_coercion(index_schema.coerce_dtype, obj.index)
            if coerced_index is not None:
                obj.index = coerced_index

        if error_handler.collected_errors:
            raise SchemaErrors(schema, error_handler.collected_errors, obj)

        return obj

    ##########
    # Checks #
    ##########

    def check_column_names_are_unique(self, check_obj: pd.DataFrame, schema):
        if not schema.unique_column_names:
            return
        failed = check_obj.columns[check_obj.columns.duplicated()]
        if failed.any():
            raise SchemaError(
                schema=schema,
                data=check_obj,
                message=(
                    "dataframe contains multiple columns with label(s): "
                    f"{failed.tolist()}"
                ),
                failure_cases=scalar_failure_case(failed),
                check="dataframe_column_labels_unique",
                reason_code="duplicate_dataframe_column_labels",
            )

    def check_column_presence(
        self, check_obj: pd.DataFrame, schema, column_info: ColumnInfo
    ):
        if column_info.absent_column_names:
            # TODO: only report the first absent column for now, need to update
            # this when backend stuff is complete
            colname, *_ = column_info.absent_column_names
            raise SchemaError(
                schema=schema,
                data=check_obj,
                message=(
                    f"column '{colname}' not in dataframe\n{check_obj.head()}"
                ),
                failure_cases=scalar_failure_case(colname),
                check="column_in_dataframe",
                reason_code="column_not_in_dataframe",
            )

    def check_column_values_are_unique(self, check_obj: pd.DataFrame, schema):
        if not schema.unique:
            return

        # NOTE: fix this pylint error
        # pylint: disable=not-an-iterable
        keep_setting = convert_uniquesettings(schema.report_duplicates)
        temp_unique: List[List] = (
            [schema.unique]
            if all(isinstance(x, str) for x in schema.unique)
            else schema.unique
        )
        for lst in temp_unique:
            subset = [x for x in lst if x in check_obj]
            duplicates = check_obj.duplicated(  # type: ignore
                subset=subset, keep=keep_setting  # type: ignore
            )
            if duplicates.any():
                # NOTE: this is a hack to support pyspark.pandas, need to
                # figure out a workaround to error: "Cannot combine the
                # series or dataframe because it comes from a different
                # dataframe."
                if type(duplicates).__module__.startswith("pyspark.pandas"):
                    # pylint: disable=import-outside-toplevel
                    import pyspark.pandas as ps

                    with ps.option_context("compute.ops_on_diff_frames", True):
                        failure_cases = check_obj.loc[duplicates, subset]
                else:
                    failure_cases = check_obj.loc[duplicates, subset]

                failure_cases = reshape_failure_cases(failure_cases)
                raise SchemaError(
                    schema=schema,
                    data=check_obj,
                    message=f"columns '{*subset,}' not unique:\n{failure_cases}",
                    failure_cases=failure_cases,
                    check="multiple_fields_uniqueness",
                    reason_code="duplicates",
                )
