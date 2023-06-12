"""Pandas Parsing, Validation, and Error Reporting Backends."""

import copy
import itertools
import traceback
from typing import Any, Callable, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel

from pandera.backends.base import CoreCheckResult
from pandera.api.pandas.types import is_table
from pandera.backends.pandas.base import ColumnInfo, PandasSchemaBackend
from pandera.backends.pandas.error_formatters import (
    reshape_failure_cases,
    scalar_failure_case,
)
from pandera.engines import pandas_engine
from pandera.backends.pandas.utils import convert_uniquesettings
from pandera.error_handlers import SchemaErrorHandler
from pandera.errors import (
    ParserError,
    SchemaDefinitionError,
    SchemaError,
    SchemaErrorReason,
    SchemaErrors,
)


class DataFrameSchemaBackend(PandasSchemaBackend):
    """Backend for pandas DataFrameSchema."""

    def preprocess(self, check_obj: pd.DataFrame, inplace: bool = False):
        """Preprocesses a check object before applying check functions."""
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
        """
        Parse and validate a check object, returning type-coerced and validated
        object.
        """
        # pylint: disable=too-many-locals
        if not is_table(check_obj):
            raise TypeError(f"expected pd.DataFrame, got {type(check_obj)}")

        error_handler = SchemaErrorHandler(lazy)

        check_obj = self.preprocess(check_obj, inplace=inplace)
        if hasattr(check_obj, "pandera"):
            check_obj = check_obj.pandera.add_schema(schema)

        column_info = self.collect_column_info(check_obj, schema, lazy)

        # collect schema components
        components = self.collect_schema_components(
            check_obj, schema, column_info
        )

        core_parsers: List[Tuple[Callable[..., Any], Tuple[Any, ...]]] = [
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

        # subsample the check object if head, tail, or sample are specified
        sample = self.subsample(check_obj, head, tail, sample, random_state)

        # check the container metadata, e.g. field names
        core_checks = [
            (self.check_column_names_are_unique, (check_obj, schema)),
            (self.check_column_presence, (check_obj, schema, column_info)),
            (self.check_column_values_are_unique, (sample, schema)),
            (self.run_schema_component_checks, (sample, components, lazy)),
            (self.run_checks, (sample, schema)),
        ]

        for check, args in core_checks:
            results = check(*args)  # type: ignore [operator]
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
        check_obj: pd.DataFrame,
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
                check_passed.append(is_table(result))
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

    def run_checks(
        self,
        check_obj: pd.DataFrame,
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
                        failure_cases=scalar_failure_case(err_str),
                        original_exc=err,
                    )
                )
        return check_results

    def collect_column_info(
        self,
        check_obj: pd.DataFrame,
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
                and col_name not in check_obj
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
        """Collects all schema components to use for validation."""

        columns = schema.columns
        try:
            is_pydantic = issubclass(
                pandas_engine.Engine.dtype(schema.dtype).type, BaseModel
            )
        except TypeError:
            is_pydantic = False

        if (
            not schema.columns
            and schema.dtype is not None
            # remove this hack when this backend has its own check dtype
            # function
            and not is_pydantic
        ):
            # NOTE: this is hack: the dataframe-level data type check should
            # be its own check function.
            # pylint: disable=import-outside-toplevel,cyclic-import
            from pandera.api.pandas.components import Column

            columns = {}
            for col in check_obj.columns:
                columns[col] = Column(schema.dtype, name=col)

        schema_components = []
        for col_name, col in columns.items():
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
            if type(check_obj).__module__.startswith("pyspark.pandas"):
                # NOTE: remove this when we have a seperate backend for pyspark
                # pandas.
                check_obj = check_obj.drop(labels=filter_out_columns, axis=1)
            else:
                check_obj.drop(labels=filter_out_columns, inplace=True, axis=1)

        return check_obj

    def coerce_dtype(
        self,
        check_obj: pd.DataFrame,
        schema=None,
    ):
        """Coerces check object to the expected type."""
        assert schema is not None, "The `schema` argument must be provided."

        error_handler = SchemaErrorHandler(lazy=True)

        if not (
            schema.coerce
            or (schema.index is not None and schema.index.coerce)
            or any(col.coerce for col in schema.columns.values())
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
        obj: pd.DataFrame,
        schema,
    ) -> pd.DataFrame:
        """Coerce dataframe to the type specified in dtype.

        :param obj: dataframe to coerce.
        :returns: dataframe with coerced dtypes
        """
        # NOTE: clean up the error handling!
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
                error_handler.collect_error(
                    SchemaErrorReason.DATATYPE_COERCION,
                    exc,
                )
                return obj

        for colname, col_schema in schema.columns.items():
            if col_schema.regex:
                try:
                    matched_columns = col_schema.get_backend(
                        obj
                    ).get_regex_columns(col_schema, obj.columns)
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
                index_schema.coerce = True
            coerced_index = _try_coercion(index_schema.coerce_dtype, obj.index)
            if coerced_index is not None:
                obj.index = coerced_index

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
        check_obj: pd.DataFrame,
        schema,
    ) -> CoreCheckResult:
        """Check for column name uniquness."""

        passed = True
        failure_cases = None
        message = None

        if not schema.unique_column_names:
            return CoreCheckResult(
                passed=passed,
                check="dataframe_column_labels_unique",
            )

        failed = check_obj.columns[check_obj.columns.duplicated()]
        if failed.any():
            passed = False
            message = (
                "dataframe contains multiple columns with label(s): "
                f"{failed.tolist()}"
            )
            failure_cases = scalar_failure_case(failed)

        return CoreCheckResult(
            passed=passed,
            check="dataframe_column_labels_unique",
            reason_code=SchemaErrorReason.DUPLICATE_COLUMN_LABELS,
            message=message,
            failure_cases=failure_cases,
        )

    # pylint: disable=unused-argument
    def check_column_presence(
        self, check_obj: pd.DataFrame, schema, column_info: ColumnInfo
    ) -> List[CoreCheckResult]:
        """Check for presence of specified columns in the data object."""
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
                            f"\n{check_obj.head()}"
                        ),
                        failure_cases=scalar_failure_case(colname),
                    )
                )
        return results

    def check_column_values_are_unique(
        self, check_obj: pd.DataFrame, schema
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

                passed = False
                message = f"columns '{*subset,}' not unique:\n{failure_cases}"
                failure_cases = reshape_failure_cases(failure_cases)
                break
        return CoreCheckResult(
            passed=passed,
            check="multiple_fields_uniqueness",
            reason_code=SchemaErrorReason.DUPLICATES,
            message=message,
            failure_cases=failure_cases,
        )
