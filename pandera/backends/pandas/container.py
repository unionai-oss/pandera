"""Pandas Parsing, Validation, and Error Reporting Backends."""

import copy
import itertools
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel

from pandera.api.base.error_handler import ErrorHandler
from pandera.api.pandas.types import is_table
from pandera.backends.base import ColumnInfo, CoreCheckResult, CoreParserResult
from pandera.backends.pandas.base import PandasSchemaBackend
from pandera.backends.pandas.error_formatters import (
    reshape_failure_cases,
    scalar_failure_case,
)
from pandera.backends.utils import convert_uniquesettings
from pandera.config import ValidationScope
from pandera.engines import pandas_engine
from pandera.errors import (
    ParserError,
    SchemaDefinitionError,
    SchemaError,
    SchemaErrorReason,
    SchemaErrors,
)
from pandera.validation_depth import validate_scope, validation_type


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

        if getattr(schema, "drop_invalid_rows", False) and not lazy:
            raise SchemaDefinitionError(
                "When drop_invalid_rows is True, lazy must be set to True."
            )

        error_handler = ErrorHandler(lazy)

        check_obj = self.preprocess(check_obj, inplace=inplace)
        if hasattr(check_obj, "pandera"):
            check_obj = check_obj.pandera.add_schema(schema)

        # Collect status of columns against schema
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
                error_handler.collect_error(
                    validation_type(exc.reason_code), exc.reason_code, exc
                )
            except SchemaErrors as exc:
                error_handler.collect_errors(exc.schema_errors)

        # run custom parsers
        check_obj = self.run_parsers(schema, check_obj)

        # We may have modified columns, for example by
        # add_missing_columns, so regenerate column info
        column_info = self.collect_column_info(check_obj, schema)

        # collect schema components
        components = self.collect_schema_components(
            check_obj, schema, column_info
        )

        # run the checks
        error_handler = self.run_checks_and_handle_errors(
            error_handler,
            schema,
            check_obj,
            column_info,
            sample,
            components,
            lazy,
            head,
            tail,
            random_state,
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

    def run_checks_and_handle_errors(
        self,
        error_handler,
        schema,
        check_obj,
        column_info,
        sample,
        components,
        lazy,
        head,
        tail,
        random_state,
    ):
        """Run checks on schema"""
        # pylint: disable=too-many-locals

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
                    validation_type(result.reason_code),
                    result.reason_code,
                    error,
                    result.original_exc,
                )

        return error_handler

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
    ) -> ColumnInfo:
        """Collect column metadata."""
        column_names: List[Any] = []
        absent_column_names: List[Any] = []
        regex_match_patterns: List[Any] = []

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
            regex_match_patterns=regex_match_patterns,
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

        if schema.index is not None:
            schema_components.append(schema.index)
        return schema_components

    ###########
    # Parsers #
    ###########

    def add_missing_columns(
        self, check_obj: pd.DataFrame, schema, column_info: ColumnInfo
    ):
        """Add columns that aren't in the dataframe."""
        # Add missing columns to dataframe based on 'add_missing_columns'
        # schema property

        if not (
            column_info.absent_column_names and schema.add_missing_columns
        ):
            return check_obj

        # Absent columns are required to have a default
        # value or be nullable
        for col_name in column_info.absent_column_names:
            col_schema = schema.columns[col_name]
            if pd.isna(col_schema.default) and not col_schema.nullable:
                raise SchemaError(
                    schema=schema,
                    data=check_obj,
                    message=(
                        f"column '{col_name}' in {schema.__class__.__name__}"
                        f" {schema.columns} requires a default value "
                        f"when non-nullable add_missing_columns is enabled"
                    ),
                    failure_cases=scalar_failure_case(col_name),
                    check="add_missing_has_default",
                    reason_code=SchemaErrorReason.ADD_MISSING_COLUMN_NO_DEFAULT,
                )

        # Ascertain order in which missing columns should be inserted into
        # dataframe. Be careful not to modify order of existing dataframe
        # columns to avoid ripple effects in downstream validation
        # (e.g., ordered schema).
        schema_cols_dict: Dict[Any, None] = {}
        for col_name, col_schema in schema.columns.items():
            if col_name in check_obj.columns or col_schema.required:
                schema_cols_dict[col_name] = None

        concat_ordered_cols = []
        for col_name in check_obj.columns:
            pop_cols = []
            for next_col_name in iter(schema_cols_dict):
                if (
                    next_col_name in column_info.absent_column_names
                    and next_col_name not in concat_ordered_cols
                ):
                    # Next schema column is missing from dataframe,
                    # so mark for insertion here
                    concat_ordered_cols.append(next_col_name)
                    pop_cols.append(next_col_name)
                else:
                    # Pop marked columns from schema list
                    for pop_col in pop_cols:
                        schema_cols_dict.pop(pop_col)
                    break

            # Add current column
            concat_ordered_cols.append(col_name)

            # Pop current column if it exists in schema
            schema_cols_dict.pop(col_name, None)

        # Add any remaining absent columns
        for col_name in column_info.absent_column_names:
            if col_name not in concat_ordered_cols:
                concat_ordered_cols.append(col_name)

        # Create companion dataframe of default values for missing columns
        missing_cols_schema = {
            k: v
            for k, v in schema.columns.items()
            if k in column_info.absent_column_names
        }
        missing_obj = self._construct_missing_df(
            check_obj, missing_cols_schema
        )

        # Append missing columns
        concat_obj = pd.concat([check_obj, missing_obj], axis=1)

        # Set column order
        concat_obj = concat_obj[concat_ordered_cols]

        return concat_obj

    def _construct_missing_df(
        self,
        obj: pd.DataFrame,
        missing_cols_schema: Dict[str, Any],
    ) -> pd.DataFrame:
        """Construct dataframe of missing columns with their default values.

        :param obj: dataframe of master dataframe from which to take index.
        :param missing_cols_schema: dictionary of Column schemas
        :returns: dataframe of missing columns
        """
        missing_obj = pd.DataFrame(
            data={k: v.default for k, v in missing_cols_schema.items()},
            index=obj.index,
        )

        # Can't specify multiple dtypes in frame construction and
        # constructing the frame as a concatenation of indexed
        # series is relatively slow due to copying the index for
        # each one. Coerce dtypes afterwards instead.
        for c in missing_obj.columns:
            missing_obj[c] = missing_cols_schema[c].dtype.try_coerce(
                missing_obj[c]
            )

        return missing_obj

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

        error_handler = ErrorHandler(lazy=True)

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
        obj: pd.DataFrame,
        schema,
    ) -> pd.DataFrame:
        """Coerce dataframe to the type specified in dtype.

        :param obj: dataframe to coerce.
        :returns: dataframe with coerced dtypes
        """
        # NOTE: clean up the error handling!
        error_handler = ErrorHandler(lazy=True)

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
                    reason_code=SchemaErrorReason.DATATYPE_COERCION,
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
                    validation_type(SchemaErrorReason.DATATYPE_COERCION),
                    SchemaErrorReason.DATATYPE_COERCION,
                    exc,
                )
                return obj

        for colname, col_schema in schema.columns.items():
            if col_schema.regex:
                try:
                    matched_columns = col_schema.get_backend(
                        obj
                    ).get_regex_columns(col_schema, obj)
                except SchemaError:
                    matched_columns = pd.Index([])

                for matched_colname in matched_columns:
                    if (
                        col_schema.coerce or schema.coerce
                    ) and schema.dtype is None:
                        _col_schema = copy.deepcopy(col_schema)
                        _col_schema.coerce = True
                        obj[matched_colname] = _try_coercion(
                            _col_schema.coerce_dtype, obj[matched_colname]
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
                schema_errors=error_handler.schema_errors,
                data=obj,
            )

        return obj

    def run_parsers(self, schema, check_obj):
        """Run parsers"""
        parser_results: List[CoreParserResult] = []
        for parser_index, parser in enumerate(schema.parsers):
            result = self.run_parser(
                check_obj,
                parser,
                parser_index,
            )
            check_obj = result.parser_output
            parser_results.append(result)
        return check_obj

    ##########
    # Checks #
    ##########

    @validate_scope(scope=ValidationScope.SCHEMA)
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

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_column_presence(
        self, check_obj: pd.DataFrame, schema, column_info: ColumnInfo
    ) -> List[CoreCheckResult]:
        """Check for presence of specified columns in the data object."""
        results = []
        if column_info.absent_column_names and not schema.add_missing_columns:
            for colname in column_info.absent_column_names:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="column_in_dataframe",
                        reason_code=SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
                        message=(
                            f"column '{colname}' not in dataframe. "
                            f"Columns in dataframe: {check_obj.columns.tolist()}"
                        ),
                        failure_cases=scalar_failure_case(colname),
                    )
                )
        return results

    @validate_scope(scope=ValidationScope.DATA)
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
