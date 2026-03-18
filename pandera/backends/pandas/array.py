"""Pandera array backends."""

from typing import Optional, cast

import pandas as pd

from pandera.api.base.error_handler import ErrorHandler, get_error_category
from pandera.api.pandas.types import is_field
from pandera.backends.base import CoreCheckResult, CoreParserResult
from pandera.backends.pandas.base import (
    PandasSchemaBackend,
    _parsed_column_values,
)
from pandera.backends.pandas.error_formatters import reshape_failure_cases
from pandera.backends.utils import convert_uniquesettings
from pandera.config import ValidationScope
from pandera.engines.pandas_engine import Engine
from pandera.errors import (
    ParserError,
    SchemaDefinitionError,
    SchemaError,
    SchemaErrorReason,
    SchemaErrors,
)
from pandera.validation_depth import validate_scope, validation_type


class ArraySchemaBackend(PandasSchemaBackend):
    """Backend for pandas arrays."""

    @staticmethod
    def _is_default_type_compatible(default_value, schema_dtype):
        """Check if the default value type is compatible with schema dtype.

        Returns True if the default value's type matches what the schema
        dtype expects (e.g., bool default for bool dtype).
        """
        if schema_dtype is None:
            return False

        # Map of schema dtype to compatible Python/numpy types
        dtype_str = str(schema_dtype).lower()

        if "bool" in dtype_str:
            return isinstance(default_value, bool)
        elif "int" in dtype_str:
            # int but not bool (bool is subclass of int in Python)
            return isinstance(default_value, int) and not isinstance(
                default_value, bool
            )
        elif "float" in dtype_str:
            return isinstance(default_value, (int, float)) and not isinstance(
                default_value, bool
            )
        elif "str" in dtype_str or dtype_str == "object":
            return isinstance(default_value, str)
        else:
            # For other types, don't try automatic conversion
            return False

    @staticmethod
    def _try_convert_dtype_after_fillna(series, schema_dtype, default_value):
        """Try to convert dtype after fillna for pandas >= 3.0 compatibility.

        In pandas >= 3.0, fillna doesn't automatically coerce the dtype
        when filling an all-null column. This function attempts to convert
        to the schema's dtype only if the default value type is compatible.
        """
        if (
            schema_dtype is None
            or not (
                pd.api.types.is_object_dtype(series.dtype)
                or pd.api.types.is_string_dtype(series.dtype)
            )
            or series.isna().any()
        ):
            return series

        # Only convert if the default value type is compatible
        if not ArraySchemaBackend._is_default_type_compatible(
            default_value, schema_dtype
        ):
            return series

        try:
            target_dtype = Engine.dtype(schema_dtype).type
            return series.astype(target_dtype)
        except (ValueError, TypeError):
            return series  # Conversion failed - keep original

    def preprocess(self, check_obj, inplace: bool = False):
        return check_obj if inplace else check_obj.copy()

    def validate(
        self,
        check_obj,
        schema,
        *,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ):
        error_handler = ErrorHandler(lazy)
        check_obj = self.preprocess(check_obj, inplace)

        if getattr(schema, "drop_invalid_rows", False) and not lazy:
            raise SchemaDefinitionError(
                "When drop_invalid_rows is True, lazy must be set to True."
            )

        # fill nans with `default` if it's present
        if hasattr(schema, "default") and schema.default is not None:
            check_obj = self.set_default(check_obj, schema)

        # run custom parsers
        check_obj = self.run_parsers(schema, check_obj)

        try:
            if is_field(check_obj) and schema.coerce:
                check_obj = self.coerce_dtype(check_obj, schema=schema)
            elif schema.coerce:
                check_obj[schema.name] = self.coerce_dtype(
                    check_obj[schema.name], schema=schema
                )
        except SchemaError as exc:
            error_handler.collect_error(
                get_error_category(exc.reason_code),
                exc.reason_code,
                exc,
            )

        # run the core checks
        error_handler = self.run_checks_and_handle_errors(
            error_handler,
            schema,
            check_obj,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
        )

        if lazy and error_handler.collected_errors:
            if getattr(schema, "drop_invalid_rows", False):
                check_obj = self.drop_invalid_rows(check_obj, error_handler)
            else:
                raise SchemaErrors(
                    schema=schema,
                    schema_errors=error_handler.schema_errors,
                    data=check_obj,
                )

        return check_obj

    def run_checks_and_handle_errors(
        self, error_handler, schema, check_obj, **subsample_kwargs
    ):
        """Run checks on schema"""

        field_obj_subsample = self.subsample(
            check_obj if is_field(check_obj) else check_obj[schema.name],
            **subsample_kwargs,
        )

        check_obj_subsample = self.subsample(check_obj, **subsample_kwargs)

        core_checks = [
            (self.check_name, (field_obj_subsample, schema)),
            (self.check_nullable, (field_obj_subsample, schema)),
            (self.check_unique, (field_obj_subsample, schema)),
            (self.check_dtype, (field_obj_subsample, schema)),
            (self.run_checks, (check_obj_subsample, schema)),
        ]

        for check, args in core_checks:
            results = check(*args)
            if isinstance(results, CoreCheckResult):
                results = [results]
            results = cast(list[CoreCheckResult], results)
            for result in results:
                if result.passed:
                    continue

                if result.schema_error is not None:
                    error = result.schema_error
                else:
                    error = SchemaError(
                        schema=schema,
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

        return error_handler

    def coerce_dtype(
        self,
        check_obj,
        schema=None,
    ):
        """Coerce type of a pd.Series by type specified in dtype.

        :param pd.Series series: One-dimensional ndarray with axis labels
            (including time series).
        :returns: ``Series`` with coerced data type
        """
        assert schema is not None, "The `schema` argument must be provided."
        if schema.dtype is None or not schema.coerce:
            return check_obj

        try:
            return schema.dtype.try_coerce(check_obj)
        except ParserError as exc:
            raise SchemaError(
                schema=schema,
                data=check_obj,
                message=(
                    f"Error while coercing '{schema.name}' to type "
                    f"{schema.dtype}: {exc}:\n{exc.failure_cases}"
                ),
                failure_cases=exc.failure_cases,
                check=f"coerce_dtype('{schema.dtype}')",
                reason_code=SchemaErrorReason.DATATYPE_COERCION,
            ) from exc

    def run_parsers(self, schema, check_obj):
        parser_results: list[CoreParserResult] = []
        for parser_index, parser in enumerate(schema.parsers):
            parser_args = [None] if is_field(check_obj) else [schema.name]
            result = self.run_parser(
                check_obj,
                parser,
                parser_index,
                *parser_args,
            )
            if is_field(check_obj):
                check_obj = result.parser_output
            else:
                # Column parsed in DataFrame context: update column in place
                # so the full DataFrame is preserved (fixes custom parser +
                # drop_invalid_rows returning None in parsed column).
                check_obj[schema.name] = result.parser_output
                # Store for drop_invalid_rows to restore if column is overwritten
                parsed = _parsed_column_values.get()
                if parsed is None:
                    parsed = {}
                    _parsed_column_values.set(parsed)
                parsed[schema.name] = result.parser_output
            parser_results.append(result)
        return check_obj

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_name(self, check_obj: pd.Series, schema) -> CoreCheckResult:
        return CoreCheckResult(
            passed=schema.name is None or check_obj.name == schema.name,
            check=f"field_name('{schema.name}')",
            reason_code=SchemaErrorReason.WRONG_FIELD_NAME,
            message=(
                f"Expected {type(check_obj)} to have name '{schema.name}', "
                f"found '{check_obj.name}'"
            ),
            failure_cases=check_obj.name,
        )

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_nullable(self, check_obj: pd.Series, schema) -> CoreCheckResult:
        if schema.nullable or not check_obj.hasnans:
            # Avoid to compute anything for perf reasons. GH#1533
            return CoreCheckResult(
                passed=True,
                check="not_nullable",
            )

        isna = check_obj.isna()
        passed = schema.nullable or not isna.any()
        failure_cases = (
            reshape_failure_cases(check_obj[isna], ignore_na=False)
            if not passed
            else None
        )
        return CoreCheckResult(
            passed=cast(bool, passed),
            check="not_nullable",
            reason_code=SchemaErrorReason.SERIES_CONTAINS_NULLS,
            message=(
                f"non-nullable series '{check_obj.name}' contains "
                f"null values:\n{check_obj[isna]}"
            ),
            failure_cases=failure_cases,
        )

    @validate_scope(scope=ValidationScope.DATA)
    def check_unique(self, check_obj: pd.Series, schema) -> CoreCheckResult:
        passed = True
        failure_cases = None
        message = None

        if schema.unique:
            keep_argument = convert_uniquesettings(schema.report_duplicates)
            duplicates = None
            failed = None

            if type(check_obj).__module__.startswith("pyspark.pandas"):
                import pyspark.pandas as ps

                duplicates = (
                    check_obj.to_frame()  # type: ignore
                    .duplicated(keep=keep_argument)  # type: ignore
                    .reindex(check_obj.index)
                )
                with ps.option_context("compute.ops_on_diff_frames", True):
                    failed = check_obj[duplicates]
            else:
                if not check_obj.is_unique:
                    duplicates = check_obj.duplicated(keep=keep_argument)  # type: ignore
                    failed = check_obj[duplicates]

            if (
                duplicates is not None
                and failed is not None
                and duplicates.any()
            ):
                passed = False
                failure_cases = reshape_failure_cases(failed)
                message = (
                    f"series '{check_obj.name}' contains duplicate "
                    f"values:\n{failed}"
                )

        return CoreCheckResult(
            passed=passed,
            check="field_uniqueness",
            reason_code=SchemaErrorReason.SERIES_CONTAINS_DUPLICATES,
            message=message,
            failure_cases=failure_cases,
        )

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_dtype(self, check_obj: pd.Series, schema) -> CoreCheckResult:
        passed = True
        failure_cases = None
        msg = None

        if schema.dtype is not None:
            dtype_check_results = schema.dtype.check(
                Engine.dtype(check_obj.dtype),
                check_obj,
            )
            if isinstance(dtype_check_results, bool):
                passed = dtype_check_results
                # When whole-series dtype check fails, get element-level failure
                # cases so drop_invalid_rows and error reporting show actual values.
                if not passed and schema.dtype is not None:
                    try:
                        from pandera.engines import utils as engine_utils

                        failure_cases = (
                            engine_utils.numpy_pandas_coerce_failure_cases(
                                check_obj, schema.dtype
                            )
                        )
                    except Exception:
                        failure_cases = None
                    if failure_cases is None or (
                        hasattr(failure_cases, "empty") and failure_cases.empty
                    ):
                        # For str/string dtype, coercion can succeed (e.g. 3 -> "3");
                        # report elements that are not already strings as failure cases.
                        try:
                            expected_dtype = Engine.dtype(schema.dtype)
                            if str(expected_dtype) in (
                                "str",
                                "string",
                                "string[pyarrow]",
                            ):
                                non_string = check_obj[
                                    ~check_obj.map(
                                        lambda x: (
                                            isinstance(x, str) or pd.isna(x)
                                        )
                                    )
                                ]
                                if not non_string.empty:
                                    failure_cases = reshape_failure_cases(
                                        non_string, ignore_na=False
                                    )
                        except Exception:
                            pass
                        if failure_cases is None or (
                            hasattr(failure_cases, "empty")
                            and failure_cases.empty
                        ):
                            failure_cases = str(check_obj.dtype)
                elif not passed:
                    failure_cases = str(check_obj.dtype)
                msg = (
                    f"expected series '{check_obj.name}' to have type "
                    f"{schema.dtype}, got {check_obj.dtype}"
                )
            else:
                passed = dtype_check_results.all()
                failure_cases = reshape_failure_cases(
                    check_obj[~dtype_check_results.astype(bool)],
                    ignore_na=False,
                )
                msg = (
                    f"expected series '{check_obj.name}' to have type "
                    f"{schema.dtype}:\nfailure cases:\n{failure_cases}"
                )

        return CoreCheckResult(
            passed=passed,
            check=f"dtype('{schema.dtype}')",
            reason_code=SchemaErrorReason.WRONG_DATATYPE,
            message=msg,
            failure_cases=failure_cases,
        )

    @validate_scope(scope=ValidationScope.DATA)
    def run_checks(self, check_obj, schema) -> list[CoreCheckResult]:
        check_results: list[CoreCheckResult] = []
        for check_index, check in enumerate(schema.checks):
            check_args = [None] if is_field(check_obj) else [schema.name]
            try:
                check_results.append(
                    self.run_check(
                        check_obj,
                        schema,
                        check,
                        check_index,
                        *check_args,
                    )
                )
            except Exception as err:
                # catch other exceptions that may occur when executing the Check
                err_msg = f'"{err.args[0]}"' if err.args else ""
                msg = f"{err.__class__.__name__}({err_msg})"
                check_results.append(
                    CoreCheckResult(
                        passed=False,
                        check=check,
                        check_index=check_index,
                        reason_code=SchemaErrorReason.CHECK_ERROR,
                        message=msg,
                        failure_cases=msg,
                        original_exc=err,
                    )
                )
        return check_results

    def set_default(self, check_obj, schema):
        """Sets the ``schema.default`` value on the ``check_obj``"""
        # Ignore sparse dtype as it can't assign default value directly
        if is_field(check_obj) and not isinstance(
            check_obj.dtype, pd.SparseDtype
        ):
            check_obj = check_obj.fillna(schema.default)
            check_obj = self._try_convert_dtype_after_fillna(
                check_obj, schema.dtype, schema.default
            )
        elif not is_field(check_obj) and not isinstance(
            check_obj[schema.name].dtype, pd.SparseDtype
        ):
            check_obj[schema.name] = check_obj[schema.name].fillna(
                schema.default
            )
            check_obj[schema.name] = self._try_convert_dtype_after_fillna(
                check_obj[schema.name], schema.dtype, schema.default
            )

        return check_obj


class SeriesSchemaBackend(ArraySchemaBackend):
    """Backend for pandas Series objects."""

    def coerce_dtype(
        self,
        check_obj,
        schema=None,
    ):
        if hasattr(check_obj, "pandera"):
            check_obj = check_obj.pandera.add_schema(schema)

        check_obj = super().coerce_dtype(check_obj, schema=schema)

        if hasattr(check_obj, "pandera"):
            check_obj = check_obj.pandera.add_schema(schema)
        return check_obj
