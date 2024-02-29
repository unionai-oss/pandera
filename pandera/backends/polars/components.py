"""Validation backend for polars components."""

from typing import Iterable, List, Optional, cast

import polars as pl

from pandera.api.polars.components import Column
from pandera.backends.base import CoreCheckResult
from pandera.backends.polars.base import PolarsSchemaBackend
from pandera.backends.polars.constants import FAILURE_CASE_KEY
from pandera.error_handlers import SchemaErrorHandler
from pandera.errors import (
    SchemaError,
    SchemaErrors,
    SchemaErrorReason,
)


class ColumnBackend(PolarsSchemaBackend):
    """Column backend for polars LazyFrames."""

    def preprocess(self, check_obj, inplace: bool = False):
        """Returns a copy of the object if inplace is False."""
        # NOTE: is this even necessary?
        return check_obj if inplace else check_obj.clone()

    # pylint: disable=too-many-locals
    def validate(
        self,
        check_obj: pl.LazyFrame,
        schema: Column,
        *,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> pl.LazyFrame:

        error_handler = SchemaErrorHandler(lazy)

        core_checks = [
            (self.check_dtype, (check_obj, schema)),
            (self.run_checks, (check_obj, schema)),
        ]

        for core_check, args in core_checks:
            results = core_check(*args)
            if isinstance(results, CoreCheckResult):
                results = [results]
            results = cast(List[CoreCheckResult], results)
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
                        result.reason_code,
                        error,
                        original_exc=result.original_exc,
                    )

        if lazy and error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.collected_errors,
                data=check_obj,
            )

        return check_obj

    def get_regex_columns(self, schema, columns) -> Iterable:
        raise NotImplementedError

    # NOTE: this will replace most of the code in the validate method above.
    def run_checks_and_handle_errors(
        self,
        error_handler,
        schema,
        check_obj: pl.LazyFrame,
        head,
        tail,
        sample,
        random_state,
    ):
        """Run checks on schema"""
        # pylint: disable=too-many-locals
        # NOTE: this method is common to pandas, so can be abstracted out to
        # the generic dataframe schema api spec.
        field_obj_subsample = self.subsample(
            check_obj[schema.name],
            head,
            tail,
            sample,
            random_state,
        )

        check_obj_subsample = self.subsample(
            check_obj,
            head,
            tail,
            sample,
            random_state,
        )

        core_checks = [
            (self.check_name, (field_obj_subsample, schema)),
            (self.check_nullable, (field_obj_subsample, schema)),
            (self.check_unique, (field_obj_subsample, schema)),
            (self.check_dtype, (field_obj_subsample, schema)),
            (self.run_checks, (check_obj_subsample, schema)),
        ]

        for core_check, args in core_checks:
            results = core_check(*args)
            if isinstance(results, CoreCheckResult):
                results = [results]
            results = cast(List[CoreCheckResult], results)
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
                        result.reason_code,
                        error,
                        original_exc=result.original_exc,
                    )

        return error_handler

    def coerce_dtype(
        self,
        check_obj: pl.LazyFrame,
        schema=None,
        # pylint: disable=unused-argument
    ) -> pl.LazyFrame:
        """Coerce type of a pd.Series by type specified in dtype.

        :param check_obj: LazyFrame to coerce
        :returns: coerced LazyFrame
        """
        assert schema is not None, "The `schema` argument must be provided."
        if schema.dtype is None or not schema.coerce:
            return check_obj

        try:
            return (
                check_obj.cast({schema.name: schema.dtype.type})
                .collect()
                .lazy()
            )
        except pl.exceptions.ComputeError as exc:
            raise SchemaError(
                schema=schema,
                data=check_obj,
                message=(
                    f"Error while coercing '{schema.name}' to type "
                    f"{schema.dtype}: {exc}"
                ),
                check=f"coerce_dtype('{schema.dtype}')",
            ) from exc

    def check_name(self, check_obj: pl.LazyFrame, schema) -> CoreCheckResult:
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

    def check_nullable(
        self, check_obj: pl.LazyFrame, schema
    ) -> CoreCheckResult:
        isna = check_obj.select([pl.col(schema.name).is_nan().alias("isna")])
        passed = (
            schema.nullable
            or isna.select([pl.col("isna").any()]).collect().item()
        )
        failure_cases = (
            check_obj.with_context(isna)
            .filter(pl.col("isna").not_())
            .rename({schema.name: FAILURE_CASE_KEY})
            .select(FAILURE_CASE_KEY)
            .collect()
        )
        return CoreCheckResult(
            passed=cast(bool, passed),
            check="not_nullable",
            reason_code=SchemaErrorReason.SERIES_CONTAINS_NULLS,
            message=(
                f"non-nullable series '{check_obj.name}' contains "
                f"null values"
            ),
            failure_cases=failure_cases,
        )

    def check_unique(self, check_obj, schema) -> CoreCheckResult:
        ...

    def check_dtype(
        self,
        check_obj: pl.LazyFrame,
        schema: Column,
    ) -> CoreCheckResult:

        passed = True
        failure_cases = None
        msg = None

        if schema.dtype is not None:
            obj_dtype = check_obj.schema[schema.name]
            passed = obj_dtype.is_(schema.dtype.type)

        if not passed:
            failure_cases = str(obj_dtype)
            msg = (
                f"expected column '{schema.name}' to have type "
                f"{schema.dtype}, got {obj_dtype}"
            )

        return CoreCheckResult(
            passed=passed,
            check=f"dtype('{schema.dtype}')",
            reason_code=SchemaErrorReason.WRONG_DATATYPE,
            message=msg,
            failure_cases=failure_cases,
        )

    # pylint: disable=unused-argument
    def run_checks(self, check_obj, schema) -> List[CoreCheckResult]:
        check_results: List[CoreCheckResult] = []
        for check_index, check in enumerate(schema.checks):
            check_args = [schema.name]  # pass in column key
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
            except Exception as err:  # pylint: disable=broad-except
                # catch other exceptions that may occur when executing the Check
                err_msg = f'"{err.args[0]}"' if len(err.args) > 0 else ""
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

    def set_default(self, check_obj, schema) -> pl.LazyFrame:
        """Set default values for missing columns."""
