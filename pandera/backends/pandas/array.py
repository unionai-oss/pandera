import traceback
from functools import singledispatchmethod
from typing import Optional, Union, overload

import pandas as pd
from multimethod import DispatchError, multimethod

from pandera.backends.pandas.base import FieldCheckObj, PandasSchemaBackend
from pandera.engines.pandas_engine import Engine
from pandera.error_formatters import reshape_failure_cases, scalar_failure_case
from pandera.error_handlers import SchemaErrorHandler
from pandera.errors import ParserError, SchemaError, SchemaErrors


class ArraySchemaBackend(PandasSchemaBackend):
    @multimethod
    def preprocess(self, check_obj, name: str = None, inplace: bool = False):
        raise NotImplementedError

    @preprocess.register
    def preprocess_field(
        self, check_obj: pd.Series, name: str = None, inplace: bool = False
    ) -> pd.Series:
        return check_obj if inplace else check_obj.copy()

    @preprocess.register
    def preprocess_container(
        self, check_obj: pd.DataFrame, name: str = None, inplace: bool = False
    ) -> pd.DataFrame:
        return (check_obj if inplace else check_obj.copy())[name]

    def validate(
        self,
        check_obj,
        schema,
        *,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ):
        error_handler = SchemaErrorHandler(lazy)
        check_obj = self.preprocess(check_obj, schema.name, inplace)

        try:
            check_obj = self.coerce_dtype(
                check_obj, schema=schema, error_handler=error_handler
            )
        except SchemaError as exc:
            error_handler.collect_error(exc.reason_code, exc)

        check_obj_subsample = self.subsample(
            check_obj,
            head,
            tail,
            sample,
            random_state,
        )
        # run the core checks
        for core_check in (
            self.check_name,
            self.check_nullable,
            self.check_unique,
            self.check_dtype,
        ):
            try:
                core_check(check_obj_subsample, schema)
            except SchemaError as exc:
                error_handler.collect_error(exc.reason_code, exc)

        # run user-provided checks
        check_results = self.run_checks(
            check_obj_subsample, schema, error_handler, lazy
        )
        assert all(check_results)

        if lazy and error_handler.collected_errors:
            raise SchemaErrors(
                schema, error_handler.collected_errors, check_obj
            )
        return check_obj

    def coerce_dtype(
        self,
        check_obj,
        *,
        schema=None,
        error_handler: SchemaErrorHandler = None,
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
            ) from exc

    def check_name(self, check_obj: pd.Series, schema):
        if schema.name is None or check_obj.name == schema.name:
            return
        raise SchemaError(
            schema=schema,
            data=check_obj,
            message=(
                f"Expected {type(check_obj)} to have name {schema.name}, "
                f"found '{check_obj.name}'"
            ),
            failure_cases=scalar_failure_case(check_obj.name),
            check=f"field_name('{schema.name}')",
            reason_code="wrong_field_name",
        )

    def check_nullable(self, check_obj: pd.Series, schema):
        if schema.nullable:
            return
        isna = check_obj.isna()
        if isna.any():
            raise SchemaError(
                schema=schema,
                data=check_obj,
                message=(
                    f"non-nullable series '{check_obj.name}' contains "
                    f"null values:\n{check_obj[isna]}"
                ),
                failure_cases=reshape_failure_cases(
                    check_obj[isna], ignore_na=False
                ),
                check="not_nullable",
                reason_code="series_contains_nulls",
            )

    def check_unique(self, check_obj: pd.Series, schema):
        if not schema.unique:
            return

        if type(check_obj).__module__.startswith("pyspark.pandas"):
            # pylint: disable=import-outside-toplevel
            import pyspark.pandas as ps

            duplicates = (
                check_obj.to_frame().duplicated().reindex(check_obj.index)
            )
            with ps.option_context("compute.ops_on_diff_frames", True):
                failed = check_obj[duplicates]
        else:
            duplicates = check_obj.duplicated()
            failed = check_obj[duplicates]

        if duplicates.any():
            raise SchemaError(
                schema=schema,
                data=check_obj,
                message=(
                    f"series '{check_obj.name}' contains duplicate "
                    f"values:\n{failed}"
                ),
                failure_cases=reshape_failure_cases(failed),
                check="field_uniqueness",
                reason_code="series_contains_duplicates",
            )

    def check_dtype(self, check_obj: pd.Series, schema):
        if schema.dtype is not None and (
            not schema.dtype.check(Engine.dtype(check_obj.dtype))
        ):
            raise SchemaError(
                schema=schema,
                data=check_obj,
                message=(
                    f"expected series '{check_obj.name}' to have type "
                    f"{schema.dtype}, got {check_obj.dtype}"
                ),
                failure_cases=scalar_failure_case(str(check_obj.dtype)),
                check=f"dtype('{schema.dtype}')",
                reason_code="wrong_dtype",
            )

    def run_checks(self, check_obj, schema, error_handler, lazy):
        check_results = []
        for check_index, check in enumerate(schema.checks):
            try:
                check_results.append(
                    self.run_check(
                        check_obj, schema, check, check_index, schema.name
                    )
                )
            except SchemaError as err:
                error_handler.collect_error("dataframe_check", err)
            except Exception as err:  # pylint: disable=broad-except
                # catch other exceptions that may occur when executing the Check
                if isinstance(err, DispatchError):
                    # if the error was raised by a check registered via
                    # multimethod, get the underlying __cause__
                    err = err.__cause__
                err_msg = f'"{err.args[0]}"' if len(err.args) > 0 else ""
                err_str = f"{err.__class__.__name__}({ err_msg})"
                error_handler.collect_error(
                    "check_error",
                    SchemaError(
                        schema=schema,
                        data=check_obj,
                        message=(
                            f"Error while executing check function: {err_str}\n"
                            + traceback.format_exc()
                        ),
                        failure_cases=scalar_failure_case(err_str),
                        check=check,
                        check_index=check_index,
                    ),
                    original_exc=err,
                )
        return check_results
