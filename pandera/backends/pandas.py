"""Pandas Parsing, Validation, and Error Reporting Backends."""

import warnings

from functools import singledispatchmethod
from typing import Optional, Generic, TypeVar, Union

import pandas as pd

from pandera.backends.base import BaseSchemaBackend, BaseCheckBackend
from pandera.engines.pandas_engine import DataType, Engine
from pandera.error_formatters import (
    format_generic_error_message,
    format_vectorized_error_message,
    reshape_failure_cases,
    scalar_failure_case,
)
from pandera.error_handlers import SchemaErrorHandler
from pandera.errors import SchemaError, SchemaErrors



FieldCheckObj = Union[pd.Series, pd.DataFrame]

T = TypeVar(
    "T",
    pd.Series,
    pd.DataFrame,
    FieldCheckObj,
    covariant=True,
)


class PandasSchemaBackend(BaseSchemaBackend, Generic[T]):

    def __init__(self):
        pass

    def subsample(
        self,
        check_obj,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        pandas_obj_subsample = []
        if head is not None:
            pandas_obj_subsample.append(check_obj.head(head))
        if tail is not None:
            pandas_obj_subsample.append(check_obj.tail(tail))
        if sample is not None:
            pandas_obj_subsample.append(
                check_obj.sample(sample, random_state=random_state)
            )
        return (
            check_obj
            if not pandas_obj_subsample
            else pd.concat(pandas_obj_subsample).pipe(
                lambda x: x[~x.index.duplicated()]
            )
        )

    def run_check(
        self,
        schema,
        check_obj,
        check,
        check_index: int,
        *args,
    ) -> bool:
        """Handle check results, raising SchemaError on check failure.

        :param check_index: index of check in the schema component check list.
        :param check: Check object used to validate pandas object.
        :param check_args: arguments to pass into check object.
        :returns: True if check results pass or check.raise_warning=True, otherwise
            False.
        """
        check_result = check(check_obj, *args)
        if not check_result.check_passed:
            if check_result.failure_cases is None:
                # encode scalar False values explicitly
                failure_cases = scalar_failure_case(check_result.check_passed)
                error_msg = format_generic_error_message(
                    schema, check, check_index
                )
            else:
                failure_cases = reshape_failure_cases(
                    check_result.failure_cases, check.ignore_na
                )
                error_msg = format_vectorized_error_message(
                    schema, check, check_index, failure_cases
                )

            # raise a warning without exiting if the check is specified to do so
            if check.raise_warning:
                warnings.warn(error_msg, UserWarning)
                return True
            raise SchemaError(
                schema,
                check_obj,
                error_msg,
                failure_cases=failure_cases,
                check=check,
                check_index=check_index,
                check_output=check_result.check_output,
            )
        return check_result.check_passed


class PandasSchemaFieldBackend(PandasSchemaBackend[FieldCheckObj]):

    @singledispatchmethod
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
    ) -> pd.Series:
        return (check_obj if inplace else check_obj.copy())[name]

    @singledispatchmethod
    def validate(
        self,
        check_obj,
        schema,
        *,
        name: Optional[str] = None,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ):
        raise NotImplementedError

    @validate.register
    def validate_series(
        self,
        check_obj: pd.Series,
        schema,
        *,
        name: Optional[str] = None,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ):
        pass

    @validate.register
    def validate_dataframe(
        self,
        check_obj: pd.DataFrame,
        schema,
        *,
        name: Optional[str] = None,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ):
        error_handler = SchemaErrorHandler(lazy)
        field_obj = self.preprocess(check_obj, name, inplace)
        field_obj_subsample = self.subsample(
            field_obj, head, tail, sample, random_state
        )
        check_obj_subsample = self.subsample(
            check_obj, head, tail, sample, random_state
        )

        for core_check in (
            self.check_name,
            self.check_nullable,
            self.check_unique,
            self.check_dtype,
        ):
            core_check(field_obj_subsample, error_handler)

        # run all checks

    def check_name(
        self,
        check_obj: pd.Series,
        error_handler: SchemaErrorHandler,
        name: Optional[str] = None,
    ):
        if name is not None and check_obj.name == name:
            return

        error_handler.collect_error(
            reason_code="wrong_field_name",
            schema_error=SchemaError(
                schema=self.schema,
                data=check_obj,
                message=(
                    f"Expected {type(check_obj)} to have name {name}, "
                    f"found '{check_obj.name}'"
                ),
                failure_cases=scalar_failure_case(check_obj.name),
                check=f"field_name('{name}')"
            )
        )

    def check_nullable(
        self,
        check_obj: pd.Series,
        error_handler: SchemaErrorHandler,
        nullable: bool = False,
    ):
        if nullable:
            return

        isna = check_obj.isna()
        if isna.any():
            error_handler.collect_error(
                reason_code="series_contains_nulls",
                schema_error=SchemaError(
                    schema=self.schema,
                    data=check_obj,
                    message=(
                        f"non-nullable series '{check_obj.name}' contains "
                        f"null values:\n{check_obj[isna]}"
                    ),
                    failure_case=reshape_failure_cases(
                        check_obj[isna], ignore_na=False
                    ),
                    check="field_not_nullable",
                )
            )

    def check_unique(
        self,
        check_obj: pd.Series,
        error_handler: SchemaErrorHandler,
        unique: bool = False,
    ):
        if not unique:
            return
        
        if type(check_obj).__module__.startswith("pyspark.pandas"):
            # pylint: disable=import-outside-toplevel
            import pyspark.pandas as ps

            duplicates = check_obj.to_frame().duplicated().reindex(
                check_obj.index
            )
            with ps.option_context("compute.ops_on_diff_frames", True):
                failed = check_obj[duplicates]
        else:
            duplicates = check_obj.duplicated()
            failed = check_obj[duplicates]

        if duplicates.any():
            error_handler.collect_error(
                "series_contains_duplicates",
                SchemaError(
                    schema=self.schema,
                    data=check_obj,
                    message=(
                        f"series '{check_obj.name}' contains duplicate "
                        f"values:\n{failed}"
                    ),
                    failure_cases=reshape_failure_cases(failed),
                    check="field_uniqueness",
                ),
            )

    def check_dtype(
        self,
        check_obj: pd.Series,
        error_handler: SchemaErrorHandler,
        dtype: Optional[DataType],
    ):
        if dtype is not None and (
            not dtype.check(Engine.dtype(check_obj.dtype))
        ):
            error_handler.collect_error(
                "wrong_dtype",
                SchemaError(
                    schema=self.schema,
                    data=check_obj,
                    message=(
                        f"expected series '{check_obj.name}' to have type "
                        f"{dtype}, got {check_obj.dtype}"
                    ),
                    failure_cases=scalar_failure_case(str(check_obj.dtype)),
                    check=f"dtype('{self.dtype}')",
                ),
            )

    def run_all_checks(self):
        check_results = []
        if check_utils.is_field(check_obj):
            check_obj, check_args = series, [None]
        else:
            check_args = [self.name]  # type: ignore

        for check_index, check in enumerate(self.checks):
            try:
                check_results.append(
                    _handle_check_results(
                        self, check_index, check, check_obj, *check_args
                    )
                )
            except SchemaError as err:
                error_handler.collect_error("dataframe_check", err)
            except Exception as err:  # pylint: disable=broad-except
                # catch other exceptions that may occur when executing the
                # Check
                err_msg = f'"{err.args[0]}"' if len(err.args) > 0 else ""
                err_str = f"{err.__class__.__name__}({ err_msg})"
                msg = (
                    f"Error while executing check function: {err_str}\n"
                    + traceback.format_exc()
                )
                error_handler.collect_error(
                    "check_error",
                    errors.SchemaError(
                        self,
                        check_obj,
                        msg,
                        failure_cases=scalar_failure_case(err_str),
                        check=check,
                        check_index=check_index,
                    ),
                    original_exc=err,
                )

        if lazy and error_handler.collected_errors:
            raise errors.SchemaErrors(
                error_handler.collected_errors, check_obj
            )

        assert all(check_results)


class PandasSchemaContainerBackend(PandasSchemaBackend[pd.DataFrame]):

    def preprocess(self):
        pass

    def validate(self):
        pass

    def check_name(self):
        pass

    def check_nullable(self):
        pass

    def check_unique(self):
        pass

    def check_dtype(self):
        pass

    def run_all_checks(self):
        pass


class PandasCheckBackend(BaseCheckBackend):
    pass


class PandasCheckFieldBackend(PandasCheckBackend):
    pass


class PandasCheckContainerBackend(PandasCheckBackend):
    pass
