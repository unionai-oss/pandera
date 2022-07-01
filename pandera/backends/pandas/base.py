"""Pandas Parsing, Validation, and Error Reporting Backends."""

import copy
import warnings
from typing import (
    FrozenSet,
    Generic,
    Iterable,
    List,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
)

import pandas as pd

from pandera.backends.base import BaseSchemaBackend
from pandera.backends.pandas.error_formatters import (
    format_generic_error_message,
    format_vectorized_error_message,
    reshape_failure_cases,
    scalar_failure_case,
)
from pandera.errors import SchemaError


class ColumnInfo(NamedTuple):
    sorted_column_names: Iterable
    expanded_column_names: FrozenSet
    destuttered_column_names: List
    absent_column_names: List
    lazy_exclude_column_names: List


FieldCheckObj = Union[pd.Series, pd.DataFrame]

T = TypeVar(
    "T",
    pd.Series,
    pd.DataFrame,
    FieldCheckObj,
    covariant=True,
)


class PandasSchemaBackend(BaseSchemaBackend):
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
        check_obj,
        schema,
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
