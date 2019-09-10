"""Data validation checks."""

import pandas as pd

from functools import partial
from typing import Union, Optional, List, Dict

from . import errors, constants
from .dtypes import PandasDtype


class Check(object):

    def __init__(
            self,
            fn: callable,
            groups: Union[str, List[str], None] = None,
            groupby: Union[str, List[str], callable, None] = None,
            element_wise: Union[bool, List[bool]] = False,
            error: Optional[str] = None,
            n_failure_cases: Optional[int] = constants.N_FAILURE_CASES):
        """Check object applies function element-wise or series-wise

        :param callable fn: A function to check series schema. If element_wise
            is True, then callable signature should be: x -> bool where x is a
            scalar element in the column. Otherwise, signature is expected
            to be: pd.Series -> bool|pd.Series[bool].
        :param groups: The dict input to the `fn` callable will be constrained
            to the groups specified by `groups`.
        :type groups: str|list[str]|None
        :param groupby: Only applies to Column Checks. If a string or list of
            strings is provided, then these columns are used to group the
            Column Series by `groupby`. If a callable is passed, the expected
            signature is DataFrame -> DataFrameGroupby. The function has access
            to the entire dataframe, but the Column.name is selected from this
            DataFrameGroupby object so that a SeriesGroupBy object is passed
            into `fn`.

            Specifying this argument changes the `fn` signature to:

            dict[str|tuple[str], Series] -> bool|pd.Series[bool]

            Where specific groups can be obtained from the input dict.
        :type groupby: str|list[str]|callable|None
        :param element_wise: Whether or not to apply validator in an
            element-wise fashion. If bool, assumes that all checks should be
            applied to the column element-wise. If list, should be the same
            number of elements as checks.
        :type element_wise: bool|list[bool]
        :param str error: custom error message if series fails validation
            check.
        :type str error:
        :param n_failure_cases: report the top n failure cases. If None, then
            report all failure cases.
        """
        if element_wise and groupby is not None:
            raise errors.SchemaInitError(
                "Cannot use groupby when element_wise=True.")
        self.fn = fn
        self.element_wise = element_wise
        self.error = error
        self.n_failure_cases = n_failure_cases

        if groupby is None and groups is not None:
            raise ValueError(
                "`groupby` argument needs to be provided when `groups` "
                "argument is defined")

        if isinstance(groupby, str):
            groupby = [groupby]
        self.groupby = groupby
        if isinstance(groups, str):
            groups = [groups]
        self.groups = groups

    @property
    def _error_message(self):
        name = getattr(self.fn, '__name__', self.fn.__class__.__name__)
        if self.error:
            return "%s: %s" % (name, self.error)
        return "%s" % name

    def _vectorized_error_message(
            self,
            parent_schema,
            check_index: int,
            failure_cases: Union[pd.DataFrame, pd.Series]) -> str:
        """Constructs an error message when an element-wise validator fails.

        :param parent_schema: The schema object that is being checked and that
            was inherited from the parent class.
        :param check_index: The validator that failed.
        :param failure_cases: The failure cases encountered by the element-wise
            validator.

        """
        return (
                "%s failed element-wise validator %d:\n"
                "%s\nfailure cases:\n%s" %
                (parent_schema, check_index,
                 self._error_message,
                 self._format_failure_cases(failure_cases)))

    def _generic_error_message(
            self,
            parent_schema,
            check_index: int) -> str:
        """Constructs an error message when a check validator fails.

        :param parent_schema: The schema object that is being checked and that
            was inherited from the parent class.
        :param check_index: The validator that failed.

        """
        return "%s failed series validator %d: %s" % \
               (parent_schema, check_index, self._error_message)

    def _format_failure_cases(
            self,
            failure_cases: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """Constructs readable error messages for vectorized_error_message.

        :param failure_cases: The failure cases encountered by the element-wise
            validator.

        """
        if isinstance(failure_cases.index, pd.MultiIndex):
            failure_cases = (
                failure_cases
                .rename("failure_case")
                .reset_index()
                .assign(
                    index=lambda df: (
                        df.apply(tuple, axis=1).astype(str)
                    )
                )
            )
        elif isinstance(failure_cases, pd.DataFrame):
            failure_cases = (
                failure_cases
                .pipe(lambda df: pd.Series(
                    df.itertuples()).map(lambda x: x.__repr__()))
                .rename("failure_case")
                .reset_index()
            )
        else:
            failure_cases = (
                failure_cases
                .rename("failure_case")
                .reset_index()
            )

        failure_cases = (
            failure_cases
            .groupby("failure_case").index.agg([list, len])
            .rename(columns={"list": "index", "len": "count"})
            .sort_values("count", ascending=False)
        )

        self.failure_cases = failure_cases
        return failure_cases.head(self.n_failure_cases)

    def _format_input(
            self,
            groupby_obj,
            groups) -> Union[Dict[str, Union[pd.Series, pd.DataFrame]]]:
        if groups is None:
            return {group_key: group for group_key, group in groupby_obj}
        group_keys = set(group_key for group_key, _ in groupby_obj)
        invalid_groups = [g for g in groups if g not in group_keys]
        if invalid_groups:
            raise KeyError(
                "groups %s provided in `groups` argument not a valid group "
                "key. Valid group keys: %s" % (invalid_groups, group_keys))
        return {
            group_key: group for group_key, group in groupby_obj
            if group_key in groups
        }

    def prepare_series_input(
            self,
            series: pd.Series,
            dataframe_context: pd.DataFrame) -> Dict[str, pd.Series]:
        """Prepare input for Column check.

        :param pd.Series series: one-dimensional ndarray with axis labels
            (including time series).
        :param pd.DataFrame dataframe_context: optional dataframe to supply
            when checking a Column in a DataFrameSchema.
        :return: a check_obj dictionary of pd.Series to be used by `_check_fn`
            and `_vectorized_check`

        """
        if dataframe_context is None or self.groupby is None:
            return series
        elif isinstance(self.groupby, list):
            groupby_obj = (
                pd.concat([series, dataframe_context[self.groupby]], axis=1)
                .groupby(self.groupby)[series.name]
            )
        elif callable(self.groupby):
            groupby_obj = self.groupby(
                pd.concat([series, dataframe_context], axis=1))[series.name]
        else:
            raise TypeError("Type %s not recognized for `groupby` argument.")

        return self._format_input(groupby_obj, self.groups)

    def prepare_dataframe_input(
            self, dataframe: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Prepare input for DataFrameSchema check."""
        if self.groupby is None:
            return dataframe
        else:
            groupby_obj = dataframe.groupby(self.groupby)
        return self._format_input(groupby_obj, self.groups)

    def _vectorized_check(
            self,
            parent_schema,
            check_index: int,
            check_obj: Dict[str, Union[pd.Series, pd.DataFrame]]):
        """Perform a vectorized check on a series.

        :param parent_schema: The schema object that is being checked and that
            was inherited from the parent class.
        :param check_index: The validator to check the series for
        :param check_obj: a dictionary of pd.Series to be used by
            `_check_fn` and `_vectorized_check`
        """
        val_result = self.fn(check_obj)
        if isinstance(val_result, pd.Series):
            if not val_result.dtype == PandasDtype.Bool.value:
                raise TypeError(
                    "validator %d: %s must return bool or Series of type "
                    "bool, found %s" %
                    (check_index, self.fn.__name__, val_result.dtype))
            if val_result.all():
                return True
            elif isinstance(check_obj, dict) or \
                    check_obj.shape[0] != val_result.shape[0] or \
                    (check_obj.index != val_result.index).all():
                raise errors.SchemaError(
                    self._generic_error_message(parent_schema, check_index))
            else:
                raise errors.SchemaError(self._vectorized_error_message(
                    parent_schema, check_index, check_obj[~val_result]))
        else:
            if val_result:
                return True
            raise errors.SchemaError(
                self._generic_error_message(parent_schema, check_index))

    def __call__(
            self,
            parent_schema,
            check_index: int,
            check_obj: Dict[str, Union[pd.Series, pd.DataFrame]]):
        _vcheck = partial(
            self._vectorized_check, parent_schema, check_index)
        if self.element_wise:
            val_result = check_obj.apply(self.fn, axis=1) if \
                isinstance(check_obj, pd.DataFrame) else check_obj.map(self.fn)
            if val_result.all():
                return True
            raise errors.SchemaError(self._vectorized_error_message(
                parent_schema, check_index, check_obj[~val_result]))
        elif isinstance(check_obj, (pd.Series, dict, pd.DataFrame)):
            return self._vectorized_check(
                parent_schema, check_index, check_obj)
        else:
            raise ValueError(
                "check_obj type %s not supported. Must be a "
                "Series, a dictionary of Series, or DataFrame" % check_obj)
