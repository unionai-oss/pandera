"""Data validation checks."""

import pandas as pd

from typing import Union, Optional, List, Dict, Callable

from . import errors, constants
from .dtypes import PandasDtype


GroupbyObject = Union[
    pd.core.groupby.SeriesGroupBy,
    pd.core.groupby.DataFrameGroupBy
]

SeriesCheckObj = Union[pd.Series, Dict[str, pd.Series]]
DataFrameCheckObj = Union[pd.DataFrame, Dict[str, pd.DataFrame]]


class Check(object):
    """Check a pandas Series or DataFrame for certain properties."""

    def __init__(
            self,
            fn: Callable,
            groups: Optional[Union[str, List[str]]] = None,
            groupby: Optional[Union[str, List[str], Callable]] = None,
            element_wise: bool = False,
            error: Optional[str] = None,
            n_failure_cases: Optional[int] = constants.N_FAILURE_CASES):
        """Apply a validation function to each element, Series, or DataFrame.

        :param fn: A function to check pandas data structure. For Column
            or SeriesSchema checks, if element_wise is True, this function
            should have the signature: ``Callable[[pd.Series],
            Union[pd.Series, bool]]``, where the output series is a boolean
            vector.

            If element_wise is False, this function should have the signature:
            ``Callable[[Any], bool]``, where ``Any`` is an element in the
            column.

            For DataFrameSchema checks, if element_wise=True, fn
            should have the signature: ``Callable[[pd.DataFrame],
            Union[pd.DataFrame, pd.Series, bool]]``, where the output dataframe
            or series contains booleans.

            If element_wise is True, fn is applied to each row in
            the dataframe with the signature ``Callable[[pd.Series], bool]``
            where the series input is a row in the dataframe.
        :param groups: The dict input to the `fn` callable will be constrained
            to the groups specified by `groups`.
        :param groupby: If a string or list of strings is provided, these
            columns are used to group the Column series. If a
            callable is passed, the expected signature is: ``Callable[
            [pd.DataFrame], pd.core.groupby.DataFrameGroupBy]``

            The the case of ``Column`` checks, this function has access to the
            entire dataframe, but ``Column.name`` is selected from this
            DataFrameGroupby object so that a SeriesGroupBy object is passed
            into ``fn``.

            Specifying the groupby argument changes the ``fn`` signature to: ``
            Callable[[Dict[Union[str, Tuple[str]], pd.Series]],
            Union[bool, pd.Series]]``, where the input is a dictionary mapping
            keys to subsets of the column/dataframe.
        :param element_wise: Whether or not to apply validator in an
            element-wise fashion. If bool, assumes that all checks should be
            applied to the column element-wise. If list, should be the same
            number of elements as checks.
        :param error: custom error message if series fails validation
            check.
        :param n_failure_cases: report the top n failure cases. If None, then
            report all failure cases.

        :example:

        >>> import pandas as pd
        >>> import pandera as pa
        >>> from pandera import Column, Check, DataFrameSchema
        >>>
        >>> # column checks are vectorized by default
        >>> check_positive = Check(lambda s: s > 0)
        >>>
        >>> # define an element-wise check
        >>> check_even = Check(lambda x: x % 2 == 0, element_wise=True)
        >>>
        >>> # specify assertions across categorical variables using `groupby`,
        >>> # for example, make sure the mean measure for group "A" is always
        >>> # larger than the mean measure for group "B"
        >>> check_by_group = Check(
        ...     lambda measures: measures["A"].mean() > measures["B"].mean(),
        ...     groupby=["group"],
        ... )
        >>>
        >>> # define a wide DataFrame-level check
        >>> check_dataframe = Check(
        ...     lambda df: df["measure_1"] > df["measure_2"])
        >>>
        >>> measure_checks = [check_positive, check_even, check_by_group]
        >>>
        >>> schema = DataFrameSchema(
        ...     columns={
        ...         "measure_1": Column(pa.Int, checks=measure_checks),
        ...         "measure_2": Column(pa.Int, checks=measure_checks),
        ...         "group": Column(pa.String),
        ...     },
        ...     checks=check_dataframe
        ... )
        >>>
        >>> df = pd.DataFrame({
        ...     "measure_1": [10, 12, 14, 16],
        ...     "measure_2": [2, 4, 6, 8],
        ...     "group": ["B", "B", "A", "A"]
        ... })
        >>>
        >>> schema.validate(df)[["measure_1", "measure_2", "group"]]
           measure_1  measure_2 group
        0         10          2     B
        1         12          4     B
        2         14          6     A
        3         16          8     A

        See :ref:`here<checks>` for more usage details.

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
        """Check error message."""
        name = getattr(self.fn, '__name__', self.fn.__class__.__name__)
        if self.error:
            return "%s: %s" % (name, self.error)
        return "%s" % name

    def _vectorized_error_message(
            self,
            parent_schema: type,
            check_index: int,
            failure_cases: Union[pd.DataFrame, pd.Series]) -> str:
        """Construct an error message when an element-wise validator fails.

        :param parent_schema: class of schema being validated.
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
            parent_schema: type,
            check_index: int) -> str:
        """Construct an error message when a check validator fails.

        :param parent_schema: class of schema being validated.
        :param check_index: The validator that failed.

        """
        return "%s failed series validator %d: %s" % \
               (parent_schema, check_index, self._error_message)

    def _format_failure_cases(
            self,
            failure_cases: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """Construct readable error messages for vectorized_error_message.

        :param failure_cases: The failure cases encountered by the element-wise
            validator.
        :returns: DataFrame where index contains failure cases, the "index"
            column contains a list of integer indexes in the validation
            DataFrame that caused the failure, and a "count" column
            representing how many failures of that case occurred.

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

    def _format_groupby_input(
            self,
            groupby_obj: GroupbyObject,
            groups: List[str]
            ) -> Union[Dict[str, Union[pd.Series, pd.DataFrame]]]:
        """Format groupby object into dict of groups to Series or DataFrame.

        :param groupby_obj: a pandas groupby object.
        :param groups: only include these groups in the output.
        :returns: dictionary mapping group names to Series or DataFrame.
        """
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

    def _prepare_series_input(
            self,
            series: pd.Series,
            dataframe_context: pd.DataFrame) -> SeriesCheckObj:
        """Prepare input for Column check.

        :param pd.Series series: one-dimensional ndarray with axis labels
            (including time series).
        :param pd.DataFrame dataframe_context: optional dataframe to supply
            when checking a Column in a DataFrameSchema.
        :returns: a Series, or a dictionary mapping groups to Series
            to be used by `_check_fn` and `_vectorized_check`

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

        return self._format_groupby_input(groupby_obj, self.groups)

    def _prepare_dataframe_input(
            self, dataframe: pd.DataFrame) -> DataFrameCheckObj:
        """Prepare input for DataFrameSchema check.

        :param dataframe: dataframe to validate.
        :returns: a DataFrame, or a dictionary mapping groups to pd.DataFrame
            to be used by `_check_fn` and `_vectorized_check`
        """
        if self.groupby is None:
            return dataframe
        else:
            groupby_obj = dataframe.groupby(self.groupby)
        return self._format_groupby_input(groupby_obj, self.groups)

    def _vectorized_check(
            self,
            parent_schema: type,
            check_index: int,
            check_obj: Dict[str, Union[pd.Series, pd.DataFrame]]
            ) -> bool:
        """Perform a vectorized check on a series.

        :param parent_schema: class of schema being validated.
        :param check_index: The validator to check the series for
        :param check_obj: a dictionary of pd.Series to be used by
            `_check_fn` and `_vectorized_check`
        :returns: True if pandas DataFramf or Series is valid.
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
            parent_schema: type,
            check_index: int,
            check_obj: Union[pd.Series, pd.DataFrame]) -> bool:
        """Validate pandas DataFrame or Series.

        :param parent_schema: class of schema being validated.
        :check_index: index of check that is being validated.
        :check_obj: pandas DataFrame of Series to validate.
        :returns: True if check passes.
        """
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
