"""Data validation checks."""

from collections import namedtuple
from typing import Dict, Union, Optional, List, Callable

import pandas as pd

from . import errors, constants


CheckResult = namedtuple(
    "CheckResult", ["check_passed", "checked_object", "failure_cases"])


GroupbyObject = Union[
    pd.core.groupby.SeriesGroupBy,
    pd.core.groupby.DataFrameGroupBy
]

SeriesCheckObj = Union[pd.Series, Dict[str, pd.Series]]
DataFrameCheckObj = Union[pd.DataFrame, Dict[str, pd.DataFrame]]


class Check():
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
        self.failure_cases = None

    def _format_groupby_input(
            self,
            groupby_obj: GroupbyObject,
            groups: List[str]
    ) -> Union[Dict[str, Union[pd.Series, pd.DataFrame]]]:
        # pylint: disable=no-self-use
        """Format groupby object into dict of groups to Series or DataFrame.

        :param groupby_obj: a pandas groupby object.
        :param groups: only include these groups in the output.
        :returns: dictionary mapping group names to Series or DataFrame.
        """
        if groups is None:
            return dict(list(groupby_obj))
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
            dataframe_context: Optional[pd.DataFrame] = None
    ) -> SeriesCheckObj:
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
        if isinstance(self.groupby, list):
            groupby_obj = (
                pd.concat([series, dataframe_context[self.groupby]], axis=1)
                .groupby(self.groupby)[series.name]
            )
            return self._format_groupby_input(groupby_obj, self.groups)
        if callable(self.groupby):
            groupby_obj = self.groupby(
                pd.concat([series, dataframe_context], axis=1))[series.name]
            return self._format_groupby_input(groupby_obj, self.groups)
        raise TypeError("Type %s not recognized for `groupby` argument.")

    def _prepare_dataframe_input(
            self, dataframe: pd.DataFrame) -> DataFrameCheckObj:
        """Prepare input for DataFrameSchema check.

        :param dataframe: dataframe to validate.
        :returns: a DataFrame, or a dictionary mapping groups to pd.DataFrame
            to be used by `_check_fn` and `_vectorized_check`
        """
        if self.groupby is None:
            return dataframe
        groupby_obj = dataframe.groupby(self.groupby)
        return self._format_groupby_input(groupby_obj, self.groups)

    def __call__(
            self,
            df_or_series: Union[pd.DataFrame, pd.Series],
            column: str = None,
    ) -> CheckResult:
        """Validate pandas DataFrame or Series.

        :df_or_series: pandas DataFrame of Series to validate.
        :column: apply the check function to this column.
        :returns: CheckResult tuple containing checked object,
            check validation result, and failure cases from the checked object.
        """
        if column is not None \
                and isinstance(df_or_series, pd.DataFrame):
            column_dataframe_context = df_or_series.drop(
                column, axis="columns")
            df_or_series = df_or_series[column].copy()
        else:
            column_dataframe_context = None

        # prepare check object
        if isinstance(df_or_series, pd.Series):
            check_obj = self._prepare_series_input(
                df_or_series, column_dataframe_context)
        elif isinstance(df_or_series, pd.DataFrame):
            check_obj = self._prepare_dataframe_input(df_or_series)
        else:
            raise ValueError(
                "object of type %s not supported. Must be a "
                "Series, a dictionary of Series, or DataFrame" %
                df_or_series)

        # apply check function to check object
        if self.element_wise:
            check_result = check_obj.apply(self.fn, axis=1) if \
                isinstance(check_obj, pd.DataFrame) else check_obj.map(self.fn)
        else:
            # vectorized check function case
            check_result = self.fn(check_obj)

        # failure cases only apply when the check function returns a boolean
        # series that matches the shape and index of the check_obj
        if isinstance(check_obj, dict) or \
                isinstance(check_result, bool) or \
                not isinstance(check_result, pd.Series) or \
                check_obj.shape[0] != check_result.shape[0] or \
                (check_obj.index != check_result.index).all():
            failure_cases = None
        else:
            failure_cases = check_obj[~check_result]

        check_passed = check_result.all() if \
            isinstance(check_result, pd.Series) else check_result

        return CheckResult(check_passed, check_obj, failure_cases)

    def __eq__(self, other):
        are_fn_objects_equal = self.__dict__["fn"].__code__.co_code == \
                               other.__dict__["fn"].__code__.co_code

        are_all_other_check_attributes_equal = \
            {i: self.__dict__[i] for i in self.__dict__ if i != 'fn'} == \
            {i: other.__dict__[i] for i in other.__dict__ if i != 'fn'}

        return are_fn_objects_equal and are_all_other_check_attributes_equal

    def __repr__(self):
        name = getattr(self.fn, '__name__', self.fn.__class__.__name__)
        return "<Check %s: %s>" % (name, self.error) \
            if self.error is not None else "<Check %s>" % name

    @staticmethod
    def greater_than(min_value) -> 'Check':
        """Ensure values of a series are above a certain threshold.

        :param min_value: Lower bound to be exceeded. Must be a type comparable to
            the type of the pandas series to be validated (e.g. a numerical type
            for float or int and a datetime for datetime).
        :returns pandera.Check object

        """
        if min_value is None:
            raise ValueError("min_value must not be None")

        def _greater_than(series: pd.Series) -> pd.Series:
            """Comparison function for check"""
            return series > min_value

        return Check(fn=_greater_than, error="greater_than(%s) check" % min_value)
