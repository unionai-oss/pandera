"""Data validation checks."""

from collections import namedtuple
import operator
import re
from typing import Dict, Union, Optional, List, Callable, Iterable

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
        """Create a :class:`Check` ensuring all values of a series are strictly greater
            than a certain value.

        :param min_value: Lower bound to be exceeded. Must be a type comparable to
            the dtype of the :class:`pandas.Series` to be validated (e.g. a numerical
            type for float or int and a datetime for datetime).

        :returns :class:`Check` object
        """
        if min_value is None:
            raise ValueError("min_value must not be None")

        def _greater_than(series: pd.Series) -> pd.Series:
            """Comparison function for check"""
            return series > min_value

        return Check(fn=_greater_than, error="greater_than(%s)" % min_value)

    @staticmethod
    def greater_than_or_equal_to(min_value) -> 'Check':
        """Create a :class:`Check` ensuring all values are greater or equal a certain value.

        :param min_value: Allowed minimum value for values of a series. Must be a type
            comparable to the dtype of the :class:`pandas.Series` to be validated.

        :returns :class:`Check` object
        """
        if min_value is None:
            raise ValueError("min_value must not be None")

        def _greater_or_equal(series: pd.Series) -> pd.Series:
            """Comparison function for check"""
            return series >= min_value

        return Check(fn=_greater_or_equal,
                     error="greater_than_or_equal_to(%s)" % min_value)

    @staticmethod
    def less_than(max_value) -> 'Check':
        """Create a :class:`Check` ensuring all values are strictly below a certain value.

        :param max_value: All elements of a series must be strictly smaller than this.
            Must be a type comparable to the dtype of the :class:`pandas.Series` to be
            validated.

        :returns :class:`Check` object
        """
        if max_value is None:
            raise ValueError("max_value must not be None")

        def _less_than(series: pd.Series) -> pd.Series:
            """Comparison function for check"""
            return series < max_value

        return Check(fn=_less_than, error="less_than(%s)" % max_value)

    @staticmethod
    def less_than_or_equal_to(max_value) -> 'Check':
        """Create a :class:`Check` ensuring no value of a series exceeds a certain value.

        :param max_value: Upper bound not to be exceeded. Must be a type comparable to
            the dtype of the :class:`pandas.Series` to be validated.

        :returns :class:`Check` object
        """
        if max_value is None:
            raise ValueError("max_value must not be None")

        def _less_or_equal(series: pd.Series) -> pd.Series:
            """Comparison function for check"""
            return series <= max_value

        return Check(fn=_less_or_equal, error="less_than_or_equal_to(%s)" % max_value)

    @staticmethod
    def in_range(min_value, max_value, include_min=True, include_max=True) -> 'Check':
        """Create a :class:`Check` ensuring all values of a series are within an interval.

        :param min_value: Left / lower endpoint of the interval.
        :param max_value: Right / upper endpoint of the interval. Must not be smaller
            than min_value.
        :param include_min: Defines whether min_value is also an allowed value
            (the default) or whether all values must be strictly greater than min_value.
        :param include_max: Defines whether min_value is also an allowed value
            (the default) or whether all values must be strictly smaller than max_value.

        Both endpoints must be a type comparable to the dtype of the
        :class:`pandas.Series` to be validated.

        :returns :class:`Check` object
        """
        if min_value is None:
            raise ValueError("min_value must not be None")
        if max_value is None:
            raise ValueError("max_value must not be None")
        if max_value < min_value or (min_value == max_value
                                     and (not include_min or not include_max)):
            raise ValueError("The combination of min_value = %s and max_value = %s "
                             "defines an empty interval!" % (min_value, max_value))
        # Using functions from operator module to keep conditions out of the closure
        left_op = operator.le if include_min else operator.lt
        right_op = operator.ge if include_max else operator.gt

        def _in_range(series: pd.Series) -> pd.Series:
            """Comparison function for check"""
            return left_op(min_value, series) & right_op(max_value, series)

        return Check(fn=_in_range,
                     error="in_range(%s, %s)" % (min_value, max_value))

    @staticmethod
    def equal_to(value) -> 'Check':
        """Create a :class:`Check` ensuring all elements of a series equal a certain value.

        :param value: This value all elements of a given :class:`pandas.Series` must have.

        :returns :class:`Check` object
        """
        def _equal(series: pd.Series) -> pd.Series:
            """Comparison function for check"""
            return series == value

        return Check(fn=_equal, error="equal_to(%s)" % value)

    @staticmethod
    def not_equal_to(value) -> 'Check':
        """Create a :class:`Check` ensuring no elements of a series equals a certain value.

        :param value: This value must not occur in a :class:`pandas.Series` to check.

        :returns :class:`Check` object
        """
        def _not_equal(series: pd.Series) -> pd.Series:
            """Comparison function for check"""
            return series != value

        return Check(fn=_not_equal, error="not_equal_to(%s)" % value)

    @staticmethod
    def isin(allowed_values: Iterable) -> 'Check':
        """Create a :class:`Check` to ensure only allowed values occur within a series.

        :param allowed_values: The set of allowed values. May be any iterable.

        :returns :class:`Check` object

        Note: It is checked whether all elements of a :class:`pandas.Series` are part
        of the set of elements of allowed values. If allowed values is a string, the
        set of elements consists of all distinct characters of the string. Thus only
        single characters which occur in allowed_values at least once can meet this
        condition. If you want to check for substrings use :func:`Check.str_is_substring`.
        """
        # Turn allowed_values into a set. Not only for performance but also avoid issues
        # with a mutable argument passed by reference which may be changed from outside.
        try:
            allowed_values = frozenset(allowed_values)
        except TypeError:
            raise ValueError("Argument allowed_values must be iterable. Got %s" %
                             allowed_values)

        def _isin(series: pd.Series) -> pd.Series:
            """Comparison function for check"""
            return series.isin(allowed_values)

        return Check(fn=_isin, error="isin(%s)" % allowed_values)

    @staticmethod
    def notin(forbidden_values: Iterable) -> 'Check':
        """Create a :class:`Check` to ensure some defined values don't occur within a series.

        :param forbidden_values: The set of values which should not occur. May be any iterable.

        :returns :class:`Check` object

        Note: Like :func:`Check.isin` this check operates on single characters if it is
        applied on strings. A string as paraforbidden_valuesmeter forbidden_values is understood as
        set of prohibited characters. Any string of length > 1 can't be in it by
        design.
        """
        # Turn forbidden_values into a set. Not only for performance but also avoid issues
        # with a mutable argument passed by reference which may be changed from outside.
        try:
            forbidden_values = frozenset(forbidden_values)
        except TypeError:
            raise ValueError("Argument forbidden_values must be iterable. Got %s" %
                             forbidden_values)

        def _notin(series: pd.Series) -> pd.Series:
            """Comparison function for check"""
            return ~series.isin(forbidden_values)

        return Check(fn=_notin, error="notin(%s)" % forbidden_values)

    @staticmethod
    def str_matches(pattern: str) -> 'Check':
        """Create a :class:`Check` to validate if strings values match a regular expression.

        :param pattern: Regular expression pattern to use for matching

        :returns :class:`Check` object

        The behaviour is as of :func:`pandas.Series.str.match`.
        """
        # By compiling the regex we get the benefit of an early argument check
        try:
            regex = re.compile(pattern)
        except TypeError:
            raise ValueError('pattern="%s" cannot be compiled as regular expression' %
                             pattern)

        def _match(series: pd.Series) -> pd.Series:
            """Check if all strings in the series match the regular expression."""
            return series.str.match(regex, na=False)

        return Check(fn=_match, error="str_matches(%s)" % regex)

    @staticmethod
    def str_contains(pattern: str) -> 'Check':
        """Create a :class:`Check` to validate if the pattern can be found within each row

        :param pattern: Regular expression pattern to use for searching

        :returns :class:`Check` object

        The behaviour is as of :func:`pandas.Series.str.contains`.
        """
        # By compiling the regex we get the benefit of an early argument check
        try:
            regex = re.compile(pattern)
        except TypeError:
            raise ValueError('pattern="%s" cannot be compiled as regular expression' %
                             pattern)

        def _contains(series: pd.Series) -> pd.Series:
            """Check if a regex search is successful within each value"""
            return series.str.contains(regex, na=False)

        return Check(fn=_contains, error="str_contains(%s)" % regex)

    @staticmethod
    def str_startswith(string: str) -> 'Check':
        """Create a :class:`Check` to validate if all values start with a certain string

        :param string: String all values should start with

        :returns :class:`Check` object
        """
        def _startswith(series: pd.Series) -> pd.Series:
            """Returns true only for strings starting with string"""
            return series.str.startswith(string, na=False)

        return Check(fn=_startswith, error="str_startswith(%s)" % string)

    @staticmethod
    def str_endswith(string: str) -> 'Check':
        """Create a :class:`Check` to validate if all values ends with a certain string

        :param string: String all values should end with

        :returns :class:`Check` object
        """
        def _endswith(series: pd.Series) -> pd.Series:
            """Returns true only for strings ending with string"""
            return series.str.endswith(string, na=False)

        return Check(fn=_endswith, error="str_endswith(%s)" % string)

    @staticmethod
    def str_length(min_len: int = None, max_len: int = None) -> 'Check':
        """Create a :class:`Check` to validate  if the length of strings is within a specified range

        :param min_len: Minimum length of strings (default: no minimum)
        :param max_len: Maximu length of strings (default: no maximum)

        :returns :class:`Check` object
        """
        if min_len is None and max_len is None:
            raise ValueError("At least a minimum or a maximum need to be specified. Got None.")
        if max_len is None:
            def check_fn(series: pd.Series) -> pd.Series:
                """Check for the minimum string length"""
                return series.str.len() >= min_len
        elif min_len is None:
            def check_fn(series: pd.Series) -> pd.Series:
                """Check for the maximum string length"""
                return series.str.len() <= max_len
        else:
            def check_fn(series: pd.Series) -> pd.Series:
                """Check for both, minimum and maximum string length"""
                return (series.str.len() <= max_len) & (series.str.len() >= min_len)

        return Check(fn=check_fn, error="str_length(%s, %s)" % (min_len, max_len))
