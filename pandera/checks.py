"""Data validation checks."""

import inspect
import operator
import re
from collections import ChainMap, namedtuple
from functools import partial, wraps
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    no_type_check,
)

import pandas as pd

from . import check_utils, errors
from . import strategies as st

CheckResult = namedtuple(
    "CheckResult",
    ["check_output", "check_passed", "checked_object", "failure_cases"],
)


GroupbyObject = Union[
    pd.core.groupby.SeriesGroupBy, pd.core.groupby.DataFrameGroupBy
]

SeriesCheckObj = Union[pd.Series, Dict[str, pd.Series]]
DataFrameCheckObj = Union[pd.DataFrame, Dict[str, pd.DataFrame]]


def register_check_statistics(statistics_args):
    """Decorator to set statistics based on Check method."""

    def register_check_statistics_decorator(class_method):
        @wraps(class_method)
        def _wrapper(cls, *args, **kwargs):
            args = list(args)
            arg_names = inspect.getfullargspec(class_method).args[1:]
            if not arg_names:
                arg_names = statistics_args
            args_dict = {**dict(zip(arg_names, args)), **kwargs}
            check = class_method(cls, *args, **kwargs)
            check.statistics = {
                stat: args_dict.get(stat) for stat in statistics_args
            }
            check.statistics_args = statistics_args
            return check

        return _wrapper

    return register_check_statistics_decorator


_T = TypeVar("_T", bound="_CheckBase")


class _CheckMeta(type):  # pragma: no cover
    """Check metaclass."""

    REGISTERED_CUSTOM_CHECKS: Dict[str, Callable] = {}  # noqa

    def __getattr__(cls, name: str) -> Any:
        """Prevent attribute errors for registered checks."""
        attr = ChainMap(cls.__dict__, cls.REGISTERED_CUSTOM_CHECKS).get(name)
        if attr is None:
            raise AttributeError(
                f"'{cls}' object has no attribute '{name}'. "
                "Make sure any custom checks have been registered "
                "using the extensions api."
            )
        return attr

    def __dir__(cls) -> Iterable[str]:
        """Allow custom checks to show up as attributes when autocompleting."""
        return chain(super().__dir__(), cls.REGISTERED_CUSTOM_CHECKS.keys())

    # pylint: disable=line-too-long
    # mypy has limited metaclass support so this doesn't pass typecheck
    # see https://mypy.readthedocs.io/en/stable/metaclasses.html#gotchas-and-limitations-of-metaclass-support
    # pylint: enable=line-too-long
    @no_type_check
    def __contains__(cls: Type[_T], item: Union[_T, str]) -> bool:
        """Allow lookups for registered checks."""
        if isinstance(item, cls):
            name = item.name
            return hasattr(cls, name)

        # assume item is str
        return hasattr(cls, item)


class _CheckBase(metaclass=_CheckMeta):
    """Check base class."""

    def __init__(
        self,
        check_fn: Union[
            Callable[[pd.Series], Union[pd.Series, bool]],
            Callable[[pd.DataFrame], Union[pd.DataFrame, pd.Series, bool]],
        ],
        groups: Optional[Union[str, List[str]]] = None,
        groupby: Optional[Union[str, List[str], Callable]] = None,
        ignore_na: bool = True,
        element_wise: bool = False,
        name: str = None,
        error: Optional[str] = None,
        raise_warning: bool = False,
        n_failure_cases: Optional[int] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        **check_kwargs,
    ) -> None:
        """Apply a validation function to each element, Series, or DataFrame.

        :param check_fn: A function to check pandas data structure. For Column
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
            into ``check_fn``.

            Specifying the groupby argument changes the ``check_fn`` signature
            to:

            ``Callable[[Dict[Union[str, Tuple[str]], pd.Series]], Union[bool, pd.Series]]``  # noqa

            where the input is a dictionary mapping
            keys to subsets of the column/dataframe.
        :param ignore_na: If True, null values will be ignored when determining
            if a check passed or failed. For dataframes, ignores rows with any
            null value. *New in version 0.4.0*
        :param element_wise: Whether or not to apply validator in an
            element-wise fashion. If bool, assumes that all checks should be
            applied to the column element-wise. If list, should be the same
            number of elements as checks.
        :param name: optional name for the check.
        :param error: custom error message if series fails validation
            check.
        :param raise_warning: if True, raise a UserWarning and do not throw
            exception instead of raising a SchemaError for a specific check.
            This option should be used carefully in cases where a failing
            check is informational and shouldn't stop execution of the program.
        :param n_failure_cases: report the first n unique failure cases. If
            None, report all failure cases.
        :param title: A human-readable label for the check.
        :param description: An arbitrary textual description of the check.
        :param check_kwargs: key-word arguments to pass into ``check_fn``

        :example:

        >>> import pandas as pd
        >>> import pandera as pa
        >>>
        >>>
        >>> # column checks are vectorized by default
        >>> check_positive = pa.Check(lambda s: s > 0)
        >>>
        >>> # define an element-wise check
        >>> check_even = pa.Check(lambda x: x % 2 == 0, element_wise=True)
        >>>
        >>> # checks can be given human-readable metadata
        >>> check_with_metadata = pa.Check(
        ...     lambda x: True,
        ...     title="Always passes",
        ...     description="This check always passes."
        ... )
        >>>
        >>> # specify assertions across categorical variables using `groupby`,
        >>> # for example, make sure the mean measure for group "A" is always
        >>> # larger than the mean measure for group "B"
        >>> check_by_group = pa.Check(
        ...     lambda measures: measures["A"].mean() > measures["B"].mean(),
        ...     groupby=["group"],
        ... )
        >>>
        >>> # define a wide DataFrame-level check
        >>> check_dataframe = pa.Check(
        ...     lambda df: df["measure_1"] > df["measure_2"])
        >>>
        >>> measure_checks = [check_positive, check_even, check_by_group]
        >>>
        >>> schema = pa.DataFrameSchema(
        ...     columns={
        ...         "measure_1": pa.Column(int, checks=measure_checks),
        ...         "measure_2": pa.Column(int, checks=measure_checks),
        ...         "group": pa.Column(str),
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
                "Cannot use groupby when element_wise=True."
            )
        self._check_fn = check_fn
        self._check_kwargs = check_kwargs
        self.element_wise = element_wise
        self.error = error
        self.name = name or getattr(
            self._check_fn, "__name__", self._check_fn.__class__.__name__
        )
        self.ignore_na = ignore_na
        self.raise_warning = raise_warning
        self.n_failure_cases = n_failure_cases
        self.title = title
        self.description = description

        if groupby is None and groups is not None:
            raise ValueError(
                "`groupby` argument needs to be provided when `groups` "
                "argument is defined"
            )

        if isinstance(groupby, str):
            groupby = [groupby]
        self.groupby = groupby
        if isinstance(groups, str):
            groups = [groups]
        self.groups = groups
        self.failure_cases = None

        self._statistics = None

    @property
    def statistics(self) -> Dict[str, Any]:
        """Get check statistics."""
        return getattr(self, "_statistics")

    @statistics.setter
    def statistics(self, statistics):
        """Set check statistics."""
        self._statistics = statistics

    @staticmethod
    def _format_groupby_input(
        groupby_obj: GroupbyObject,
        groups: Optional[List[str]],
    ) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
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
                f"groups {invalid_groups} provided in `groups` argument not a valid group "
                f"key. Valid group keys: {group_keys}"
            )
        return {
            group_key: group
            for group_key, group in groupby_obj
            if group_key in groups
        }

    def _prepare_series_input(
        self,
        df_or_series: Union[pd.Series, pd.DataFrame],
        column: Optional[str] = None,
    ) -> SeriesCheckObj:
        """Prepare input for Column check.

        :param pd.Series series: one-dimensional ndarray with axis labels
            (including time series).
        :param pd.DataFrame dataframe_context: optional dataframe to supply
            when checking a Column in a DataFrameSchema.
        :returns: a Series, or a dictionary mapping groups to Series
            to be used by `_check_fn` and `_vectorized_check`

        """
        if check_utils.is_field(df_or_series):
            return df_or_series
        elif self.groupby is None:
            return df_or_series[column]
        elif isinstance(self.groupby, list):
            return self._format_groupby_input(
                df_or_series.groupby(self.groupby)[column],
                self.groups,
            )
        elif callable(self.groupby):
            return self._format_groupby_input(
                self.groupby(df_or_series)[column],
                self.groups,
            )
        raise TypeError("Type %s not recognized for `groupby` argument.")

    def _prepare_dataframe_input(
        self, dataframe: pd.DataFrame
    ) -> DataFrameCheckObj:
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
        column: Optional[str] = None,
    ) -> CheckResult:
        # pylint: disable=too-many-branches
        """Validate pandas DataFrame or Series.

        :param df_or_series: pandas DataFrame of Series to validate.
        :param column: for dataframe checks, apply the check function to this
            column.
        :returns: CheckResult tuple containing:

            ``check_output``: boolean scalar, ``Series`` or ``DataFrame``
            indicating which elements passed the check.

            ``check_passed``: boolean scalar that indicating whether the check
            passed overall.

            ``checked_object``: the checked object itself. Depending on the
            options provided to the ``Check``, this will be a pandas Series,
            DataFrame, or if the ``groupby`` option is specified, a
            ``Dict[str, Series]`` or ``Dict[str, DataFrame]`` where the keys
            are distinct groups.

            ``failure_cases``: subset of the check_object that failed.
        """
        # prepare check object
        if check_utils.is_field(df_or_series) or (
            column is not None and check_utils.is_table(df_or_series)
        ):
            check_obj = self._prepare_series_input(df_or_series, column)
        elif check_utils.is_table(df_or_series):
            check_obj = self._prepare_dataframe_input(df_or_series)
        else:
            raise ValueError(
                f"object of type {type(df_or_series)} not supported. Must be "
                "a Series, a dictionary of Series, or DataFrame"
            )

        # apply check function to check object
        check_fn = partial(self._check_fn, **self._check_kwargs)

        if self.element_wise:
            check_output = (
                check_obj.apply(check_fn, axis=1)  # type: ignore
                if check_utils.is_table(check_obj)
                else check_obj.map(check_fn)  # type: ignore
                if check_utils.is_field(check_obj)
                else check_fn(check_obj)
            )
        else:
            # vectorized check function case
            check_output = check_fn(check_obj)

        # failure cases only apply when the check function returns a boolean
        # series that matches the shape and index of the check_obj
        if (
            isinstance(check_obj, dict)
            or isinstance(check_output, bool)
            or not check_utils.is_supported_check_obj(check_output)
            or check_obj.shape[0] != check_output.shape[0]
            or (check_obj.index != check_output.index).all()
        ):
            failure_cases = None
        elif check_utils.is_field(check_output):
            (
                check_output,
                failure_cases,
            ) = check_utils.prepare_series_check_output(
                check_obj,
                check_output,
                ignore_na=self.ignore_na,
                n_failure_cases=self.n_failure_cases,
            )
        elif check_utils.is_table(check_output):
            (
                check_output,
                failure_cases,
            ) = check_utils.prepare_dataframe_check_output(
                check_obj,
                check_output,
                df_orig=df_or_series,
                ignore_na=self.ignore_na,
                n_failure_cases=self.n_failure_cases,
            )
        else:
            raise TypeError(
                f"output type of check_fn not recognized: {type(check_output)}"
            )

        check_passed = (
            check_output.all()
            if check_utils.is_field(check_output)
            else check_output.all(axis=None)
            if check_utils.is_table(check_output)
            else check_output
        )
        return CheckResult(
            check_output, check_passed, check_obj, failure_cases
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        are_check_fn_objects_equal = (
            self._get_check_fn_code() == other._get_check_fn_code()
        )

        try:
            are_strategy_fn_objects_equal = all(
                getattr(self.__dict__.get("strategy"), attr)
                == getattr(other.__dict__.get("strategy"), attr)
                for attr in ["func", "args", "keywords"]
            )
        except AttributeError:
            are_strategy_fn_objects_equal = True

        are_all_other_check_attributes_equal = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ["_check_fn", "strategy"]
        } == {
            k: v
            for k, v in other.__dict__.items()
            if k not in ["_check_fn", "strategy"]
        }

        return (
            are_check_fn_objects_equal
            and are_strategy_fn_objects_equal
            and are_all_other_check_attributes_equal
        )

    def _get_check_fn_code(self):
        check_fn = self.__dict__["_check_fn"]
        try:
            code = check_fn.__code__.co_code
        except AttributeError:
            # try accessing the functools.partial wrapper
            code = check_fn.func.__code__.co_code

        return code

    def __hash__(self) -> int:
        return hash(self._get_check_fn_code())

    def __repr__(self) -> str:
        return (
            f"<Check {self.name}: {self.error}>"
            if self.error is not None
            else f"<Check {self.name}>"
        )


class Check(_CheckBase):
    """Check a pandas Series or DataFrame for certain properties."""

    @classmethod
    @st.register_check_strategy(st.eq_strategy)
    @register_check_statistics(["value"])
    def equal_to(cls, value, **kwargs) -> "Check":
        """Ensure all elements of a series equal a certain value.

        *New in version 0.4.5*
        Alias: ``eq``

        :param value: All elements of a given :class:`pandas.Series` must have
            this value
        :param kwargs: key-word arguments passed into the `Check` initializer.

        :returns: :class:`Check` object
        """

        def _equal(series: pd.Series) -> pd.Series:
            """Comparison function for check"""
            return series == value

        return cls(
            _equal,
            name=cls.equal_to.__name__,
            error=f"equal_to({value})",
            **kwargs,
        )

    eq = equal_to

    @classmethod
    @st.register_check_strategy(st.ne_strategy)
    @register_check_statistics(["value"])
    def not_equal_to(cls, value, **kwargs) -> "Check":
        """Ensure no elements of a series equals a certain value.

        *New in version 0.4.5*
        Alias: ``ne``

        :param value: This value must not occur in the checked
            :class:`pandas.Series`.
        :param kwargs: key-word arguments passed into the `Check` initializer.

        :returns: :class:`Check` object
        """

        def _not_equal(series: pd.Series) -> pd.Series:
            """Comparison function for check"""
            return series != value

        return cls(
            _not_equal,
            name=cls.not_equal_to.__name__,
            error=f"not_equal_to({value})",
            **kwargs,
        )

    ne = not_equal_to

    @classmethod
    @st.register_check_strategy(st.gt_strategy)
    @register_check_statistics(["min_value"])
    def greater_than(cls, min_value, **kwargs) -> "Check":
        """Ensure values of a series are strictly greater than a minimum value.

        *New in version 0.4.5*
        Alias: ``gt``

        :param min_value: Lower bound to be exceeded. Must be a type comparable
            to the dtype of the :class:`pandas.Series` to be validated (e.g. a
            numerical type for float or int and a datetime for datetime).
        :param kwargs: key-word arguments passed into the `Check` initializer.

        :returns: :class:`Check` object
        """
        if min_value is None:
            raise ValueError("min_value must not be None")

        def _greater_than(series: pd.Series) -> pd.Series:
            """Comparison function for check"""
            return series > min_value

        return cls(
            _greater_than,
            name=cls.greater_than.__name__,
            error=f"greater_than({min_value})",
            **kwargs,
        )

    gt = greater_than

    @classmethod
    @st.register_check_strategy(st.ge_strategy)
    @register_check_statistics(["min_value"])
    def greater_than_or_equal_to(cls, min_value, **kwargs) -> "Check":
        """Ensure all values are greater or equal a certain value.

        *New in version 0.4.5*
        Alias: ``ge``

        :param min_value: Allowed minimum value for values of a series. Must be
            a type comparable to the dtype of the :class:`pandas.Series` to be
            validated.
        :param kwargs: key-word arguments passed into the `Check` initializer.

        :returns: :class:`Check` object
        """
        if min_value is None:
            raise ValueError("min_value must not be None")

        def _greater_or_equal(series: pd.Series) -> pd.Series:
            """Comparison function for check"""
            return series >= min_value

        return cls(
            _greater_or_equal,
            name=cls.greater_than_or_equal_to.__name__,
            error=f"greater_than_or_equal_to({min_value})",
            **kwargs,
        )

    ge = greater_than_or_equal_to

    @classmethod
    @st.register_check_strategy(st.lt_strategy)
    @register_check_statistics(["max_value"])
    def less_than(cls, max_value, **kwargs) -> "Check":
        """Ensure values of a series are strictly below a maximum value.

        *New in version 0.4.5*
        Alias: ``lt``

        :param max_value: All elements of a series must be strictly smaller
            than this. Must be a type comparable to the dtype of the
            :class:`pandas.Series` to be validated.
        :param kwargs: key-word arguments passed into the `Check` initializer.

        :returns: :class:`Check` object
        """
        if max_value is None:
            raise ValueError("max_value must not be None")

        def _less_than(series: pd.Series) -> pd.Series:
            """Comparison function for check"""
            return series < max_value

        return cls(
            _less_than,
            name=cls.less_than.__name__,
            error=f"less_than({max_value})",
            **kwargs,
        )

    lt = less_than

    @classmethod
    @st.register_check_strategy(st.le_strategy)
    @register_check_statistics(["max_value"])
    def less_than_or_equal_to(cls, max_value, **kwargs) -> "Check":
        """Ensure values are less than or equal to a maximum value.

        *New in version 0.4.5*
        Alias: ``le``

        :param max_value: Upper bound not to be exceeded. Must be a type
            comparable to the dtype of the :class:`pandas.Series` to be
            validated.
        :param kwargs: key-word arguments passed into the `Check` initializer.

        :returns: :class:`Check` object
        """
        if max_value is None:
            raise ValueError("max_value must not be None")

        def _less_or_equal(series: pd.Series) -> pd.Series:
            """Comparison function for check"""
            return series <= max_value

        return cls(
            _less_or_equal,
            name=cls.less_than_or_equal_to.__name__,
            error=f"less_than_or_equal_to({max_value})",
            **kwargs,
        )

    le = less_than_or_equal_to

    @classmethod
    @st.register_check_strategy(st.in_range_strategy)
    @register_check_statistics(
        ["min_value", "max_value", "include_min", "include_max"]
    )
    def in_range(
        cls, min_value, max_value, include_min=True, include_max=True, **kwargs
    ) -> "Check":
        """Ensure all values of a series are within an interval.

        :param min_value: Left / lower endpoint of the interval.
        :param max_value: Right / upper endpoint of the interval. Must not be
            smaller than min_value.
        :param include_min: Defines whether min_value is also an allowed value
            (the default) or whether all values must be strictly greater than
            min_value.
        :param include_max: Defines whether min_value is also an allowed value
            (the default) or whether all values must be strictly smaller than
            max_value.
        :param kwargs: key-word arguments passed into the `Check` initializer.

        Both endpoints must be a type comparable to the dtype of the
        :class:`pandas.Series` to be validated.

        :returns: :class:`Check` object
        """
        if min_value is None:
            raise ValueError("min_value must not be None")
        if max_value is None:
            raise ValueError("max_value must not be None")
        if max_value < min_value or (
            min_value == max_value and (not include_min or not include_max)
        ):
            raise ValueError(
                f"The combination of min_value = {min_value} and max_value = {max_value} "
                "defines an empty interval!"
            )
        # Using functions from operator module to keep conditions out of the
        # closure
        left_op = operator.le if include_min else operator.lt
        right_op = operator.ge if include_max else operator.gt

        def _in_range(series: pd.Series) -> pd.Series:
            """Comparison function for check"""
            return left_op(min_value, series) & right_op(max_value, series)

        return cls(
            _in_range,
            name=cls.in_range.__name__,
            error=f"in_range({min_value}, {max_value})",
            **kwargs,
        )

    @classmethod
    @st.register_check_strategy(st.isin_strategy)
    @register_check_statistics(["allowed_values"])
    def isin(cls, allowed_values: Iterable, **kwargs) -> "Check":
        """Ensure only allowed values occur within a series.

        :param allowed_values: The set of allowed values. May be any iterable.
        :param kwargs: key-word arguments passed into the `Check` initializer.

        :returns: :class:`Check` object

        .. note::
            It is checked whether all elements of a :class:`pandas.Series`
            are part of the set of elements of allowed values. If allowed
            values is a string, the set of elements consists of all distinct
            characters of the string. Thus only single characters which occur
            in allowed_values at least once can meet this condition. If you
            want to check for substrings use :func:`Check.str_is_substring`.
        """
        # Turn allowed_values into a set. Not only for performance but also
        # avoid issues with a mutable argument passed by reference which may be
        # changed from outside.
        try:
            allowed_values = frozenset(allowed_values)
        except TypeError as exc:
            raise ValueError(
                f"Argument allowed_values must be iterable. Got {allowed_values}"
            ) from exc

        def _isin(series: pd.Series) -> pd.Series:
            """Comparison function for check"""
            return series.isin(allowed_values)

        return cls(
            _isin,
            name=cls.isin.__name__,
            error=f"isin({set(allowed_values)})",
            **kwargs,
        )

    @classmethod
    @st.register_check_strategy(st.notin_strategy)
    @register_check_statistics(["forbidden_values"])
    def notin(cls, forbidden_values: Iterable, **kwargs) -> "Check":
        """Ensure some defined values don't occur within a series.

        :param forbidden_values: The set of values which should not occur. May
            be any iterable.
        :param raise_warning: if True, check raises UserWarning instead of
            SchemaError on validation.

        :returns: :class:`Check` object

        .. note::
            Like :func:`Check.isin` this check operates on single characters if
            it is applied on strings. A string as paraforbidden_valuesmeter
            forbidden_values is understood as set of prohibited characters. Any
            string of length > 1 can't be in it by design.
        """
        # Turn forbidden_values into a set. Not only for performance but also
        # avoid issues with a mutable argument passed by reference which may be
        # changed from outside.
        try:
            forbidden_values = frozenset(forbidden_values)
        except TypeError as exc:
            raise ValueError(
                f"Argument forbidden_values must be iterable. Got {forbidden_values}"
            ) from exc

        def _notin(series: pd.Series) -> pd.Series:
            """Comparison function for check"""
            return ~series.isin(forbidden_values)

        return cls(
            _notin,
            name=cls.notin.__name__,
            error=f"notin({set(forbidden_values)})",
            **kwargs,
        )

    @classmethod
    @st.register_check_strategy(st.str_matches_strategy)
    @register_check_statistics(["pattern"])
    def str_matches(cls, pattern: str, **kwargs) -> "Check":
        """Ensure that string values match a regular expression.

        :param pattern: Regular expression pattern to use for matching
        :param kwargs: key-word arguments passed into the `Check` initializer.

        :returns: :class:`Check` object

        The behaviour is as of :func:`pandas.Series.str.match`.
        """
        # By compiling the regex we get the benefit of an early argument check
        try:
            regex = re.compile(pattern)
        except TypeError as exc:
            raise ValueError(
                f'pattern="{pattern}" cannot be compiled as regular expression'
            ) from exc

        def _match(series: pd.Series) -> pd.Series:
            """
            Check if all strings in the series match the regular expression.
            """
            return series.str.match(regex, na=False)

        return cls(
            _match,
            name=cls.str_matches.__name__,
            error=f"str_matches({regex})",
            **kwargs,
        )

    @classmethod
    @st.register_check_strategy(st.str_contains_strategy)
    @register_check_statistics(["pattern"])
    def str_contains(cls, pattern: str, **kwargs) -> "Check":
        """Ensure that a pattern can be found within each row.

        :param pattern: Regular expression pattern to use for searching
        :param kwargs: key-word arguments passed into the `Check` initializer.

        :returns: :class:`Check` object

        The behaviour is as of :func:`pandas.Series.str.contains`.
        """
        # By compiling the regex we get the benefit of an early argument check
        try:
            regex = re.compile(pattern)
        except TypeError as exc:
            raise ValueError(
                f'pattern="{pattern}" cannot be compiled as regular expression'
            ) from exc

        def _contains(series: pd.Series) -> pd.Series:
            """Check if a regex search is successful within each value"""
            return series.str.contains(regex, na=False)

        return cls(
            _contains,
            name=cls.str_contains.__name__,
            error=f"str_contains({regex})",
            **kwargs,
        )

    @classmethod
    @st.register_check_strategy(st.str_startswith_strategy)
    @register_check_statistics(["string"])
    def str_startswith(cls, string: str, **kwargs) -> "Check":
        """Ensure that all values start with a certain string.

        :param string: String all values should start with
        :param kwargs: key-word arguments passed into the `Check` initializer.

        :returns: :class:`Check` object
        """

        def _startswith(series: pd.Series) -> pd.Series:
            """Returns true only for strings starting with string"""
            return series.str.startswith(string, na=False)

        return cls(
            _startswith,
            name=cls.str_startswith.__name__,
            error=f"str_startswith({string})",
            **kwargs,
        )

    @classmethod
    @st.register_check_strategy(st.str_endswith_strategy)
    @register_check_statistics(["string"])
    def str_endswith(cls, string: str, **kwargs) -> "Check":
        """Ensure that all values end with a certain string.

        :param string: String all values should end with
        :param kwargs: key-word arguments passed into the `Check` initializer.

        :returns: :class:`Check` object
        """

        def _endswith(series: pd.Series) -> pd.Series:
            """Returns true only for strings ending with string"""
            return series.str.endswith(string, na=False)

        return cls(
            _endswith,
            name=cls.str_endswith.__name__,
            error=f"str_endswith({string})",
            **kwargs,
        )

    @classmethod
    @st.register_check_strategy(st.str_length_strategy)
    @register_check_statistics(["min_value", "max_value"])
    def str_length(
        cls, min_value: int = None, max_value: int = None, **kwargs
    ) -> "Check":
        """Ensure that the length of strings is within a specified range.

        :param min_value: Minimum length of strings (default: no minimum)
        :param max_value: Maximum length of strings (default: no maximum)
        :param kwargs: key-word arguments passed into the `Check` initializer.

        :returns: :class:`Check` object
        """
        if min_value is None and max_value is None:
            raise ValueError(
                "At least a minimum or a maximum need to be specified. Got "
                "None."
            )
        if max_value is None:

            def _str_length(series: pd.Series) -> pd.Series:
                """Check for the minimum string length"""
                return series.str.len() >= min_value

        elif min_value is None:

            def _str_length(series: pd.Series) -> pd.Series:
                """Check for the maximum string length"""
                return series.str.len() <= max_value

        else:

            def _str_length(series: pd.Series) -> pd.Series:
                """Check for both, minimum and maximum string length"""
                return (series.str.len() <= max_value) & (
                    series.str.len() >= min_value
                )

        return cls(
            _str_length,
            name=cls.str_length.__name__,
            error=f"str_length({min_value}, {max_value})",
            **kwargs,
        )
