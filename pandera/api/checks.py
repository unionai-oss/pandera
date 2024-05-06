"""Data validation check definition."""

import re
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union,
)

from pandera import errors
from pandera.api.base.checks import BaseCheck, CheckResult
from pandera.strategies.base_strategies import SearchStrategy

T = TypeVar("T")


# pylint: disable=too-many-public-methods
class Check(BaseCheck):
    """Check a data object for certain properties."""

    def __init__(
        self,
        check_fn: Callable,
        groups: Optional[Union[str, List[str]]] = None,
        groupby: Optional[Union[str, List[str], Callable]] = None,
        ignore_na: bool = True,
        element_wise: bool = False,
        name: Optional[str] = None,
        error: Optional[str] = None,
        raise_warning: bool = False,
        n_failure_cases: Optional[int] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        statistics: Optional[Dict[str, Any]] = None,
        strategy: Optional[SearchStrategy] = None,
        **check_kwargs,
    ) -> None:
        """Apply a validation function to a data object.

        :param check_fn: A function to check data object. For Column
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
        :param raise_warning: if True, raise a SchemaWarning and do not throw
            exception instead of raising a SchemaError for a specific check.
            This option should be used carefully in cases where a failing
            check is informational and shouldn't stop execution of the program.
        :param n_failure_cases: report the first n unique failure cases. If
            None, report all failure cases.
        :param title: A human-readable label for the check.
        :param description: An arbitrary textual description of the check.
        :param statistics: kwargs to pass into the check function. These values
            are serialized and represent the constraints of the checks.
        :param strategy: A hypothesis strategy, used for implementing data
            synthesis strategies for this check. See the
            :ref:`User Guide <custom-strategies>` for more details.
        :param check_kwargs: key-word arguments to pass into ``check_fn``

        :example:

        The example below uses ``pandas``, but will apply to any of the supported
        :ref:`dataframe libraries <dataframe-libraries>`.

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
        super().__init__(name=name, error=error)
        if element_wise and groupby is not None:
            raise errors.SchemaInitError(
                "Cannot use groupby when element_wise=True."
            )
        self._check_fn = check_fn
        self._check_kwargs = check_kwargs
        self.element_wise = element_wise
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
        self.groups: Optional[List[str]] = groups

        self.statistics = statistics or check_kwargs or {}
        self.statistics_args = [*self.statistics.keys()]
        self.strategy = strategy

    def __call__(
        self,
        check_obj: Any,
        column: Optional[str] = None,
    ) -> CheckResult:
        # pylint: disable=too-many-branches
        """Validate DataFrame or Series.

        :param check_obj: DataFrame of Series to validate.
        :param column: for dataframe checks, apply the check function to this
            column.
        :returns: CheckResult tuple containing:

            ``check_output``: boolean scalar, ``Series`` or ``DataFrame``
            indicating which elements passed the check.

            ``check_passed``: boolean scalar that indicating whether the check
            passed overall.

            ``checked_object``: the checked object itself. Depending on the
            options provided to the ``Check``, this will be a Series,
            DataFrame, or if the ``groupby`` option is supported by the validation
            backend and specified, a ``Dict[str, Series]`` or ``Dict[str, DataFrame]``
            where the keys are distinct groups.

            ``failure_cases``: subset of the check_object that failed.
        """
        backend = self.get_backend(check_obj)(self)
        return backend(check_obj, column)

    @classmethod
    def equal_to(cls, value: Any, **kwargs) -> "Check":
        """Ensure all elements of a data container equal a certain value.

        :param value: values in this data object must be
            equal to this value.
        """
        return cls.from_builtin_check_name(
            "equal_to",
            kwargs,
            error=f"equal_to({value})",
            value=value,
        )

    @classmethod
    def not_equal_to(cls, value: Any, **kwargs) -> "Check":
        """Ensure no elements of a data container equals a certain value.

        :param value: This value must not occur in the data object.
        """
        return cls.from_builtin_check_name(
            "not_equal_to",
            kwargs,
            error=f"not_equal_to({value})",
            value=value,
        )

    @classmethod
    def greater_than(cls, min_value: Any, **kwargs) -> "Check":
        """
        Ensure values of a data container are strictly greater than a minimum
        value.

        :param min_value: Lower bound to be exceeded. Must be a type comparable
            to the dtype of the data object to be validated (e.g. a
            numerical type for float or int and a datetime for datetime).
        """
        if min_value is None:
            raise ValueError("min_value must not be None")
        return cls.from_builtin_check_name(
            "greater_than",
            kwargs,
            error=f"greater_than({min_value})",
            min_value=min_value,
        )

    @classmethod
    def greater_than_or_equal_to(cls, min_value: Any, **kwargs) -> "Check":
        """Ensure all values are greater or equal a certain value.

        :param min_value: Allowed minimum value for values of the data. Must be
            a type comparable to the dtype of the data object to be
            validated.
        """
        if min_value is None:
            raise ValueError("min_value must not be None")
        return cls.from_builtin_check_name(
            "greater_than_or_equal_to",
            kwargs,
            error=f"greater_than_or_equal_to({min_value})",
            min_value=min_value,
        )

    @classmethod
    def less_than(cls, max_value: Any, **kwargs) -> "Check":
        """Ensure values of a series are strictly below a maximum value.

        :param max_value: All elements of a series must be strictly smaller
            than this. Must be a type comparable to the dtype of the
            data object to be validated.
        """
        if max_value is None:
            raise ValueError("max_value must not be None")
        return cls.from_builtin_check_name(
            "less_than",
            kwargs,
            error=f"less_than({max_value})",
            max_value=max_value,
        )

    @classmethod
    def less_than_or_equal_to(cls, max_value: Any, **kwargs) -> "Check":
        """Ensure values of a series are strictly below a maximum value.

        :param max_value: Upper bound not to be exceeded. Must be a type
            comparable to the dtype of the data object to be
            validated.
        """
        if max_value is None:
            raise ValueError("max_value must not be None")
        return cls.from_builtin_check_name(
            "less_than_or_equal_to",
            kwargs,
            error=f"less_than_or_equal_to({max_value})",
            max_value=max_value,
        )

    @classmethod
    def in_range(
        cls,
        min_value: T,
        max_value: T,
        include_min: bool = True,
        include_max: bool = True,
        **kwargs,
    ) -> "Check":
        """Ensure all values of a series are within an interval.

        Both endpoints must be a type comparable to the dtype of the
        data object to be validated.

        :param min_value: Left / lower endpoint of the interval.
        :param max_value: Right / upper endpoint of the interval. Must not be
            smaller than min_value.
        :param include_min: Defines whether min_value is also an allowed value
            (the default) or whether all values must be strictly greater than
            min_value.
        :param include_max: Defines whether min_value is also an allowed value
            (the default) or whether all values must be strictly smaller than
            max_value.
        """
        if min_value is None:
            raise ValueError("min_value must not be None")
        if max_value is None:
            raise ValueError("max_value must not be None")
        if max_value < min_value or (  # type: ignore
            min_value == max_value and (not include_min or not include_max)
        ):
            raise ValueError(
                f"The combination of min_value = {min_value} and "
                f"max_value = {max_value} defines an empty interval!"
            )
        return cls.from_builtin_check_name(
            "in_range",
            kwargs,
            error=f"in_range({min_value}, {max_value})",
            min_value=min_value,
            max_value=max_value,
            include_min=include_min,
            include_max=include_max,
        )

    @classmethod
    def isin(cls, allowed_values: Iterable, **kwargs) -> "Check":
        """Ensure only allowed values occur within a series.

        This checks whether all elements of a data object
        are part of the set of elements of allowed values. If allowed
        values is a string, the set of elements consists of all distinct
        characters of the string. Thus only single characters which occur
        in allowed_values at least once can meet this condition. If you
        want to check for substrings use :meth:`Check.str_contains`.

        :param allowed_values: The set of allowed values. May be any iterable.
        :param kwargs: key-word arguments passed into the `Check` initializer.
        """
        try:
            allowed_values_mod = frozenset(allowed_values)
        except TypeError as exc:
            raise ValueError(
                f"Argument allowed_values must be iterable. Got {allowed_values}"
            ) from exc
        return cls.from_builtin_check_name(
            "isin",
            kwargs,
            error=f"isin({allowed_values})",
            statistics={"allowed_values": allowed_values},
            allowed_values=allowed_values_mod,
        )

    @classmethod
    def notin(cls, forbidden_values: Iterable, **kwargs) -> "Check":
        """Ensure some defined values don't occur within a series.

        Like :meth:`Check.isin` this check operates on single characters if
        it is applied on strings. If forbidden_values is a string, it is
        understood as set of prohibited characters. Any string of length > 1
        can't be in it by design.

        :param forbidden_values: The set of values which should not occur. May
            be any iterable.
        :param raise_warning: if True, check raises SchemaWarning instead of
            SchemaError on validation.
        """
        try:
            forbidden_values_mod = frozenset(forbidden_values)
        except TypeError as exc:
            raise ValueError(
                "Argument forbidden_values must be iterable. "
                f"Got {forbidden_values}"
            ) from exc
        return cls.from_builtin_check_name(
            "notin",
            kwargs,
            error=f"notin({forbidden_values})",
            statistics={"forbidden_values": forbidden_values},
            forbidden_values=forbidden_values_mod,
        )

    @classmethod
    def str_matches(cls, pattern: Union[str, re.Pattern], **kwargs) -> "Check":
        """Ensure that strings start with regular expression match.

        :param pattern: Regular expression pattern to use for matching
        :param kwargs: key-word arguments passed into the `Check` initializer.
        """
        try:
            pattern_mod = re.compile(pattern)
        except TypeError as exc:
            raise ValueError(
                f'pattern="{pattern}" cannot be compiled as regular expression'
            ) from exc
        return cls.from_builtin_check_name(
            "str_matches",
            kwargs,
            error=f"str_matches('{pattern}')",
            statistics={"pattern": pattern},
            pattern=pattern_mod,
        )

    @classmethod
    def str_contains(
        cls, pattern: Union[str, re.Pattern], **kwargs
    ) -> "Check":
        """Ensure that a pattern can be found within each row.

        :param pattern: Regular expression pattern to use for searching
        :param kwargs: key-word arguments passed into the `Check` initializer.
        """
        try:
            pattern_mod = re.compile(pattern)
        except TypeError as exc:
            raise ValueError(
                f'pattern="{pattern}" cannot be compiled as regular expression'
            ) from exc
        return cls.from_builtin_check_name(
            "str_contains",
            kwargs,
            error=f"str_contains('{pattern}')",
            statistics={"pattern": pattern},
            pattern=pattern_mod,
        )

    @classmethod
    def str_startswith(cls, string: str, **kwargs) -> "Check":
        """Ensure that all values start with a certain string.

        :param string: String all values should start with
        :param kwargs: key-word arguments passed into the `Check` initializer.
        """

        return cls.from_builtin_check_name(
            "str_startswith",
            kwargs,
            error=f"str_startswith('{string}')",
            string=string,
        )

    @classmethod
    def str_endswith(cls, string: str, **kwargs) -> "Check":
        """Ensure that all values end with a certain string.

        :param string: String all values should end with
        :param kwargs: key-word arguments passed into the `Check` initializer.
        """
        return cls.from_builtin_check_name(
            "str_endswith",
            kwargs,
            error=f"str_endswith('{string}')",
            string=string,
        )

    @classmethod
    def str_length(
        cls,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        **kwargs,
    ) -> "Check":
        """Ensure that the length of strings is within a specified range.

        :param min_value: Minimum length of strings (default: no minimum)
        :param max_value: Maximum length of strings (default: no maximum)
        """
        if min_value is None and max_value is None:
            raise ValueError(
                "At least a minimum or a maximum need to be specified. Got "
                "None."
            )
        return cls.from_builtin_check_name(
            "str_length",
            kwargs,
            error=f"str_length({min_value}, {max_value})",
            min_value=min_value,
            max_value=max_value,
        )

    @classmethod
    def unique_values_eq(cls, values: str, **kwargs) -> "Check":
        """Ensure that unique values in the data object contain all values.

        .. note::
            In constrast with :func:`isin`, this check makes sure that all the
            items in the ``values`` iterable are contained within the series.

        :param values: The set of values that must be present. Maybe any iterable.
        """
        try:
            values_mod = frozenset(values)
        except TypeError as exc:
            raise ValueError(
                f"Argument values must be iterable. Got {values}"
            ) from exc
        return cls.from_builtin_check_name(
            "unique_values_eq",
            kwargs,
            error=f"unique_values_eq({values})",
            statistics={"values": values_mod},
            values=values_mod,
        )

    # Aliases
    # -------

    @classmethod
    def eq(cls, value: Any, **kwargs) -> "Check":
        """Alias of :meth:`~pandera.api.checks.Check.equal_to`"""
        return cls.equal_to(value, **kwargs)

    @classmethod
    def ne(cls, value: Any, **kwargs) -> "Check":
        """Alias of :meth:`~pandera.api.checks.Check.not_equal_to`"""
        return cls.not_equal_to(value, **kwargs)

    @classmethod
    def gt(cls, min_value: Any, **kwargs) -> "Check":
        """Alias of :meth:`~pandera.api.checks.Check.greater_than`"""
        return cls.greater_than(min_value, **kwargs)

    @classmethod
    def ge(cls, min_value: Any, **kwargs) -> "Check":
        """
        Alias of :meth:`~pandera.api.checks.Check.greater_than_or_equal_to`
        """
        return cls.greater_than_or_equal_to(min_value, **kwargs)

    @classmethod
    def lt(cls, max_value: Any, **kwargs) -> "Check":
        """Alias of :meth:`~pandera.api.checks.Check.less_than`"""
        return cls.less_than(max_value, **kwargs)

    @classmethod
    def le(cls, max_value: Any, **kwargs) -> "Check":
        """Alias of :meth:`~pandera.api.checks.Check.less_than_or_equal_to`"""
        return cls.less_than_or_equal_to(max_value, **kwargs)

    @classmethod
    def between(
        cls,
        min_value: T,
        max_value: T,
        include_min: bool = True,
        include_max: bool = True,
        **kwargs,
    ) -> "Check":
        """Alias of :meth:`~pandera.api.checks.Check.in_range`"""
        return cls.in_range(
            min_value,
            max_value,
            include_min,
            include_max,
            **kwargs,
        )
