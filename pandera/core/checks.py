"""Data validation check definition."""

from functools import partial, wraps
from inspect import Parameter, Signature, _empty, signature
from typing import Any, Callable, Dict, List, Optional, Type, Union

import pandas as pd
from multimethod import multidispatch

from pandera import errors
from pandera.core.base.checks import BaseCheck, CheckResult
from pandera.strategies import SearchStrategy


class Check(BaseCheck):
    """Check a pandas Series or DataFrame for certain properties."""

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
        statistics: Dict[str, Any] = None,
        strategy: Optional[SearchStrategy] = None,
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
        self.groups: Optional[List[str]] = groups

        self.statistics = statistics or {}
        self.statistics_args = [*self.statistics.keys()]
        self.strategy = strategy

    def __call__(
        self,
        check_obj: Union[pd.DataFrame, pd.Series],
        column: Optional[str] = None,
    ) -> CheckResult:
        # pylint: disable=too-many-branches
        """Validate pandas DataFrame or Series.

        :param check_obj: pandas DataFrame of Series to validate.
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
        backend = Check.get_backend(check_obj)(self)
        return backend(check_obj, column)


def register_check(
    fn=None,
    pre_init_hook: Optional[Callable] = None,
    aliases: Optional[List[str]] = None,
    strategy: Optional[Callable] = None,
    error: Optional[Union[str, Callable]] = None,
    **kwargs,
):
    """Register a check method to the Check namespace.

    This is the primary way for extending the Check api to define additional
    built-in checks.
    """
    if fn is None:
        return partial(
            register_check,
            pre_init_hook=pre_init_hook,
            aliases=aliases,
            strategy=strategy,
            error=error,
        )

    name = fn.__name__
    check_fn = Check.CHECK_FUNCTION_REGISTRY.get(name)

    fn_sig = signature(fn)
    statistics = [*fn_sig.parameters.keys()][1:]
    statistics_params = [*fn_sig.parameters.values()][1:]
    statistics_defaults = {
        p.name: p.default
        for p in fn_sig.parameters.values()
        if p.default is not _empty
    }

    if check_fn is None:

        dispatch_check_fn = multidispatch(fn)

        # create proxy function so we can modify the signature and docstring
        # of the method to reflect correctly in the documentation
        def check_function_proxy(cls, *args, **kwargs):
            return dispatch_check_fn(*args)

        update_check_fn_proxy(
            check_function_proxy, fn, fn_sig, statistics_params
        )

        @wraps(check_function_proxy)
        def check_method(cls, *args, **check_kwargs):
            statistics_kwargs = dict(zip(statistics, args))
            for stat in statistics:
                if stat in check_kwargs:
                    statistics_kwargs[stat] = check_kwargs.pop(stat)
                elif stat not in statistics_kwargs:
                    statistics_kwargs[stat] = statistics_defaults[stat]

            _check_kwargs = {
                "error": (
                    error.format(**statistics_kwargs)
                    if isinstance(error, str)
                    else error(**statistics_kwargs)
                )
            }
            _check_kwargs.update(kwargs)
            _check_kwargs.update(check_kwargs)

            # This is kind of ugly... basically we're creating another
            # stats kwargs variable that's actually used when invoking the check
            # function (which may or may not be processed by pre_init_hook)
            # This is so that the original value is not modified by
            # pre_init_hook when, for e.g. the check is serialized with the io
            # module. Figure out a better way to do this!
            check_fn_stats_kwargs = (
                pre_init_hook(statistics_kwargs)
                if pre_init_hook is not None
                else statistics_kwargs
            )

            # internal wrapper is needed here to make sure the inner check_fn
            # produced by this method is consistent with the registered check
            # function
            @wraps(fn)
            def _check_fn(check_obj, **inner_kwargs):
                """inner_kwargs will be based in via Check.__init__ kwargs."""
                # Raise more informative error when this fails to dispatch
                return check_function_proxy(
                    cls,
                    check_obj,
                    *check_fn_stats_kwargs.values(),
                    **inner_kwargs,
                )

            return cls(
                _check_fn,
                statistics=statistics_kwargs,
                strategy=(
                    None
                    if strategy is None
                    else partial(strategy, **statistics_kwargs)
                ),
                **_check_kwargs,
            )

        Check.CHECK_FUNCTION_REGISTRY[name] = dispatch_check_fn
        setattr(Check, name, classmethod(check_method))

        class_check_method = getattr(Check, name)

        for _name in [] if aliases is None else aliases:
            setattr(Check, _name, class_check_method)
    else:
        check_fn.register(fn)  # type: ignore

    return getattr(Check, name)


def generate_check_signature(
    signature: Signature,
    statistics_params: List[Parameter],
) -> Signature:
    # assume the first argument is the check object
    return signature.replace(
        parameters=[
            # This first parameter will be ignored when
            Parameter("_", Parameter.POSITIONAL_OR_KEYWORD),
            Parameter("cls", Parameter.POSITIONAL_OR_KEYWORD),
            *statistics_params,
            Parameter(
                "kwargs", Parameter.VAR_KEYWORD, annotation=Dict[str, Any]
            ),
        ],
        return_annotation=Check,
    )


def generate_check_annotations(
    statistics_params: List[Parameter],
) -> Dict[str, Type]:
    return {
        **{p.name: p.annotation for p in statistics_params},
        "kwargs": Dict[
            str,
            Any,
        ],
        "return": Check,
    }


def modify_check_fn_doc(doc: str) -> str:
    """Adds"""
    return (
        f"{doc}\n{' ' * 4}:param kwargs: arguments forwarded to the "
        ":py:class:`~pandera.core.checks.Check` constructor."
    )


def update_check_fn_proxy(check_function_proxy, fn, fn_sig, statistics_params):
    """
    Manually update the signature of `check_function` so that docstring matches
    original function's signature, but includes **kwargs, etc.
    """
    check_function_proxy.__name__ = fn.__name__
    check_function_proxy.__module__ = fn.__module__
    check_function_proxy.__qualname__ = fn.__qualname__
    check_function_proxy.__signature__ = generate_check_signature(
        fn_sig, statistics_params
    )
    check_function_proxy.__doc__ = modify_check_fn_doc(fn.__doc__)
    check_function_proxy.__annotations__ = generate_check_annotations(
        statistics_params
    )
