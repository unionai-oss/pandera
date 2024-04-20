"""Extensions module."""

import inspect
import warnings
from enum import Enum
from functools import partial, wraps
from inspect import signature
from typing import Callable, List, Optional, Tuple, Type, Union

import pandas as pd
import typing_inspect

from pandera.api.checks import Check
from pandera.api.hypotheses import Hypothesis
from pandera.strategies.base_strategies import STRATEGY_DISPATCHER

try:
    import pyspark.sql as ps

    PYSPARK_INSTALLED = True
except ImportError:  # pragma: no cover
    PYSPARK_INSTALLED = False


class BuiltinCheckRegistrationError(Exception):
    """
    Exception raised when registering a built-in check implementation but the
    default check function implementation hasn't been registered with
    :py:meth:`~flytekit.core.base.BaseCheck.register_builtin_check_fn`.
    """


# pylint: disable=too-many-locals
def register_builtin_check(
    fn=None,
    strategy: Optional[Callable] = None,
    _check_cls: Type = Check,
    **outer_kwargs,
):
    """Register a check method to the Check namespace.

    This is the primary way for extending the Check api to define additional
    built-in checks.
    """

    if fn is None:
        return partial(
            register_builtin_check,
            strategy=strategy,
            _check_cls=_check_cls,
            **outer_kwargs,
        )

    name = fn.__name__

    # see if the check function is already registered
    check_fn = _check_cls.CHECK_FUNCTION_REGISTRY.get(name)
    fn_sig = signature(fn)

    # register the check strategy for this particular check, identified
    # by the check `name`, and the data type of the check function. This
    # supports Union types. Also assume that the data type of the data
    # object to validate is the first argument.
    data_type = [*fn_sig.parameters.values()][0].annotation

    if typing_inspect.get_origin(data_type) is Tuple:
        data_type, *_ = typing_inspect.get_args(data_type)

    if typing_inspect.get_origin(data_type) is Union:
        data_types = typing_inspect.get_args(data_type)
    else:
        data_types = (data_type,)

    if strategy is not None:
        for dt in data_types:
            STRATEGY_DISPATCHER[(name, dt)] = strategy

    if check_fn is None:  # pragma: no cover
        raise BuiltinCheckRegistrationError(
            f"Check '{name}' doesn't have a base check implementation. "
            f"You need to create a stub method in the {_check_cls} class and "
            "then register a base check function implementation with the "
            f"{_check_cls}.register_builtin_check_fn method.\n"
            "See the `pandera.api.base.builtin_checks` and "
            "`pandera.backends.pandas.builtin_checks` modules as an example."
        )

    check_fn.register(fn)  # type: ignore

    return fn


def register_builtin_hypothesis(**kwargs):
    """Register a new hypothesis."""
    return partial(
        register_builtin_check,
        _check_cls=Hypothesis,
        **kwargs,
    )


# --------------------------------
# CUSTOM CHECK REGISTRATION METHOD
# --------------------------------
#
# The `register_check_method` decorator is the legacy method for registering
# custom checks and will slated for deprecation after merging the core
# internals overhaul.


class CheckType(Enum):
    """Check types for registered check methods."""

    VECTORIZED = 1  #: Check applied to a Series or DataFrame
    ELEMENT_WISE = 2  #: Check applied to an element of a Series or DataFrame
    GROUPBY = 3  #: Check applied to dictionary of Series or DataFrames.


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


def register_check_method(  # pylint:disable=too-many-branches
    check_fn=None,
    *,
    statistics: Optional[List[str]] = None,
    supported_types: Optional[Union[type, Tuple, List]] = None,
    check_type: Union[CheckType, str] = "vectorized",
    strategy=None,
):
    """Registers a function as a :class:`~pandera.api.checks.Check` method.

    See the :ref:`user guide<extensions>` for more details.

    :param check_fn: check function to register. The function should take one
        positional argument for the object to validate and additional
        keyword-only arguments for the check statistics.
    :param statistics: list of keyword-only arguments in the ``check_fn``,
        which serve as the statistics needed to serialize/de-serialize the
        check and generate data if a ``strategy`` function is provided.
    :param supported_types: the pandas type(s) supported by the check function.
        Valid values are ``pd.DataFrame``, ``pd.Series``, ``ps.DataFrame``, or a list/tuple of
        ``(pa.DataFrame, pa.Series, ps.DataFrame)`` if both types are supported.
        Valid values are ``pd.DataFrame``, ``pd.Series``, ``ps.DataFrame``, or a list/tuple of
        ``(pa.DataFrame, pa.Series, ps.DataFrame)`` if both types are supported.
    :param check_type: the expected input of the check function. Valid values
        are :class:`~pandera.extensions.CheckType` enums or
        ``{"vectorized", "element_wise", "groupby"}``. The input signature of
        ``check_fn`` is determined by this argument:

        - if ``vectorized``, the first positional argument of ``check_fn``
          should be one of the ``supported_types``.
        - if ``element_wise``, the first positional argument of ``check_fn``
          should be a single scalar element in the pandas Series or DataFrame.
        - if ``groupby``, the first positional argument of ``check_fn`` should
          be a dictionary mapping group names to subsets of the Series or
          DataFrame.

    :param strategy: data-generation strategy associated with the check
        function.
    :return: register check function wrapper.
    """

    # pylint: disable=import-outside-toplevel
    from pandera.strategies.pandas_strategies import register_check_strategy

    if statistics is None:
        statistics = []

    if isinstance(check_type, str):
        check_type = CheckType[check_type.upper()]

    msg = (
        "{} is not a valid input type for check_fn. You must specify one of "
        "pandas.DataFrame, pandas.Series, or a tuple of both."
    )

    if supported_types is None:
        supported_types = [pd.DataFrame, pd.Series]
        if PYSPARK_INSTALLED:
            supported_types.append(ps.DataFrame)

    if isinstance(supported_types, list):
        supported_types = tuple(supported_types)
    elif not isinstance(supported_types, tuple):
        supported_types = (supported_types,)

    ALLOWED_TYPES = (
        {pd.DataFrame, pd.Series, ps.DataFrame}
        if PYSPARK_INSTALLED
        else {pd.DataFrame, pd.Series}
    )
    for supported_type in supported_types:  # type: ignore
        if supported_type not in ALLOWED_TYPES:
            raise TypeError(msg.format(supported_type))

    if check_type is CheckType.ELEMENT_WISE and set(supported_types) != ALLOWED_TYPES:  # type: ignore
        raise ValueError(
            "Element-wise checks should support DataFrame and Series "
            "validation. Use the default setting for the 'supported_types' "
            "argument."
        )

    if check_fn is None:
        return partial(
            register_check_method,
            statistics=statistics,
            supported_types=supported_types,
            check_type=check_type,
            strategy=strategy,
        )
    else:
        sig = signature(check_fn)
        for statistic in statistics:
            if statistic not in sig.parameters:
                raise TypeError(
                    f"statistic '{statistic}' is not part of "
                    f"{check_fn.__name__}'s signature."
                )

    def register_check_wrapper(check_fn: Callable):
        """Register a function as a :class:`~pandera.api.checks.Check` method."""

        if hasattr(Check, check_fn.__name__):
            raise ValueError(
                f"method with name '{check_fn.__name__}' already defined. "
                "Check methods must have a unique method name."
            )

        @wraps(check_fn)
        def check_fn_wrapper(validate_obj, **kwargs):
            """Wrapper for check_fn to validate inputs."""
            return check_fn(validate_obj, **kwargs)

        def validate_check_kwargs(check_kwargs):
            msg = (
                f"'{check_fn.__name__} has check_type={check_type}. "
                "Providing the following arguments will have no effect: "
                "{}. Remove these arguments to avoid this warning."
            )

            no_effect_args = {
                CheckType.ELEMENT_WISE: ["element_wise", "groupby", "groups"],
                CheckType.VECTORIZED: ["element_wise", "groupby", "groups"],
                CheckType.GROUPBY: ["element_wise"],
            }[check_type]

            if any(arg in check_kwargs for arg in no_effect_args):
                warnings.warn(msg.format(no_effect_args))
                for arg in no_effect_args:
                    check_kwargs.pop(arg, None)

            if check_type is CheckType.ELEMENT_WISE:
                check_kwargs["element_wise"] = True

            return check_kwargs

        @register_check_statistics(statistics)
        def check_method(cls, *args, **kwargs):
            """Wrapper function that serves as the Check method."""
            stats, check_kwargs = {}, {}

            if args:
                stats = dict(zip(statistics, args))

            for k, v in kwargs.items():
                if k in statistics:
                    stats[k] = v
                else:
                    check_kwargs[k] = v

            error_stats = ", ".join(f"{k}={v}" for k, v in stats.items())
            error = f"{check_fn.__name__}({error_stats})" if stats else None

            return cls(
                partial(check_fn_wrapper, **stats),
                name=check_fn.__name__,
                error=error,
                **validate_check_kwargs(check_kwargs),
            )

        if strategy is not None:
            check_method = register_check_strategy(strategy)(check_method)

        Check.REGISTERED_CUSTOM_CHECKS[check_fn.__name__] = partial(
            check_method, Check
        )
        return check_fn

    return register_check_wrapper(check_fn)
