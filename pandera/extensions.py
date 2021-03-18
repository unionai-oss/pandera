"""pandera API extensions

*new in 0.6.0*

This module provides utilities for extending the ``pandera`` API.
"""

import warnings
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd

from . import strategies as st
from .checks import Check, register_check_statistics


class CheckType(Enum):
    """Check types for registered check methods."""

    VECTORIZED = 1  #: Check applied to a Series or DataFrame
    ELEMENT_WISE = 2  #: Check applied to an element of a Series or DataFrame
    GROUPBY = 3  #: Check applied to dictionary of Series or DataFrames.


def register_check_method(
    check_fn=None,
    *,
    statistics: Optional[List[str]] = None,
    supported_types: Union[type, Tuple, List] = (pd.DataFrame, pd.Series),
    check_type: Union[CheckType, str] = "vectorized",
    strategy=None,
):
    """Registers a function as a :class:`~pandera.checks.Check` method.

    See the :ref:`user guide<extensions>` for more details.

    :param check_fn: check function to register. The function should take one
        positional argument for the object to validate and additional
        keyword-only arguments for the check statistics.
    :param statistics: list of keyword-only arguments in the ``check_fn``,
        which serve as the statistics needed to serialize/de-serialize the
        check and generate data if a ``strategy`` function is provided.
    :param supported_types: the pandas type(s) supported by the check function.
        Valid values are ``pd.DataFrame``, ``pd.Series``, or a list/tuple of
        ``(pa.DataFrame, pa.Series)`` if both types are supported.
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

    if statistics is None:
        statistics = []

    if isinstance(check_type, str):
        check_type = CheckType[check_type.upper()]

    msg = (
        "{} is not a valid input type for check_fn. You must specify one of "
        "pandas.DataFrame, pandas.Series, or a tuple of both."
    )
    if isinstance(supported_types, list):
        supported_types = tuple(supported_types)
    elif not isinstance(supported_types, tuple):
        supported_types = (supported_types,)

    for supported_type in supported_types:  # type: ignore
        if supported_type not in {pd.DataFrame, pd.Series}:
            raise TypeError(msg.format(supported_type))

    if check_type is CheckType.ELEMENT_WISE and set(supported_types) != {
        pd.DataFrame,
        pd.Series,
    }:  # type: ignore
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

    def register_check_wrapper(check_fn: Callable):
        """Register a function as a :class:`~pandera.checks.Check` method."""

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

            return cls(
                partial(check_fn_wrapper, **stats),
                name=check_fn.__name__,
                **validate_check_kwargs(check_kwargs),
            )

        if strategy is not None:
            check_method = st.register_check_strategy(strategy)(check_method)

        Check.REGISTERED_CUSTOM_CHECKS[check_fn.__name__] = partial(
            check_method, Check
        )

    return register_check_wrapper(check_fn)
