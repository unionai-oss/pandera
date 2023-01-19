"""Data validation base check."""

import inspect
from collections import namedtuple
from functools import wraps
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Type,
    TypeVar,
    Union,
    no_type_check,
)

import pandas as pd

from pandera.backends.base import BaseCheckBackend

CheckResult = namedtuple(
    "CheckResult",
    ["check_output", "check_passed", "checked_object", "failure_cases"],
)


GroupbyObject = Union[
    pd.core.groupby.generic.SeriesGroupBy,
    pd.core.groupby.generic.DataFrameGroupBy,
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


_T = TypeVar("_T", bound="BaseCheck")


class MetaCheck(type):  # pragma: no cover
    """Check metaclass."""

    BACKEND_REGISTRY: Dict[Type, Type[BaseCheckBackend]] = {}  # noqa
    CHECK_FUNCTION_REGISTRY: Dict[str, Callable] = {}  # noqa
    CHECK_REGISTRY: Dict[str, Callable] = {}  # noqa
    REGISTERED_CUSTOM_CHECKS: Dict[str, Callable] = {}  # noqa

    def __getattr__(cls, name: str) -> Any:
        """Prevent attribute errors for registered checks."""
        attr = {
            **cls.__dict__,
            **cls.CHECK_REGISTRY,
            **cls.REGISTERED_CUSTOM_CHECKS,
        }.get(name)
        if attr is None:
            raise AttributeError(
                f"'{cls}' object has no attribute '{name}'. "
                "Make sure any custom checks have been registered "
                "using the extensions api."
            )
        return attr

    def __dir__(cls) -> Iterable[str]:
        """Allow custom checks to show up as attributes when autocompleting."""
        return chain(
            super().__dir__(),
            cls.CHECK_REGISTRY.keys(),
            cls.REGISTERED_CUSTOM_CHECKS.keys(),
        )

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


class BaseCheck(metaclass=MetaCheck):
    """Check base class."""

    def __init__(
        self,
        name: Optional[str] = None,
        error: Optional[str] = None,
    ):
        self.name = name
        self.error = error

    @classmethod
    def register_backend(cls, type_: Type, backend: Type[BaseCheckBackend]):
        """Register a backend for the specified type."""
        cls.BACKEND_REGISTRY[type_] = backend

    @classmethod
    def get_backend(cls, check_obj: Any) -> Type[BaseCheckBackend]:
        """Get the backend associated with the type of ``check_obj`` ."""
        return cls.BACKEND_REGISTRY[type(check_obj)]

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
