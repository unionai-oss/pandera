"""Data validation base check."""

import inspect
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    no_type_check,
)

import pandas as pd
from multimethod import multidispatch as _multidispatch

from pandera.backends.base import BaseCheckBackend


class CheckResult(NamedTuple):
    """Check result for user-defined checks."""

    check_output: Any
    check_passed: Any
    checked_object: Any
    failure_cases: Any


GroupbyObject = Union[
    pd.core.groupby.generic.SeriesGroupBy,
    pd.core.groupby.generic.DataFrameGroupBy,
]

SeriesCheckObj = Union[pd.Series, Dict[str, pd.Series]]
DataFrameCheckObj = Union[pd.DataFrame, Dict[str, pd.DataFrame]]


_T = TypeVar("_T", bound="BaseCheck")


# pylint: disable=invalid-name
class multidispatch(_multidispatch):
    """
    Custom multidispatch class to handle copy, deepcopy, and code retrieval.
    """

    @property
    def __code__(self):
        """Retrieves the 'base' function of the multidispatch object."""
        assert (
            len(self) > 0
        ), f"multidispatch object {self} has no functions registered"
        fn, *_ = [*self.values()]  # type: ignore[misc]
        return fn.__code__

    def __reduce__(self):
        """
        Handle custom pickling reduction method by initializing a new
        multidispatch object, wrapped with the base function.
        """
        state = self.__dict__
        # make sure all registered functions at time of pickling are captured
        state["__registered_functions__"] = [*self.values()]
        return (
            multidispatch,  # object creation function
            (state["__wrapped__"],),  # arguments to said function
            state,  # arguments to `__setstate__` after creation
        )

    def __setstate__(self, state):
        """Custom unpickling logic."""
        self.__dict__ = state
        # rehydrate the multidispatch object with unpickled registered functions
        for fn in state["__registered_functions__"]:
            self.register(fn)


class MetaCheck(type):  # pragma: no cover
    """Check metaclass."""

    BACKEND_REGISTRY: Dict[
        Tuple[Type, Type], Type[BaseCheckBackend]
    ] = {}  # noqa
    """Registry of check backends implemented for specific data objects."""

    CHECK_FUNCTION_REGISTRY: Dict[str, Callable] = {}  # noqa
    """Built-in check function registry."""

    REGISTERED_CUSTOM_CHECKS: Dict[str, Callable] = {}  # noqa
    """User-defined custom checks."""

    def __getattr__(cls, name: str) -> Any:
        """Prevent attribute errors for registered checks."""
        attr = {
            **cls.__dict__,
            **cls.CHECK_FUNCTION_REGISTRY,
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
            cls.CHECK_FUNCTION_REGISTRY.keys(),
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
        statistics: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.error = error
        self.statistics = statistics

    @classmethod
    def register_builtin_check_fn(cls, fn: Callable):
        """Registers a built-in check function"""
        cls.CHECK_FUNCTION_REGISTRY[fn.__name__] = multidispatch(fn)
        return fn

    @classmethod
    def get_builtin_check_fn(cls, name: str):
        """Gets a built-in check function"""
        return cls.CHECK_FUNCTION_REGISTRY[name]

    @classmethod
    def from_builtin_check_name(
        cls,
        name: str,
        init_kwargs,
        error: Union[str, Callable],
        statistics: Optional[Dict[str, Any]] = None,
        **check_kwargs,
    ):
        """Create a Check object from a built-in check's name."""
        kws = {**init_kwargs, **check_kwargs}
        if "error" not in kws:
            kws["error"] = error

        # statistics are the raw check constraint values that are untransformed
        # by the check object
        if statistics is None:
            statistics = check_kwargs
        return cls(
            cls.get_builtin_check_fn(name),
            statistics=statistics,
            **kws,
        )

    @classmethod
    def register_backend(cls, type_: Type, backend: Type[BaseCheckBackend]):
        """Register a backend for the specified type."""
        if (cls, type_) not in cls.BACKEND_REGISTRY:
            cls.BACKEND_REGISTRY[(cls, type_)] = backend

    @classmethod
    def get_backend(cls, check_obj: Any) -> Type[BaseCheckBackend]:
        """Get the backend associated with the type of ``check_obj`` ."""

        check_obj_cls = type(check_obj)
        classes = inspect.getmro(check_obj_cls)
        for _class in classes:
            try:
                return cls.BACKEND_REGISTRY[(cls, _class)]
            except KeyError:
                pass
        raise KeyError(
            f"Backend not found for class: {check_obj_cls}. Looked up the "
            f"following base classes: {classes}"
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
