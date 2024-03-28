"""Data validation base parse."""

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

from pandera.api.base.checks import multidispatch
from pandera.backends.base import BaseParserBackend


class ParserResult(NamedTuple):
    """Parser result for user-defined parsers."""

    parser_output: Any
    parsed_object: Any


_T = TypeVar("_T", bound="BaseParser")


class MetaParser(type):
    """Parser metaclass."""

    BACKEND_REGISTRY: Dict[Tuple[Type, Type], Type[BaseParserBackend]] = {}
    """Registry of parser backends implemented for specific data objects."""

    PARSER_FUNCTION_REGISTRY: Dict[str, Callable] = {}
    """Built-in parser function registry"""

    def __getattr__(cls, name: str) -> Any:
        """Prevent attribute errors for registered parsers."""
        attr = {
            **cls.__dict__,
            **cls.PARSER_FUNCTION_REGISTRY,
        }.get(name)
        if attr is None:
            raise AttributeError(
                f"'{cls}' object has no attribute '{name}'. "
                "Make sure any custom parsers have been registered "
                "using the extensions api."
            )
        return attr

    def __dir__(cls) -> Iterable[str]:
        """Allow custom parsers to show up as attributes when autocompleting."""
        return chain(
            super().__dir__(),
            cls.PARSER_FUNCTION_REGISTRY.keys(),
        )

    # pylint: disable=line-too-long
    # mypy has limited metaclass support so this doesn't pass typecheck
    # see https://mypy.readthedocs.io/en/stable/metaclasses.html#gotchas-and-limitations-of-metaclass-support
    # pylint: enable=line-too-long
    @no_type_check
    def __contains__(cls: Type[_T], item: Union[_T, str]) -> bool:
        """Allow lookups for registered parsers."""
        if isinstance(item, cls):
            name = item.name
            return hasattr(cls, name)

        # assume item is str
        return hasattr(cls, item)


class BaseParser(metaclass=MetaParser):
    """Parser base class."""

    def __init__(self, name: Optional[str] = None):
        self.name = name

    @classmethod
    def register_builtin_parser_fn(cls, fn: Callable):
        """Registers a built-in parser function"""
        cls.PARSER_FUNCTION_REGISTRY[fn.__name__] = multidispatch(fn)
        return fn

    @classmethod
    def get_builtin_parser_fn(cls, name: str):
        """Gets a built-in parser function"""
        return cls.PARSER_FUNCTION_REGISTRY[name]

    @classmethod
    def from_builtin_parser_name(cls, name: str, init_kwargs, **parser_kwargs):
        """Create a Parse object from a built-in parse's name."""
        kws = {**init_kwargs, **parser_kwargs}
        return cls(cls.get_builtin_parser_fn(name), **kws)

    @classmethod
    def register_backend(cls, type_: Type, backend: Type[BaseParserBackend]):
        """Register a backend for the specified type."""
        cls.BACKEND_REGISTRY[(cls, type_)] = backend

    @classmethod
    def get_backend(cls, parse_obj: Any) -> Type[BaseParserBackend]:
        """Get the backend associated with the type of ``parse_obj`` ."""

        parse_obj_cls = type(parse_obj)
        classes = inspect.getmro(parse_obj_cls)
        for _class in classes:
            try:
                return cls.BACKEND_REGISTRY[(cls, _class)]
            except KeyError:
                pass
        raise KeyError(
            f"Backend not found for class: {parse_obj_cls}. Looked up the "
            f"following base classes: {classes}"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        are_parser_fn_objects_equal = (
            self._get_parser_fn_code() == other._get_parser_fn_code()
        )

        are_all_other_parser_attributes_equal = {
            k: v for k, v in self.__dict__.items() if k != "_parser_fn"
        } == {k: v for k, v in other.__dict__.items() if k != "_parser_fn"}

        return (
            are_parser_fn_objects_equal
            and are_all_other_parser_attributes_equal
        )

    def _get_parser_fn_code(self):
        parser_fn = self.__dict__["_parser_fn"]
        try:
            code = parser_fn.__code__.co_code
        except AttributeError:
            # try accessing the functools.partial wrapper
            code = parser_fn.func.__code__.co_code

        return code

    def __hash__(self) -> int:
        return hash(self._get_parser_fn_code())

    def __repr__(self) -> str:
        return f"<Parser {self.name}>"
