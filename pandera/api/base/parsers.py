"""Data validation base parse."""

import inspect
from typing import Any, Dict, NamedTuple, Optional, Tuple, Type

from pandera.backends.base import BaseParserBackend


class ParserResult(NamedTuple):
    """Parser result for user-defined parsers."""

    parser_output: Any
    parsed_object: Any


class MetaParser(type):
    """Parser metaclass."""

    BACKEND_REGISTRY: Dict[Tuple[Type, Type], Type[BaseParserBackend]] = {}
    """Registry of parser backends implemented for specific data objects."""


class BaseParser(metaclass=MetaParser):
    """Parser base class."""

    def __init__(self, name: Optional[str] = None):
        self.name = name

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
        code = parser_fn.__code__.co_code

        return code

    def __repr__(self) -> str:
        return f"<Parser {self.name}>"
