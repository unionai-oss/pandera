"""Model component base classes."""

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    Union,
    cast,
)

from pandera.api.checks import Check
from pandera.api.parsers import Parser

CheckArg = Union[Check, List[Check]]
ParserArg = Union[Parser, List[Parser]]
AnyCallable = Callable[..., Any]


def to_checklist(checks: Optional[CheckArg]) -> List[Check]:
    """Convert value to list of checks."""
    checks = checks or []
    return [checks] if isinstance(checks, Check) else checks


def to_parserlist(parsers: Optional[ParserArg]) -> List[Parser]:
    parsers = parsers or []
    return [parsers] if isinstance(parsers, Parser) else parsers


class BaseFieldInfo:
    """Captures extra information about a field.

    *new in 0.5.0*
    """

    __slots__ = (
        "checks",
        "parses",
        "nullable",
        "unique",
        "coerce",
        "regex",
        "check_name",
        "alias",
        "original_name",
        "dtype_kwargs",
        "title",
        "description",
        "default",
        "metadata",
    )

    def __init__(
        self,
        checks: Optional[CheckArg] = None,
        parses: Optional[ParserArg] = None,
        nullable: bool = False,
        unique: bool = False,
        coerce: bool = False,
        regex: bool = False,
        alias: Any = None,
        check_name: Optional[bool] = None,
        dtype_kwargs: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        default: Optional[Any] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self.checks = to_checklist(checks)
        self.parses = to_parserlist(parses)
        self.nullable = nullable
        self.unique = unique
        self.coerce = coerce
        self.regex = regex
        self.alias = alias
        self.check_name = check_name
        self.original_name = cast(str, None)  # always set by BaseModel
        self.dtype_kwargs = dtype_kwargs
        self.title = title
        self.description = description
        self.default = default
        self.metadata = metadata

    @property
    def name(self) -> str:
        """Return the name of the field used in the data container object."""
        if self.alias is not None:
            return self.alias
        return self.original_name

    def __set_name__(self, owner: Type, name: str) -> None:
        self.original_name = name

    def __get__(self, instance: Any, owner: Type) -> str:
        return self.name

    def __str__(self):
        return f'{self.__class__}("{self.name}")'

    def __repr__(self):
        cls = self.__class__
        return (
            f'<{cls.__module__}.{cls.__name__}("{self.name}") '
            f"object at {hex(id(self))}>"
        )

    def __hash__(self):
        return str(self.name).__hash__()

    def __eq__(self, other):
        return self.name == other

    def __ne__(self, other):
        return self.name != other

    def __set__(self, instance: Any, value: Any) -> None:  # pragma: no cover
        raise AttributeError(f"Can't set the {self.original_name} field.")


class BaseCheckInfo:  # pylint:disable=too-few-public-methods
    """Captures extra information about a Check."""

    def __init__(self, check_fn: AnyCallable, **check_kwargs: Any):
        self.check_fn = check_fn
        self.check_kwargs = check_kwargs

    def to_check(self, model_cls: Type) -> Check:
        """Create a Check from metadata."""
        name = self.check_kwargs.pop("name", None)
        if not name:
            name = getattr(
                self.check_fn, "__name__", self.check_fn.__class__.__name__
            )

        def _adapter(arg: Any) -> Union[bool, Iterable[bool]]:
            return self.check_fn(model_cls, arg)

        return Check(_adapter, name=name, **self.check_kwargs)


class BaseParserInfo:  # pylint:disable=too-few-public-methods
    """Captures extra information about a Parse."""

    def __init__(self, parser_fn: AnyCallable, **parser_kwargs: Any) -> None:
        self.parser_fn = parser_fn
        self.parser_kwargs = parser_kwargs

    def to_parser(self, model_cls: Type) -> Parser:
        """Create a Parser from metadata."""
        name = self.parser_kwargs.pop("name", None)
        if not name:
            name = getattr(
                self.parser_fn, "__name__", self.parser_fn.__class__.__name__
            )

        def _adapter(arg: Any) -> Union[bool, Iterable[bool]]:
            return self.parser_fn(model_cls, arg)

        return Parser(_adapter, name=name, **self.parser_kwargs)
