"""Base type definitions for pandera."""

from typing import Any, Literal, Protocol, Union

from pandera.api.checks import Check
from pandera.api.hypotheses import Hypothesis
from pandera.api.parsers import Parser

StrictType = Union[bool, Literal["filter"]]
CheckList = Union[Check, list[Union[Check, Hypothesis]]]
ParserList = Union[Parser, list[Parser]]


class CheckData(Protocol):
    """Protocol for data wrappers passed to check functions."""

    @property
    def frame(self) -> Any:
        """The native dataframe (LazyFrame, Table, DataFrame, etc.)."""
        ...

    @property
    def key(self) -> str | None:
        """The column name to check, or None for all columns."""
        ...
