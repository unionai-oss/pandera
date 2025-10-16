"""Base type definitions for pandera."""

from typing import Literal, Union

from pandera.api.checks import Check
from pandera.api.hypotheses import Hypothesis
from pandera.api.parsers import Parser

StrictType = Union[bool, Literal["filter"]]
CheckList = Union[Check, list[Union[Check, Hypothesis]]]
ParserList = Union[Parser, list[Parser]]
