"""Base type definitions for pandera."""

from typing import List, Union

from pandera.api.checks import Check
from pandera.api.hypotheses import Hypothesis
from pandera.api.parsers import Parser

try:
    # python 3.8+
    from typing import Literal  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    from typing_extensions import Literal  # type: ignore[assignment]


StrictType = Union[bool, Literal["filter"]]
CheckList = Union[Check, List[Union[Check, Hypothesis]]]
ParserList = Union[Parser, List[Parser]]
