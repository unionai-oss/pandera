"""Base type definitions for pandera."""

from typing import Union

try:
    # python 3.8+
    from typing import Literal  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    from typing_extensions import Literal  # type: ignore[misc]


StrictType = Union[bool, Literal["filter"]]
