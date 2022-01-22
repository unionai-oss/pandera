"""Serialization formats for dataframes."""

from enum import Enum
from typing import Union

try:
    from typing import Literal  # type: ignore
except ImportError:
    from typing_extensions import Literal  # type: ignore


class Formats(Enum):
    """Data container serialization formats."""

    # pylint: disable=invalid-name
    csv = "csv"
    dict = "dict"
    json = "json"
    feather = "feather"
    parquet = "parquet"
    pickle = "pickle"


Format = Union[
    Literal[Formats.csv],
    Literal[Formats.dict],
    Literal[Formats.json],
    Literal[Formats.feather],
    Literal[Formats.parquet],
    Literal[Formats.pickle],
]
