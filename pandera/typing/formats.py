from enum import Enum
from typing import Literal, Union


class Formats(Enum):
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
