import typing_inspect
from typing import List, Dict, TypedDict, Tuple, NamedTuple, Set

from pyspark.sql.types import MapType, StringType

import pandera as pa
import pandas as pd


class CustomDict(TypedDict):
    foo: str
    bar: int
    baz: float


class CustomTuple(NamedTuple):
    foo: str
    bar: int
    baz: float


TupleType = Tuple[int, ...]
SetType = Set[str]


schema = pa.DataFrameSchema(
    {
        "x": pa.Column(Dict[str, str]),
        "y": pa.Column(List[int]),
        "z": pa.Column(TupleType),
        "a": pa.Column(CustomDict),
        "c": pa.Column(CustomTuple),
    }
)

data = pd.DataFrame(
    {
        "x": [{"1": "bar"}, {"2": "foo"}],
        "y": [[1, 2], [3]],
        "z": [(1, 2), (3, 4)],
        "a": [{"foo": "1", "bar": 1, "baz": 1.0}, {"foo": "1", "bar": 1, "baz": 1.0}],
        "c": [CustomTuple("a", 1, 1.0), CustomTuple("b", 2, 2.0)]
    }
)

validated = schema(data)
print(validated)
# print(validated["z"].map(lambda x: [type(i) for i in x]))
