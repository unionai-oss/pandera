from typing import Annotated

import pandera.polars as pa
import polars as pl

from pandera.typing import Series
from pandera.errors import SchemaInitError


df = pl.DataFrame({
    "id": [1, 2, 3],
    "lists": [["a"], ["a", "b"], ["a", "b", "c"]],
})


class Lists(pa.DataFrameModel):
    """Most basic, expected form given the working schema above."""
    id: int
    lists: list[str]


try:
    Lists.validate(df)
except SchemaInitError as e:
    print("\nLists validation failed")
    print(e)
else:
    print("\nLists validation passed")
