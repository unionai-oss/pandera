import polars as pl
from pandera.polars import Column, DataFrameSchema
from joblib import Parallel, delayed


def test():

    schema = DataFrameSchema(
        {
            "a": Column(pl.Int32),
        },
        strict=True,
        coerce=True,
        ordered=True,
    )
    schema.validate(pl.DataFrame({"a": [1]}))


Parallel(2)([delayed(test)() for _ in range(10)])
