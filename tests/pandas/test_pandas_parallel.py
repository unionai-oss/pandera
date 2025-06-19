"""Test parallelization with pandas using joblib."""

import pandas as pd
from joblib import Parallel, delayed

from pandera.pandas import Column, DataFrameSchema

schema = DataFrameSchema({"a": Column("int64")}, coerce=True)


def test_polars_parallel():
    def fn():
        return schema.validate(pd.DataFrame({"a": [1]}))

    results = Parallel(2)([delayed(fn)() for _ in range(10)])
    assert len(results) == 10
    for result in results:
        assert result.dtypes["a"] == "int64"
