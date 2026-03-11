"""Test parallelization with polars using joblib."""

import polars as pl
from joblib import Parallel, delayed

from pandera.polars import Column, DataFrameSchema

schema = DataFrameSchema({"a": Column(pl.Int32)}, coerce=True)


def test_polars_parallel():
    def fn():
        return schema.validate(pl.DataFrame({"a": [1]}))

    # Use threads to avoid loky process spawn issues on Windows CI (TerminatedWorkerError)
    results = Parallel(2, prefer="threads")(
        [delayed(fn)() for _ in range(10)]
    )
    assert len(results) == 10
    for result in results:
        assert result.schema["a"] == pl.Int32
