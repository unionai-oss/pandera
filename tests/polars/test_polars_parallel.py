"""Test parallelization with polars using joblib."""

import polars as pl
import pytest
from joblib import Parallel, delayed

from pandera.polars import Column, DataFrameSchema

try:
    import narwhals  # noqa: F401

    narwhals_installed = True
except ImportError:
    narwhals_installed = False

schema = DataFrameSchema({"a": Column(pl.Int32)}, coerce=True)


@pytest.mark.xfail(
    condition=narwhals_installed,
    reason="Narwhals backend does not support dtype coercion (schema uses coerce=True)",
    strict=True,
)
def test_polars_parallel():
    def fn():
        return schema.validate(pl.DataFrame({"a": [1]}))

    # Use threads to avoid loky process spawn issues on Windows CI (TerminatedWorkerError)
    results = Parallel(2, prefer="threads")([delayed(fn)() for _ in range(10)])
    assert len(results) == 10
    for result in results:
        assert result.schema["a"] == pl.Int32
