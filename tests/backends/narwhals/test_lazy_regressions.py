"""Regression tests for critical lazy=True bugs (Phase 8)."""

import warnings

import polars as pl
import pytest

from pandera.api.checks import Check
from pandera.api.polars.components import Column
from pandera.api.polars.container import DataFrameSchema
from pandera.errors import SchemaErrors


@pytest.fixture(autouse=True, scope="module")
def _suppress_narwhals_warning():
    """Initialise narwhals backends and suppress the auto-activation UserWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        from pandera.backends.ibis.register import register_ibis_backends
        from pandera.backends.polars.register import register_polars_backends

        register_polars_backends.cache_clear()
        register_ibis_backends.cache_clear()
        register_polars_backends()
        register_ibis_backends()
        yield


def test_lazy_failure_cases_per_row_polars():
    """SchemaErrors.failure_cases has N rows (not 1 repr string) for polars lazy=True.

    # MISSING-01 regression: failure_cases must have N per-row values, not 1 repr string
    # TEST-02: intentionally polars_eager-specific — regression test for pl.DataFrame lazy=True behavior
    """
    schema = DataFrameSchema(
        columns={"a": Column(pl.Int64, checks=[Check.greater_than(10)])}
    )
    # 3 failing rows — failure_cases must have 3 rows, not 1 repr string
    with pytest.raises(SchemaErrors) as exc_info:
        schema.validate(pl.DataFrame({"a": [1, 2, 3]}), lazy=True)
    fc = exc_info.value.failure_cases
    assert len(fc) == 3, f"Expected 3 rows, got {len(fc)}: {fc}"
    assert "failure_case" in fc.columns
    # Values must be individual values (int, float, or numeric string), not a DataFrame repr string.
    # The eager polars path casts failure_case to Utf8; numeric strings like '1', '2', '3'
    # are acceptable — what matters is 3 separate rows, not a single repr string.
    failure_values = fc["failure_case"].to_list()
    for v in failure_values:
        if isinstance(v, str):
            try:
                float(v)  # must be numeric-castable, not a DataFrame repr
            except ValueError:
                raise AssertionError(
                    f"Expected individual numeric values, got non-numeric string: {v!r}"
                )
        elif not isinstance(v, (int, float)):
            raise AssertionError(
                f"Expected numeric value, got: {type(v).__name__} {v!r}"
            )


def test_lazy_failure_cases_per_row_ibis():
    """SchemaErrors.failure_cases is ibis.Table with N rows for ibis lazy=True.

    # MISSING-01 regression: ibis failure_cases must be ibis.Table with N rows
    # TEST-02: intentionally ibis_table-specific — regression test for ibis lazy=True behavior
    """
    ibis = pytest.importorskip("ibis")
    import ibis.expr.datatypes as dt
    import pandas as pd

    from pandera.api.ibis.components import Column as IbisColumn
    from pandera.api.ibis.container import DataFrameSchema as IbisSchema

    schema = IbisSchema(
        columns={"a": IbisColumn(dt.int64, checks=[Check.greater_than(10)])}
    )
    # 3 failing rows — failure_cases must be an ibis.Table with 3 rows
    # TEST-02: intentionally ibis_table-specific — uses ibis.memtable for ibis lazy=True test
    frame = ibis.memtable(pd.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(SchemaErrors) as exc_info:
        schema.validate(frame, lazy=True)
    fc = exc_info.value.failure_cases
    assert isinstance(fc, ibis.Table) or hasattr(fc, "execute"), (
        f"Expected ibis.Table, got: {type(fc)}"
    )
    assert fc.count().execute() == 3, (
        f"Expected 3 rows, got {fc.count().execute()}"
    )


def test_lazy_bool_output_check_does_not_crash():
    """lazy=True with bool-output check raises SchemaErrors, not TypeError.

    # MISSING-02 regression: _count_failure_cases() must not crash with TypeError for bool scalar
    #
    # Trigger path: native=True check returns bool False -> postprocess_bool_output
    # sets failure_cases=None -> run_check sets failure_cases=passed=False ->
    # _count_failure_cases(False) raises TypeError without the fix.
    # TEST-02: intentionally polars_eager-specific — regression test for bool scalar crash
    """
    schema = DataFrameSchema(
        columns={
            "a": Column(
                pl.Int64,
                # native=True: receives (native_frame, key) and returns bool
                checks=[Check(lambda native_frame, key: False)],
            )
        }
    )
    # Must raise SchemaErrors, not TypeError
    with pytest.raises(SchemaErrors):
        # TEST-02: intentionally polars_eager-specific — uses pl.DataFrame to trigger the bug path
        schema.validate(pl.DataFrame({"a": [1, 2, 3]}), lazy=True)
