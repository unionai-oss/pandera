"""Ibis-specific narwhals tests and nw.Expr accumulation contracts.

Cross-backend behavioral parity (strict, nullable, check decorators,
drop_invalid_rows, etc.) lives in tests/narwhals/backends/test_e2e.py,
parametrized via BackendFixture.

This file covers ibis-specific contracts (failure_cases type, BooleanScalar
normalization, element_wise rejection) and the nw.Expr accumulation contract
for drop_invalid_rows (Polars-specific, checks check_output type). Covers TEST-04.

Coerce-dependent tests are marked xfail(strict=True) — coerce is a v2
feature; strict=True ensures CI breaks when coerce lands so marks are
cleaned up rather than silently accumulating.
"""

import narwhals.stable.v1 as nw
import polars as pl
import pytest

from pandera.api.checks import Check
from pandera.api.polars.components import Column
from pandera.api.polars.container import DataFrameSchema
from pandera.config import ValidationDepth, config_context
from pandera.errors import SchemaError, SchemaErrors

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ibis_table(data: dict):
    import ibis
    import pandas as pd

    return ibis.memtable(pd.DataFrame(data))


# ---------------------------------------------------------------------------
# TEST-04: DataFrameModel parity (Polars vs Ibis)
# ---------------------------------------------------------------------------


def test_dataframe_model_ibis():
    """Ibis DataFrameModel schema validates an ibis Table correctly."""
    import ibis.expr.datatypes as dt

    import pandera.ibis as pa_ibis

    class MyModel(pa_ibis.DataFrameModel):
        a: dt.int64

    t = _make_ibis_table({"a": [1, 2, 3]})
    result = MyModel.validate(t)
    assert result is not None


# ---------------------------------------------------------------------------
# TEST-04: Coerce parity (strict=True — must fail until coerce lands in v2)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="TEST-04: coerce not yet implemented in narwhals backend (v2 feature)",
    strict=True,
)
def test_coerce_ibis():
    """coerce=True on ibis Column coerces dtype — xfail strict=True until v2."""
    import ibis
    import ibis.expr.datatypes as dt
    import pandas as pd

    from pandera.api.ibis.components import Column as IbisColumn
    from pandera.api.ibis.container import DataFrameSchema as IbisSchema

    schema = IbisSchema(columns={"a": IbisColumn(dt.int64, coerce=True)})
    # ibis.memtable with string column — should coerce to int64
    t = ibis.memtable(pd.DataFrame({"a": ["1", "2", "3"]}))
    result = schema.validate(t)
    assert result is not None


# ---------------------------------------------------------------------------
# TEST-02: element_wise=True check on ibis Table raises (not supported)
# ---------------------------------------------------------------------------


def test_element_wise_check_raises_not_implemented_ibis():
    """element_wise=True on ibis Table raises SchemaError wrapping NotImplementedError.

    SQL-lazy backends (Ibis, DuckDB, PySpark) cannot apply row-level Python
    functions to lazy query plans. element_wise checks are rejected at check
    application time. The NotImplementedError is captured by run_checks and
    surfaced as a SchemaError with CHECK_ERROR reason_code. TEST-02.
    """
    import ibis.expr.datatypes as dt

    from pandera.api.ibis.components import Column as IbisColumn
    from pandera.api.ibis.container import DataFrameSchema as IbisSchema

    schema = IbisSchema(
        columns={
            "a": IbisColumn(
                dt.int64, checks=[Check(lambda x: x > 0, element_wise=True)]
            )
        }
    )
    t = _make_ibis_table({"a": [1, 2, 3]})
    with pytest.raises(SchemaError) as exc_info:
        schema.validate(t)
    # The SchemaError wraps the NotImplementedError from element_wise check rejection
    assert "NotImplementedError" in str(
        exc_info.value
    ) or "element_wise" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Gap closure: lazy=True custom ibis check must not crash (UAT test 8)
# ---------------------------------------------------------------------------


def test_custom_check_ibis_lazy():
    """schema.validate(ibis_table, lazy=True) with a custom check completes without crashing.

    Regression test for _count_failure_cases calling len() on ibis.Table.
    ibis raises ExpressionError('Use .count() instead') — not TypeError — so
    the original except clause missed it. Fix: detect ibis.Table and use
    .count().execute() instead.

    The check must use IbisData wrapping to produce an ibis.Table as
    failure_cases, which is what triggers the _count_failure_cases bug.
    """
    import ibis.expr.datatypes as dt
    import ibis.selectors as s
    from ibis import _ as ibis_deferred

    from pandera.api.ibis.components import Column as IbisColumn
    from pandera.api.ibis.container import DataFrameSchema as IbisSchema
    from pandera.api.ibis.types import IbisData

    # Custom check using IbisData — returns a column-selection ibis expression.
    # Values [1, 2, 3] are not all == 0, so the check will fail.
    # The failure produces an ibis.Table as failure_cases, which triggered the
    # _count_failure_cases len() crash (ExpressionError, not TypeError).
    def _check_all_zero(data: IbisData):
        return data.table.select(s.across(s.all(), ibis_deferred == 0))

    schema = IbisSchema(
        columns={"a": IbisColumn(dt.int64, checks=[Check(_check_all_zero)])},
    )
    t = _make_ibis_table({"a": [1, 2, 3]})
    with pytest.raises((SchemaError, SchemaErrors)):
        schema.validate(t, lazy=True)


# ---------------------------------------------------------------------------
# TEST-09: drop_invalid_rows — nw.Expr accumulation contract (Polars)
# ---------------------------------------------------------------------------


def test_drop_invalid_rows_expr_accumulation():
    """drop_invalid_rows with lazy=True correctly filters invalid rows using nw.Expr accumulation.

    GREEN after 09-02: apply() returns nw.Expr directly; postprocess_expr_output()
    stores check_output=expr (no wide table built during check loop); drop_invalid_rows
    uses nw.all_horizontal on accumulated exprs — pure narwhals, no backend delegation.

    With drop_invalid_rows=True, SchemaErrors is NOT raised — invalid rows are silently
    dropped. The result contains only the rows that pass all checks.
    """
    schema = DataFrameSchema(
        columns={"a": Column(pl.Int64, checks=[Check.greater_than(0)])},
        drop_invalid_rows=True,
    )
    lf = pl.LazyFrame({"a": [-1, 1, 2]})
    result = schema.validate(lf, lazy=True)
    # Should not raise — invalid row (-1) is dropped
    result_df = result.collect()
    assert len(result_df) == 2, (
        f"Expected 2 valid rows after drop_invalid_rows, got {len(result_df)}: {result_df}"
    )
    assert result_df["a"].to_list() == [1, 2], (
        f"Expected [1, 2] after dropping -1, got {result_df['a'].to_list()}"
    )

    # Verify check_output stored in schema_errors is nw.Expr (Phase 09 contract).
    # Access via lazy=True with drop_invalid_rows=False to see the check_output.
    # Force SCHEMA_AND_DATA so data checks run even on lazy frames (matching the
    # polars test conftest behavior).
    schema_no_drop = DataFrameSchema(
        columns={"a": Column(pl.Int64, checks=[Check.greater_than(0)])},
        drop_invalid_rows=False,
    )
    with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
        try:
            schema_no_drop.validate(lf, lazy=True)
            pytest.fail("Expected SchemaErrors was not raised")
        except SchemaErrors as err:
            check_output = err.schema_errors[0].check_output
            assert isinstance(check_output, nw.Expr), (
                f"check_output should be nw.Expr after Phase 09 fix, got {type(check_output)}"
            )
