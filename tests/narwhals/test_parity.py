"""Ibis-specific and complex cross-backend narwhals tests.

Cross-backend behavioral parity (strict, nullable, check decorators, etc.)
lives in tests/narwhals/backends/test_e2e.py, parametrized via BackendFixture.

This file covers ibis-specific contracts (failure_cases type, BooleanScalar
normalization, element_wise rejection) and multi-ibis-backend drop_invalid_rows
parity that requires monkeypatching the ibis backend. Covers TEST-04.

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
    import ibis.expr.datatypes as dt

    from pandera.api.ibis.components import Column as IbisColumn
    from pandera.api.ibis.container import DataFrameSchema as IbisSchema

    schema = IbisSchema(columns={"a": IbisColumn(dt.int64, coerce=True)})
    # ibis.memtable with string column — should coerce to int64
    import ibis
    import pandas as pd

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
# TEST-09: drop_invalid_rows — Polars/Ibis parity
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


def _setup_drop_invalid_rows_backend(backend_name, monkeypatch):
    """Return (Schema, Column, make_frame, collect) for drop_invalid_rows parity tests.

    make_frame(data_dict) wraps a plain dict of lists into the backend frame type.
    collect(result) returns {col: [values...]} with NaN normalised to None.
    Both Schema and Column accept Python native types (int, str) as dtype.
    """
    if backend_name == "polars":
        from pandera.api.polars.components import Column
        from pandera.api.polars.container import DataFrameSchema as Schema

        def make_frame(data):
            return pl.LazyFrame(data)

        def collect(result):
            df = result.collect()
            return {col: df[col].to_list() for col in df.columns}

    else:
        import ibis

        ibis_backend = backend_name.split("_")[1]
        monkeypatch.setattr(ibis.options, "default_backend", None)
        ibis.set_backend(ibis_backend)
        from pandera.api.ibis.components import Column
        from pandera.api.ibis.container import DataFrameSchema as Schema

        _ibis_type = {int: "int64", str: "string"}

        def make_frame(data):
            fields = [
                (k, _ibis_type[type(next(x for x in v if x is not None))])
                for k, v in data.items()
            ]
            return ibis.memtable(data, schema=ibis.schema(fields))

        def collect(result):
            df = result.execute()
            out = {}
            for col in df.columns:
                out[col] = [
                    None
                    if (isinstance(v, float) and v != v)
                    else (
                        int(v) if isinstance(v, float) and v == int(v) else v
                    )
                    for v in df[col].tolist()
                ]
            return out

    return Schema, Column, make_frame, collect


@pytest.mark.parametrize(
    "backend_name", ["polars", "ibis_duckdb", "ibis_sqlite"]
)
def test_drop_invalid_rows_parity(backend_name, monkeypatch):
    """drop_invalid_rows=True, lazy=True filters invalid rows for both Polars and Ibis."""
    Schema, Column, make_frame, collect = _setup_drop_invalid_rows_backend(
        backend_name, monkeypatch
    )
    schema = Schema(
        columns={"a": Column(int, Check.ge(0))},
        drop_invalid_rows=True,
    )
    result = collect(
        schema.validate(make_frame({"a": [-1, 0, 1, 2]}), lazy=True)
    )
    assert result["a"] == [0, 1, 2]


@pytest.mark.parametrize(
    "backend_name", ["polars", "ibis_duckdb", "ibis_sqlite"]
)
def test_drop_invalid_rows_lazy_false_raises_parity(backend_name, monkeypatch):
    """drop_invalid_rows=True with lazy=False raises SchemaDefinitionError on all backends."""
    from pandera.errors import SchemaDefinitionError

    Schema, Column, make_frame, _ = _setup_drop_invalid_rows_backend(
        backend_name, monkeypatch
    )
    schema = Schema(
        columns={"a": Column(int, Check.ge(0))},
        drop_invalid_rows=True,
    )
    with pytest.raises(SchemaDefinitionError):
        schema.validate(make_frame({"a": [-1, 1, 2]}), lazy=False)


@pytest.mark.parametrize(
    "backend_name", ["polars", "ibis_duckdb", "ibis_sqlite"]
)
def test_drop_invalid_rows_nullable_parity(backend_name, monkeypatch):
    """drop_invalid_rows with nullable=True: null rows pass, invalid non-null rows are dropped."""
    Schema, Column, make_frame, collect = _setup_drop_invalid_rows_backend(
        backend_name, monkeypatch
    )
    schema = Schema(
        columns={"a": Column(int, Check.ge(0), nullable=True)},
        drop_invalid_rows=True,
    )
    result = collect(
        schema.validate(make_frame({"a": [None, -1, 0, 1]}), lazy=True)
    )
    assert result["a"] == [None, 0, 1]


@pytest.mark.parametrize(
    "backend_name", ["polars", "ibis_duckdb", "ibis_sqlite"]
)
def test_drop_invalid_rows_multiple_checks_parity(backend_name, monkeypatch):
    """drop_invalid_rows drops a row if ANY per-column check fails, not only when all fail.

    Exercises nw.all_horizontal accumulation across multiple nw.Expr check_outputs.

    Data:
      a=-1, b="0"  → dropped (a fails ge(0))
      a=0,  b="x"  → dropped (b fails isin)
      a=0,  b="0"  → kept
      a=1,  b="1"  → kept
    """
    Schema, Column, make_frame, collect = _setup_drop_invalid_rows_backend(
        backend_name, monkeypatch
    )
    schema = Schema(
        columns={
            "a": Column(int, Check.ge(0)),
            "b": Column(str, Check.isin([*"012"])),
        },
        drop_invalid_rows=True,
    )
    result = collect(
        schema.validate(
            make_frame({"a": [-1, 0, 0, 1], "b": ["0", "x", "0", "1"]}),
            lazy=True,
        )
    )
    assert result["a"] == [0, 1]
    assert result["b"] == ["0", "1"]
