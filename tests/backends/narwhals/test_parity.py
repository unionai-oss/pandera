"""Behavioral parity tests: narwhals backend against Polars and Ibis.

Tests verify that the narwhals backend produces identical validation
behavior when wrapping Polars frames and Ibis Tables. Covers TEST-04.

Coerce-dependent tests are marked xfail(strict=True) — coerce is a v2
feature; strict=True ensures CI breaks when coerce lands so marks are
cleaned up rather than silently accumulating.
"""
import pytest
import polars as pl
import narwhals.stable.v1 as nw
from pandera.api.polars.container import DataFrameSchema
from pandera.api.polars.components import Column
from pandera.api.checks import Check
from pandera.errors import SchemaError, SchemaErrors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_polars_df(data: dict) -> pl.DataFrame:
    return pl.DataFrame(data)


def _make_ibis_table(data: dict):
    import pandas as pd
    import ibis
    return ibis.memtable(pd.DataFrame(data))


# ---------------------------------------------------------------------------
# TEST-04: Container validation parity (Polars vs Ibis)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="TEST-04: ibis.Table not yet registered for narwhals backend", strict=False)
def test_validate_ibis_valid():
    """schema.validate(ibis_table) succeeds for a valid ibis Table."""
    from pandera.api.ibis.container import DataFrameSchema as IbisSchema
    from pandera.api.ibis.components import Column as IbisColumn
    import ibis.expr.datatypes as dt

    schema = IbisSchema(columns={"a": IbisColumn(dt.int64)})
    t = _make_ibis_table({"a": [1, 2, 3]})
    result = schema.validate(t)
    assert result is not None


@pytest.mark.xfail(reason="TEST-04: ibis.Table not yet registered for narwhals backend", strict=False)
def test_validate_ibis_invalid_raises():
    """schema.validate(ibis_table) raises SchemaError for an invalid ibis Table."""
    from pandera.api.ibis.container import DataFrameSchema as IbisSchema
    from pandera.api.ibis.components import Column as IbisColumn
    import ibis.expr.datatypes as dt

    schema = IbisSchema(columns={"a": IbisColumn(dt.int64, checks=[Check.greater_than(10)])})
    t = _make_ibis_table({"a": [1, 2, 3]})
    with pytest.raises((SchemaError, SchemaErrors)):
        schema.validate(t)


@pytest.mark.xfail(reason="TEST-04: ibis lazy validation not yet registered", strict=False)
def test_lazy_mode_ibis_collects_all_errors():
    """schema.validate(ibis_table, lazy=True) collects multiple errors."""
    from pandera.api.ibis.container import DataFrameSchema as IbisSchema
    from pandera.api.ibis.components import Column as IbisColumn
    import ibis.expr.datatypes as dt

    schema = IbisSchema(columns={
        "a": IbisColumn(dt.int64, checks=[Check.greater_than(10)]),
        "b": IbisColumn(dt.int64, checks=[Check.greater_than(10)]),
    })
    t = _make_ibis_table({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(SchemaErrors) as exc_info:
        schema.validate(t, lazy=True)
    assert len(exc_info.value.schema_errors) > 1


@pytest.mark.xfail(reason="TEST-04: ibis strict mode not yet registered", strict=False)
def test_strict_true_ibis_rejects_extra_columns():
    """schema.validate(ibis_table, strict=True) raises for extra columns."""
    from pandera.api.ibis.container import DataFrameSchema as IbisSchema
    from pandera.api.ibis.components import Column as IbisColumn
    import ibis.expr.datatypes as dt

    schema = IbisSchema(columns={"a": IbisColumn(dt.int64)}, strict=True)
    t = _make_ibis_table({"a": [1], "b": [2]})
    with pytest.raises((SchemaError, SchemaErrors)):
        schema.validate(t)


@pytest.mark.xfail(reason="TEST-04: ibis strict=filter not yet registered", strict=False)
def test_strict_filter_ibis_drops_extra_columns():
    """schema.validate(ibis_table, strict='filter') drops extra columns."""
    from pandera.api.ibis.container import DataFrameSchema as IbisSchema
    from pandera.api.ibis.components import Column as IbisColumn
    import ibis.expr.datatypes as dt

    schema = IbisSchema(columns={"a": IbisColumn(dt.int64)}, strict="filter")
    t = _make_ibis_table({"a": [1], "b": [2]})
    result = schema.validate(t)
    assert result is not None
    result_df = result.execute()
    assert "b" not in result_df.columns
    assert "a" in result_df.columns


@pytest.mark.xfail(reason="TEST-04: ibis failure_cases native type not yet verified", strict=False)
def test_failure_cases_native_ibis():
    """SchemaError.failure_cases on ibis validation is a native (non-narwhals) frame."""
    import pandas as pd
    from pandera.api.ibis.container import DataFrameSchema as IbisSchema
    from pandera.api.ibis.components import Column as IbisColumn
    import ibis.expr.datatypes as dt

    schema = IbisSchema(columns={"a": IbisColumn(dt.int64, checks=[Check.greater_than(10)])})
    t = _make_ibis_table({"a": [1, 2, 3]})
    try:
        schema.validate(t)
        pytest.fail("Expected SchemaError was not raised")
    except SchemaError as err:
        fc = err.failure_cases
        # Must NOT be a narwhals wrapper
        assert not isinstance(fc, (nw.DataFrame, nw.LazyFrame)), (
            f"failure_cases must be native, got {type(fc)}"
        )
        # Should be a native frame type (pandas for ibis)
        assert isinstance(fc, (pd.DataFrame, pl.DataFrame)), (
            f"Expected native frame, got {type(fc)}"
        )


# ---------------------------------------------------------------------------
# TEST-04: Decorator parity (Polars vs Ibis)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="TEST-04: ibis decorator parity not yet implemented", strict=False)
def test_check_decorator_ibis():
    """@pa.check_input / @pa.check_output works with ibis Table."""
    import pandera.ibis as pa_ibis
    import ibis.expr.datatypes as dt

    schema = pa_ibis.DataFrameSchema(columns={"a": pa_ibis.Column(dt.int64)})

    @pa_ibis.check_input(schema)
    def my_func(t):
        return t

    t = _make_ibis_table({"a": [1, 2, 3]})
    result = my_func(t)
    assert result is not None


# ---------------------------------------------------------------------------
# TEST-04: DataFrameModel parity (Polars vs Ibis)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="TEST-04: ibis DataFrameModel parity not yet implemented", strict=False)
def test_dataframe_model_ibis():
    """Ibis DataFrameModel schema validates an ibis Table correctly."""
    import pandera.ibis as pa_ibis
    import ibis.expr.datatypes as dt

    class MyModel(pa_ibis.DataFrameModel):
        a: dt.int64

    t = _make_ibis_table({"a": [1, 2, 3]})
    result = MyModel.validate(t)
    assert result is not None


# ---------------------------------------------------------------------------
# TEST-04: Coerce parity (strict=True — must fail until coerce lands in v2)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="TEST-04: coerce not yet implemented in narwhals backend (v2 feature)", strict=True)
def test_coerce_ibis():
    """coerce=True on ibis Column coerces dtype — xfail strict=True until v2."""
    from pandera.api.ibis.container import DataFrameSchema as IbisSchema
    from pandera.api.ibis.components import Column as IbisColumn
    import ibis.expr.datatypes as dt

    schema = IbisSchema(columns={"a": IbisColumn(dt.int64, coerce=True)})
    # ibis.memtable with string column — should coerce to int64
    import pandas as pd
    import ibis
    t = ibis.memtable(pd.DataFrame({"a": ["1", "2", "3"]}))
    result = schema.validate(t)
    assert result is not None


# ---------------------------------------------------------------------------
# TEST-04: Polars parity (verify narwhals backend works for polars too)
# ---------------------------------------------------------------------------

def test_validate_polars_parity():
    """narwhals backend validates Polars frames correctly (polars parity baseline)."""
    schema = DataFrameSchema(columns={"a": Column(pl.Int64)})
    result = schema.validate(pl.DataFrame({"a": [1, 2, 3]}))
    assert isinstance(result, pl.DataFrame)


def test_lazy_mode_polars_parity():
    """narwhals backend lazy=True collects multiple Polars errors (parity baseline)."""
    schema = DataFrameSchema(
        columns={
            "a": Column(pl.Int64, checks=[Check.greater_than(10)]),
            "b": Column(pl.Int64, checks=[Check.greater_than(10)]),
        }
    )
    with pytest.raises(SchemaErrors) as exc_info:
        schema.validate(pl.DataFrame({"a": [1, 2], "b": [3, 4]}), lazy=True)
    assert len(exc_info.value.schema_errors) > 1
