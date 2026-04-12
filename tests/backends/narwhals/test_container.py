"""Container-level tests for the narwhals backend.

Tests cover:
  CONTAINER-01: NarwhalsSchemaBackend has failure_cases_metadata, drop_invalid_rows
  CONTAINER-02: schema.validate(pl.DataFrame/pl.LazyFrame) works end-to-end
  CONTAINER-03: strict=True raises SchemaError; strict="filter" drops extra columns
  CONTAINER-04: lazy=True collects all errors before raising SchemaErrors
  REGISTER-01: register_polars_backends() is idempotent via lru_cache
  REGISTER-02: DataFrameSchema backend for pl.DataFrame/pl.LazyFrame is narwhals after registration
  REGISTER-04: narwhals backend auto-activated when narwhals is installed
  TEST-03: SchemaError.failure_cases is a native pl.DataFrame, not nw.DataFrame
"""

import narwhals.stable.v1 as nw
import polars as pl
import pytest

from pandera.api.checks import Check
from pandera.api.polars.components import Column
from pandera.api.polars.container import DataFrameSchema
from pandera.errors import SchemaError, SchemaErrors

# ---------------------------------------------------------------------------
# CONTAINER-01: NarwhalsSchemaBackend.failure_cases_metadata / drop_invalid_rows
# ---------------------------------------------------------------------------


def test_failure_cases_metadata():
    """NarwhalsSchemaBackend.failure_cases_metadata returns object with .failure_cases."""
    from pandera.api.polars.container import DataFrameSchema as Schema
    from pandera.backends.narwhals.base import NarwhalsSchemaBackend

    schema = Schema(columns={"a": Column(pl.Int64)})
    failure_cases_df = pl.DataFrame({"a": [0]})
    err = SchemaError(
        schema=schema,
        data=failure_cases_df,
        message="test error",
        failure_cases=failure_cases_df,
    )
    backend = NarwhalsSchemaBackend()
    result = backend.failure_cases_metadata("test", [err])
    assert hasattr(result, "failure_cases")
    assert isinstance(result.failure_cases, pl.DataFrame)


# ---------------------------------------------------------------------------
# CONTAINER-02: schema.validate end-to-end
# ---------------------------------------------------------------------------


def test_validate_polars_dataframe():
    """schema.validate(pl.DataFrame) returns a native pl.DataFrame."""
    schema = DataFrameSchema(columns={"a": Column(pl.Int64)})
    result = schema.validate(pl.DataFrame({"a": [1, 2, 3]}))
    assert isinstance(result, pl.DataFrame)


def test_validate_polars_lazyframe():
    """schema.validate(pl.LazyFrame) returns a native pl.LazyFrame."""
    schema = DataFrameSchema(columns={"a": Column(pl.Int64)})
    result = schema.validate(pl.LazyFrame({"a": [1, 2, 3]}))
    assert isinstance(result, pl.LazyFrame)


def test_validate_invalid_raises_schema_error():
    """schema.validate raises SchemaError when a check fails."""
    schema = DataFrameSchema(
        columns={"a": Column(pl.Int64, checks=[Check.greater_than(10)])}
    )
    try:
        schema.validate(pl.DataFrame({"a": [1, 2, 3]}))
        pytest.fail("Expected SchemaError was not raised")
    except SchemaError:
        pass


# ---------------------------------------------------------------------------
# CONTAINER-03: strict mode
# ---------------------------------------------------------------------------


def test_strict_true_rejects_extra_columns():
    """schema.validate raises when extra columns present and strict=True."""
    schema = DataFrameSchema(columns={"a": Column(pl.Int64)}, strict=True)
    with pytest.raises((SchemaError, SchemaErrors)):
        schema.validate(pl.DataFrame({"a": [1], "b": [2]}))


def test_strict_filter_drops_extra_columns():
    """schema.validate drops extra columns when strict='filter'."""
    schema = DataFrameSchema(columns={"a": Column(pl.Int64)}, strict="filter")
    result = schema.validate(pl.DataFrame({"a": [1], "b": [2]}))
    assert "b" not in result.columns
    assert "a" in result.columns


# ---------------------------------------------------------------------------
# CONTAINER-04: lazy mode collects all errors
# ---------------------------------------------------------------------------


def test_lazy_mode_collects_all_errors():
    """schema.validate(lazy=True) raises SchemaErrors with multiple failures."""
    schema = DataFrameSchema(
        columns={
            "a": Column(pl.Int64, checks=[Check.greater_than(10)]),
            "b": Column(pl.Int64, checks=[Check.greater_than(10)]),
        }
    )
    with pytest.raises(SchemaErrors) as exc_info:
        schema.validate(pl.DataFrame({"a": [1, 2], "b": [3, 4]}), lazy=True)
    assert len(exc_info.value.schema_errors) > 1
    from pandera.errors import SchemaErrorReason

    for err in exc_info.value.schema_errors:
        assert err.reason_code == SchemaErrorReason.DATAFRAME_CHECK, (
            f"Expected DATAFRAME_CHECK but got {err.reason_code}"
        )


# ---------------------------------------------------------------------------
# REGISTER-01: register_polars_backends is idempotent
# ---------------------------------------------------------------------------


def test_register_is_idempotent():
    """Calling register_polars_backends() twice does not raise or corrupt state.

    lru_cache ensures the second call is a no-op — registry state is preserved.
    """
    from pandera.backends.polars.register import register_polars_backends

    register_polars_backends()
    register_polars_backends()
    # No exception should be raised; lru_cache makes second call a no-op


# ---------------------------------------------------------------------------
# REGISTER-02: DataFrameSchema backend is narwhals after registration
# ---------------------------------------------------------------------------


def test_polars_backends_registered():
    """After register_polars_backends(), pl.DataFrame uses narwhals DataFrameSchemaBackend."""
    from pandera.backends.narwhals.container import DataFrameSchemaBackend

    backend = DataFrameSchema.get_backend(pl.DataFrame({}))
    assert isinstance(backend, DataFrameSchemaBackend)


# ---------------------------------------------------------------------------
# REGISTER-04: narwhals backend auto-activated when narwhals is installed
# ---------------------------------------------------------------------------


def test_narwhals_auto_activated_when_installed():
    """register_polars_backends() emits UserWarning when narwhals is installed."""
    import warnings

    from pandera.backends.polars.register import register_polars_backends

    register_polars_backends.cache_clear()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        register_polars_backends()
        narwhals_warnings = [x for x in w if "Narwhals" in str(x.message)]
        assert len(narwhals_warnings) == 1
    register_polars_backends.cache_clear()  # restore clean state


# ---------------------------------------------------------------------------
# TEST-03: SchemaError.failure_cases is nw.DataFrame (Phase 4+)
# ---------------------------------------------------------------------------


def test_failure_cases_is_native():
    """SchemaError.failure_cases is a native pl.DataFrame for polars inputs.

    Phase 6 contract: failure_cases is native (pl.DataFrame for polars) — not nw.DataFrame.
    RED until Plan 03 materializes failure_cases to native in the error pipeline.
    """
    schema = DataFrameSchema(
        columns={"a": Column(pl.Int64, checks=[Check.greater_than(10)])}
    )
    try:
        schema.validate(pl.DataFrame({"a": [1, 2, 3]}))
        pytest.fail("Expected SchemaError was not raised")
    except SchemaError as err:
        fc = err.failure_cases
        assert isinstance(fc, pl.DataFrame), (
            f"failure_cases should be native pl.DataFrame (Phase 6 contract), got {type(fc)}"
        )


# ---------------------------------------------------------------------------
# REGISTER-03: ibis.Table uses narwhals DataFrameSchemaBackend after registration
# ---------------------------------------------------------------------------


def test_ibis_narwhals_auto_activated():
    """register_ibis_backends() emits UserWarning when narwhals is installed."""
    import warnings

    from pandera.backends.ibis.register import register_ibis_backends

    register_ibis_backends.cache_clear()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        register_ibis_backends()
        narwhals_warnings = [x for x in w if "Narwhals" in str(x.message)]
        assert len(narwhals_warnings) == 1
    register_ibis_backends.cache_clear()  # restore clean state


def test_ibis_backend_is_narwhals():
    """After register_ibis_backends(), ibis.Table uses narwhals DataFrameSchemaBackend."""
    import ibis

    from pandera.api.ibis.container import (
        DataFrameSchema as IbisDataFrameSchema,
    )
    from pandera.backends.ibis.register import register_ibis_backends
    from pandera.backends.narwhals.container import DataFrameSchemaBackend

    register_ibis_backends()
    t = ibis.memtable({"a": [1, 2, 3]})
    backend = IbisDataFrameSchema.get_backend(t)
    assert isinstance(backend, DataFrameSchemaBackend)
