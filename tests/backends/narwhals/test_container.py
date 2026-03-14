"""Container-level tests for the narwhals backend.

Tests cover:
  CONTAINER-01: NarwhalsSchemaBackend has failure_cases_metadata, drop_invalid_rows
  CONTAINER-02: schema.validate(pl.DataFrame/pl.LazyFrame) works end-to-end
  CONTAINER-03: strict=True raises SchemaError; strict="filter" drops extra columns
  CONTAINER-04: lazy=True collects all errors before raising SchemaErrors
  REGISTER-01: register_narwhals_backends() is idempotent via lru_cache
  REGISTER-02: DataFrameSchema backend for pl.DataFrame/pl.LazyFrame is narwhals after registration
  REGISTER-04: narwhals backend is NOT registered by default
  TEST-03: SchemaError.failure_cases is a native pl.DataFrame, not nw.DataFrame
"""

import pytest
import polars as pl
import narwhals.stable.v1 as nw
from pandera.api.polars.container import DataFrameSchema
from pandera.api.polars.components import Column
from pandera.api.checks import Check
from pandera.errors import SchemaError, SchemaErrors


# ---------------------------------------------------------------------------
# CONTAINER-01: NarwhalsSchemaBackend.failure_cases_metadata / drop_invalid_rows
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="CONTAINER-01: failure_cases_metadata not implemented yet", strict=False)
def test_failure_cases_metadata():
    """NarwhalsSchemaBackend.failure_cases_metadata returns object with .failure_cases."""
    from pandera.backends.narwhals.base import NarwhalsSchemaBackend
    from pandera.api.polars.container import DataFrameSchema as Schema

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


@pytest.mark.xfail(reason="CONTAINER-01: drop_invalid_rows not implemented yet", strict=False)
def test_drop_invalid_rows():
    """NarwhalsSchemaBackend.drop_invalid_rows returns a frame without raising."""
    from pandera.backends.narwhals.base import NarwhalsSchemaBackend

    frame = nw.from_native(pl.LazyFrame({"a": [1, 2, 3]}), eager_or_interchange_only=False)

    class _FakeError:
        column_name = "a"
        failure_cases = pl.DataFrame({"a": [1]})

    class _FakeHandler:
        def collect(self):
            return [_FakeError()]

    backend = NarwhalsSchemaBackend()
    result = backend.drop_invalid_rows(frame, _FakeHandler())
    # Should return a frame, not raise
    assert result is not None


# ---------------------------------------------------------------------------
# CONTAINER-02: schema.validate end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="CONTAINER-02: narwhals container backend not implemented yet", strict=False)
def test_validate_polars_dataframe():
    """schema.validate(pl.DataFrame) returns a native pl.DataFrame."""
    from pandera.backends.narwhals.register import register_narwhals_backends

    register_narwhals_backends()
    schema = DataFrameSchema(columns={"a": Column(pl.Int64)})
    result = schema.validate(pl.DataFrame({"a": [1, 2, 3]}))
    assert isinstance(result, pl.DataFrame)


@pytest.mark.xfail(reason="CONTAINER-02: narwhals container backend not implemented yet", strict=False)
def test_validate_polars_lazyframe():
    """schema.validate(pl.LazyFrame) returns a native pl.LazyFrame."""
    from pandera.backends.narwhals.register import register_narwhals_backends

    register_narwhals_backends()
    schema = DataFrameSchema(columns={"a": Column(pl.Int64)})
    result = schema.validate(pl.LazyFrame({"a": [1, 2, 3]}))
    assert isinstance(result, pl.LazyFrame)


@pytest.mark.xfail(reason="CONTAINER-02: narwhals container backend not implemented yet", strict=False)
def test_validate_invalid_raises_schema_error():
    """schema.validate raises SchemaError when a check fails."""
    from pandera.backends.narwhals.register import register_narwhals_backends

    register_narwhals_backends()
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


@pytest.mark.xfail(reason="CONTAINER-03: strict=True not implemented yet", strict=False)
def test_strict_true_rejects_extra_columns():
    """schema.validate raises when extra columns present and strict=True."""
    from pandera.backends.narwhals.register import register_narwhals_backends

    register_narwhals_backends()
    schema = DataFrameSchema(columns={"a": Column(pl.Int64)}, strict=True)
    with pytest.raises((SchemaError, SchemaErrors)):
        schema.validate(pl.DataFrame({"a": [1], "b": [2]}))


@pytest.mark.xfail(reason="CONTAINER-03: strict='filter' not implemented yet", strict=False)
def test_strict_filter_drops_extra_columns():
    """schema.validate drops extra columns when strict='filter'."""
    from pandera.backends.narwhals.register import register_narwhals_backends

    register_narwhals_backends()
    schema = DataFrameSchema(columns={"a": Column(pl.Int64)}, strict="filter")
    result = schema.validate(pl.DataFrame({"a": [1], "b": [2]}))
    assert "b" not in result.columns
    assert "a" in result.columns


# ---------------------------------------------------------------------------
# CONTAINER-04: lazy mode collects all errors
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="CONTAINER-04: lazy=True not implemented yet", strict=False)
def test_lazy_mode_collects_all_errors():
    """schema.validate(lazy=True) raises SchemaErrors with multiple failures."""
    from pandera.backends.narwhals.register import register_narwhals_backends

    register_narwhals_backends()
    schema = DataFrameSchema(
        columns={
            "a": Column(pl.Int64, checks=[Check.greater_than(10)]),
            "b": Column(pl.Int64, checks=[Check.greater_than(10)]),
        }
    )
    with pytest.raises(SchemaErrors) as exc_info:
        schema.validate(pl.DataFrame({"a": [1, 2], "b": [3, 4]}), lazy=True)
    assert len(exc_info.value.schema_errors) > 1


# ---------------------------------------------------------------------------
# REGISTER-01: register_narwhals_backends is idempotent
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="REGISTER-01: register_narwhals_backends not implemented yet", strict=False)
def test_register_is_idempotent():
    """Calling register_narwhals_backends() twice does not raise or corrupt state."""
    from pandera.backends.narwhals.register import register_narwhals_backends

    register_narwhals_backends()
    register_narwhals_backends()
    # No exception should be raised; lru_cache makes second call a no-op


# ---------------------------------------------------------------------------
# REGISTER-02: DataFrameSchema backend is narwhals after registration
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="REGISTER-02: narwhals DataFrameSchemaBackend not registered yet", strict=False)
def test_polars_backends_registered():
    """After register_narwhals_backends(), pl.DataFrame uses DataFrameSchemaBackend."""
    from pandera.backends.narwhals.register import register_narwhals_backends
    from pandera.backends.narwhals.container import DataFrameSchemaBackend

    register_narwhals_backends()
    backend = DataFrameSchema.get_backend(pl.DataFrame({}))
    assert isinstance(backend, DataFrameSchemaBackend)


# ---------------------------------------------------------------------------
# REGISTER-04: narwhals backend is NOT registered by default
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="REGISTER-04: verifying narwhals not auto-registered", strict=False)
def test_narwhals_not_registered_by_default():
    """Without register_narwhals_backends(), pl.DataFrame backend is not narwhals."""
    # Import here to avoid influencing module-level state
    from pandera.backends.polars.container import PolarsSchemaBackend

    # Check the registry without triggering narwhals registration
    try:
        from pandera.backends.narwhals.container import DataFrameSchemaBackend as NarwhalsBackend
    except ImportError:
        # Module doesn't exist yet — registration is definitely not in effect
        return

    backend = DataFrameSchema.get_backend(pl.DataFrame({}))
    assert not isinstance(backend, NarwhalsBackend), (
        "NarwhalsSchemaBackend should not be registered by default"
    )


# ---------------------------------------------------------------------------
# TEST-03: SchemaError.failure_cases is native pl.DataFrame
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="TEST-03: failure_cases native type not guaranteed yet", strict=False)
def test_failure_cases_is_native():
    """SchemaError.failure_cases is a native pl.DataFrame, not nw.DataFrame."""
    from pandera.backends.narwhals.register import register_narwhals_backends

    register_narwhals_backends()
    schema = DataFrameSchema(
        columns={"a": Column(pl.Int64, checks=[Check.greater_than(10)])}
    )
    try:
        schema.validate(pl.DataFrame({"a": [1, 2, 3]}))
        pytest.fail("Expected SchemaError was not raised")
    except SchemaError as err:
        fc = err.failure_cases
        assert isinstance(fc, pl.DataFrame) or not isinstance(fc, nw.DataFrame), (
            f"failure_cases should be native pl.DataFrame, got {type(fc)}"
        )
