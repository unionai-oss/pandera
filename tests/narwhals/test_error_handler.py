"""Tests for NarwhalsErrorHandler and NarwhalsSchemaBackend wiring."""

import polars as pl
import pytest

# ---------------------------------------------------------------------------
# NarwhalsErrorHandler — subclass + scalar/list counting
# ---------------------------------------------------------------------------


def test_narwhals_error_handler_is_subclass_of_base():
    """NarwhalsErrorHandler is a proper subclass of the base ErrorHandler."""
    from pandera.api.base.error_handler import ErrorHandler as BaseEH
    from pandera.api.narwhals.error_handler import ErrorHandler as NarwhalsEH

    assert issubclass(NarwhalsEH, BaseEH), (
        "NarwhalsErrorHandler must subclass base ErrorHandler"
    )


def test_narwhals_error_handler_counts_list_failure_cases():
    """NarwhalsErrorHandler._count_failure_cases returns correct count for a polars DataFrame."""
    from pandera.api.narwhals.error_handler import ErrorHandler as NarwhalsEH

    # polars DataFrame is the common native failure-cases type
    df = pl.DataFrame({"a": [1, 2, 3]})
    count = NarwhalsEH._count_failure_cases(df)
    assert count == 3, f"Expected 3, got {count}"


def test_narwhals_error_handler_counts_string_as_one():
    """NarwhalsErrorHandler._count_failure_cases returns 1 for a string failure case.

    The narwhals backends pass string failure_cases (e.g., column names, dtype strings)
    to SchemaError. NarwhalsErrorHandler must not crash on them.
    """
    from pandera.api.narwhals.error_handler import ErrorHandler as NarwhalsEH

    count = NarwhalsEH._count_failure_cases("some_column")
    assert count == 1, f"Expected 1 for string, got {count}"


# ---------------------------------------------------------------------------
# NarwhalsErrorHandler wiring — container.py, components.py, base.py
# ---------------------------------------------------------------------------


def test_container_uses_narwhals_error_handler():
    """container.py imports and uses NarwhalsErrorHandler, not the base ErrorHandler."""
    import pandera.backends.narwhals.container as container_mod
    from pandera.api.narwhals.error_handler import ErrorHandler as NarwhalsEH

    assert container_mod.ErrorHandler is NarwhalsEH, (
        "container.py must use NarwhalsErrorHandler, not base ErrorHandler"
    )


def test_components_uses_narwhals_error_handler():
    """components.py imports and uses NarwhalsErrorHandler, not the base ErrorHandler."""
    import pandera.backends.narwhals.components as components_mod
    from pandera.api.narwhals.error_handler import ErrorHandler as NarwhalsEH

    assert components_mod.ErrorHandler is NarwhalsEH, (
        "components.py must use NarwhalsErrorHandler, not base ErrorHandler"
    )


def test_base_backend_uses_narwhals_error_handler():
    """base.py imports and uses NarwhalsErrorHandler, not the base ErrorHandler."""
    import pandera.backends.narwhals.base as base_mod
    from pandera.api.narwhals.error_handler import ErrorHandler as NarwhalsEH

    assert base_mod.ErrorHandler is NarwhalsEH, (
        "base.py must use NarwhalsErrorHandler, not base ErrorHandler"
    )


# ---------------------------------------------------------------------------
# _to_frame_kind_nw — returns correct native frame type
# ---------------------------------------------------------------------------


def test_to_frame_kind_returns_lazyframe_for_lazyframe_input():
    """_to_frame_kind_nw returns a pl.LazyFrame when return_type is pl.LazyFrame."""
    import narwhals.stable.v1 as nw

    from pandera.backends.narwhals.container import _to_frame_kind_nw

    lf = nw.from_native(
        pl.LazyFrame({"a": [1, 2, 3]}), eager_or_interchange_only=False
    )
    result = _to_frame_kind_nw(lf, pl.LazyFrame)
    assert isinstance(result, pl.LazyFrame), (
        f"Expected pl.LazyFrame for lazy input, got {type(result)}"
    )


def test_to_frame_kind_returns_dataframe_for_dataframe_input():
    """_to_frame_kind_nw returns a pl.DataFrame when return_type is pl.DataFrame."""
    import narwhals.stable.v1 as nw

    from pandera.backends.narwhals.container import _to_frame_kind_nw

    # Wrap a pl.DataFrame as a nw.LazyFrame to simulate post-check state
    lf = nw.from_native(
        pl.LazyFrame({"a": [1, 2, 3]}), eager_or_interchange_only=False
    )
    result = _to_frame_kind_nw(lf, pl.DataFrame)
    assert isinstance(result, pl.DataFrame), (
        f"Expected pl.DataFrame for eager input, got {type(result)}"
    )


# ---------------------------------------------------------------------------
# validate() — deferred materialization, round-trip frame type preservation
# ---------------------------------------------------------------------------


def test_validate_lazyframe_returns_lazyframe():
    """schema.validate(pl.LazyFrame) returns pl.LazyFrame — deferred materialization works end-to-end."""
    from pandera.api.polars.components import Column
    from pandera.api.polars.container import DataFrameSchema

    schema = DataFrameSchema(columns={"a": Column(pl.Int64)})
    result = schema.validate(pl.LazyFrame({"a": [1, 2, 3]}))
    assert isinstance(result, pl.LazyFrame), (
        f"Expected pl.LazyFrame, got {type(result)}"
    )


def test_validate_dataframe_returns_dataframe():
    """schema.validate(pl.DataFrame) returns pl.DataFrame — deferred materialization works end-to-end."""
    from pandera.api.polars.components import Column
    from pandera.api.polars.container import DataFrameSchema

    schema = DataFrameSchema(columns={"a": Column(pl.Int64)})
    result = schema.validate(pl.DataFrame({"a": [1, 2, 3]}))
    assert isinstance(result, pl.DataFrame), (
        f"Expected pl.DataFrame, got {type(result)}"
    )


# ---------------------------------------------------------------------------
# SE-01 regression guard — no _handle_pyspark_validation_result method
# ---------------------------------------------------------------------------


def test_validate_no_handle_pyspark_method_after_se01():
    """DataFrameSchemaBackend must not have a _handle_pyspark_validation_result attribute.

    SE-01 regression guard: the method was deleted to unify PySpark Narwhals
    with the Polars/Ibis Narwhals error-raising contract.
    """
    from pandera.backends.narwhals.container import DataFrameSchemaBackend

    assert not hasattr(
        DataFrameSchemaBackend, "_handle_pyspark_validation_result"
    ), (
        "DataFrameSchemaBackend must NOT have _handle_pyspark_validation_result after SE-01 removal"
    )


# ---------------------------------------------------------------------------
# infer_columns() — returns correct Column subclass per schema type
# ---------------------------------------------------------------------------


def test_infer_columns_returns_correct_column_type_for_polars():
    """DataFrameSchema.infer_columns() returns polars Column instances for a polars schema."""
    import polars as pl

    import pandera.polars as pa_pl

    schema = pa_pl.DataFrameSchema(dtype=pl.Int64)
    cols = schema.infer_columns(["a", "b"])
    from pandera.api.polars.components import Column as PolarsColumn

    assert len(cols) == 2
    assert all(isinstance(c, PolarsColumn) for c in cols), (
        f"infer_columns() returned {[type(c) for c in cols]}, expected PolarsColumn"
    )
