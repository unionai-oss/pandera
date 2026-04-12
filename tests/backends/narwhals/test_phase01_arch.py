"""Behavioral tests for Phase 01 — PR Review Architecture Fixes.

Covers ARCH-01 through ARCH-04:
  ARCH-01: Base ErrorHandler has no ibis logic; NarwhalsErrorHandler subclasses it.
  ARCH-02: validate() defers native materialization to return (no premature _to_frame_kind_nw).
  ARCH-03: NarwhalsErrorHandler is wired into all narwhals backends.
  ARCH-04: container.py is backend-agnostic (_to_frame_kind_nw uses duck-typing, not polars import).
"""

import inspect

import polars as pl
import pytest

# ---------------------------------------------------------------------------
# ARCH-01 / Task 1-01-01
# Base ErrorHandler must contain NO ibis-specific imports or isinstance checks
# ---------------------------------------------------------------------------


def test_base_error_handler_has_no_ibis_references():
    """Base ErrorHandler._count_failure_cases contains no ibis imports or isinstance checks."""
    import pandera.api.base.error_handler as base_mod

    src = inspect.getsource(base_mod)
    assert "ibis" not in src, (
        "pandera/api/base/error_handler.py still contains ibis reference"
    )


# ---------------------------------------------------------------------------
# ARCH-01 / Task 1-01-02
# NarwhalsErrorHandler exists, subclasses base ErrorHandler, handles scalars/lists
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
# ARCH-03 / Task 1-02-01
# NarwhalsErrorHandler is wired into container.py, components.py, and base.py
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
# ARCH-04 / Task 1-02-01
# container.py _to_frame_kind_nw is backend-agnostic (no polars-specific type check)
# ---------------------------------------------------------------------------


def test_container_has_no_polars_issubclass_check_in_to_frame_kind():
    """_to_frame_kind_nw uses duck-typing (hasattr), not issubclass(return_type, pl.DataFrame)."""
    import pandera.backends.narwhals.container as container_mod

    src = inspect.getsource(container_mod._to_frame_kind_nw)
    assert "issubclass" not in src, (
        "_to_frame_kind_nw must not use issubclass — use hasattr duck-typing"
    )
    assert "hasattr" in src, (
        "_to_frame_kind_nw must use hasattr for duck-typing lazy vs eager frames"
    )


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
# ARCH-02 / Task 1-03-01
# validate() passes check_lf (nw.LazyFrame) to subsample() — no premature materialization
# ---------------------------------------------------------------------------


def test_validate_does_not_materialize_before_subsample():
    """In validate(), _to_frame_kind_nw is not called before subsample().

    The source of validate() is inspected: the first call to _to_frame_kind_nw
    must appear AFTER the call to self.subsample(check_lf, ...).
    """
    from pandera.backends.narwhals.container import DataFrameSchemaBackend

    src = inspect.getsource(DataFrameSchemaBackend.validate)
    lines = src.splitlines()

    subsample_lineno = next(
        (i for i, line in enumerate(lines) if "self.subsample(" in line), None
    )
    # _to_frame_kind_nw NOT in a return or drop_invalid_rows block means premature call
    premature_calls = [
        i
        for i, line in enumerate(lines)
        if "_to_frame_kind_nw(" in line
        and "return" not in line
        and "drop_invalid_rows" not in line
        and "SchemaErrors" not in line
        and (subsample_lineno is None or i < subsample_lineno)
    ]
    assert not premature_calls, (
        f"_to_frame_kind_nw called before subsample() at source lines: {premature_calls}"
    )


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
# CLEAN-01 / v1.2 Phase 1
# checks.py must contain no polars-specific imports
# ---------------------------------------------------------------------------


def test_checks_has_no_polars_import():
    """narwhals/checks.py must not import polars at module level or inside functions.

    Polars is an optional dependency. checks.py uses narwhals operations only;
    any polars import here would break ibis-only environments.
    """
    import inspect

    import pandera.backends.narwhals.checks as checks_mod

    src = inspect.getsource(checks_mod)
    assert "import polars" not in src, (
        "pandera/backends/narwhals/checks.py must not contain 'import polars'"
    )


# ---------------------------------------------------------------------------
# CLEAN-02 / v1.2 Phase 1
# container.py must not reach into pandera.api.polars.components
# ---------------------------------------------------------------------------


def test_container_has_no_polars_components_import():
    """narwhals/container.py must not import from pandera.api.polars.components.

    The narwhals backend is framework-agnostic. Column class lookup is now
    delegated to schema.infer_columns() in the schema API layer.
    """
    import inspect

    import pandera.backends.narwhals.container as container_mod

    src = inspect.getsource(container_mod)
    assert "pandera.api.polars.components" not in src, (
        "pandera/backends/narwhals/container.py must not reference pandera.api.polars.components"
    )
    assert "importlib.import_module" not in src, (
        "pandera/backends/narwhals/container.py must not use importlib to look up Column class"
    )


def test_container_uses_infer_columns_for_schema_components():
    """collect_schema_components must call schema.infer_columns() — not importlib."""
    import inspect

    from pandera.backends.narwhals.container import DataFrameSchemaBackend

    src = inspect.getsource(DataFrameSchemaBackend.collect_schema_components)
    assert "schema.infer_columns(" in src, (
        "collect_schema_components must call schema.infer_columns() to obtain Column objects"
    )


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
