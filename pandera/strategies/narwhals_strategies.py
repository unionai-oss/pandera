"""Hypothesis strategies for the Narwhals-backed schemas (polars + ibis).

The Narwhals backend does not yet have its own dtype-aware data
generator. Instead, this module reuses the existing pandas strategies
infrastructure:

1. Translate each column's polars / ibis dtype into a numpy / pandas
   equivalent dtype.
2. Build a parallel pandas-flavoured ``DataFrameSchema``.
3. Call :func:`pandera.strategies.pandas_strategies.dataframe_strategy`.
4. Convert the resulting pandas DataFrame to the target backend
   (polars eager DataFrame, polars LazyFrame, or ibis memtable).

This keeps every built-in pandera check (``ge``, ``isin``,
``in_range``, ``str_matches``, …) working out of the box because the
pandas strategy already knows how to honour them.

**Limitations**

Only "primitive" dtypes that round-trip cleanly through pandas are
supported (int, uint, float, bool, string, date, datetime). Complex
polars/ibis types such as ``Struct``, ``List``, or
``Categorical`` will raise ``NotImplementedError`` with a message
pointing at the unsupported column.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any

import pandas as pd

from pandera.strategies.base_strategies import (
    HAS_HYPOTHESIS,
    strategy_import_error,
)

if HAS_HYPOTHESIS:
    from hypothesis.strategies import SearchStrategy, composite
else:  # pragma: no cover
    from pandera.strategies.base_strategies import SearchStrategy, composite


# ---------------------------------------------------------------------------
# dtype translation helpers
# ---------------------------------------------------------------------------


def _polars_dtype_to_numpy(polars_dtype) -> Any:
    """Return a pandas-compatible dtype for a polars dtype class/instance.

    Uses polars itself as the source of truth: build an empty
    ``pl.DataFrame`` typed as the requested column and ask polars for the
    pandas dtype it would produce on ``to_pandas()``. This is robust to
    upstream renames (Utf8 -> String, etc.).
    """
    import polars as pl

    try:
        empty = pl.DataFrame(schema={"x": polars_dtype})
        return empty.to_pandas().dtypes["x"]
    except Exception as exc:
        raise NotImplementedError(
            f"Cannot convert polars dtype {polars_dtype!r} to a pandas "
            f"dtype for hypothesis strategy generation."
        ) from exc


def _ibis_dtype_to_numpy(ibis_dtype) -> Any:
    """Return a pandas-compatible dtype for an ibis dtype."""
    try:
        return ibis_dtype.to_pandas()
    except Exception as exc:
        raise NotImplementedError(
            f"Cannot convert ibis dtype {ibis_dtype!r} to a pandas "
            f"dtype for hypothesis strategy generation."
        ) from exc


def _to_pandas_dtype(pandera_dtype) -> Any:
    """Best-effort translation of a pandera dtype object to a pandas dtype.

    Handles polars-engine dtypes (whose ``.type`` is a polars dtype) and
    ibis-engine dtypes (whose ``.type`` is an ibis dtype). Falls back to
    treating ``pandera_dtype.type`` itself as a pandas/numpy dtype if no
    backend module can be identified.
    """
    if pandera_dtype is None:
        return None

    raw = getattr(pandera_dtype, "type", pandera_dtype)
    module = getattr(raw, "__module__", "") or type(raw).__module__

    if module.startswith("polars"):
        return _polars_dtype_to_numpy(raw)
    if module.startswith("ibis"):
        return _ibis_dtype_to_numpy(raw)
    return raw


# ---------------------------------------------------------------------------
# Schema cloning
# ---------------------------------------------------------------------------


def _clone_columns_for_pandas(columns: dict) -> dict:
    """Return a parallel ``dict[str, pandas Column]`` for pandas strategies.

    Each input column's dtype is translated to a pandas-compatible dtype
    and wrapped in a ``pandera.api.pandas.components.Column`` whose
    other attributes (checks, nullable, unique, regex) are copied across.
    The original schema's columns are left untouched.
    """
    from pandera.api.pandas.components import Column as PandasColumn

    cloned: dict = {}
    for name, col in columns.items():
        target = _to_pandas_dtype(col.dtype)
        if target is None:
            raise NotImplementedError(
                f"Hypothesis strategy generation requires a dtype for "
                f"column {name!r}; got None."
            )
        try:
            new_col = PandasColumn(
                target,
                checks=deepcopy(col.checks),
                nullable=col.nullable,
                unique=getattr(col, "unique", False),
                regex=getattr(col, "regex", False),
                name=name,
            )
        except Exception as exc:
            raise NotImplementedError(
                f"Hypothesis strategy generation does not support "
                f"column {name!r} with dtype {col.dtype!r}: {exc}"
            ) from exc
        cloned[name] = new_col
    return cloned


# ---------------------------------------------------------------------------
# Conversion of generated pandas DataFrame -> target backend
# ---------------------------------------------------------------------------


def _to_target(df: pd.DataFrame, target: str):
    """Convert a generated pandas DataFrame into the requested backend type.

    ``target`` is one of:

    - ``"pandas"``: return the dataframe unchanged
    - ``"polars"`` / ``"polars_eager"``: ``polars.DataFrame``
    - ``"polars_lazy"``: ``polars.LazyFrame``
    - ``"ibis"``: ``ibis.memtable``
    """
    if target == "pandas":
        return df
    if target in ("polars", "polars_eager"):
        import polars as pl

        return pl.from_pandas(df)
    if target == "polars_lazy":
        import polars as pl

        return pl.from_pandas(df).lazy()
    if target == "ibis":
        import ibis

        return ibis.memtable(df)
    raise ValueError(f"Unknown target {target!r}")


# ---------------------------------------------------------------------------
# Public strategies
# ---------------------------------------------------------------------------


def _ensure_pandas_check_backends_registered() -> None:
    """Register pandas check backends so ``Check(check_obj=pd.Series)`` works.

    The pandas strategies fall back to ``strategy.filter(check)`` for
    checks without a registered hypothesis strategy, which calls
    ``check(pd.Series)`` to evaluate. That requires the pandas check
    backend to be registered, which is normally only done lazily when a
    pandas schema is instantiated. Here we proactively register both
    ``pd.DataFrame`` and ``pd.Series`` backends.
    """
    from pandera.api.pandas.array import SeriesSchema
    from pandera.api.pandas.container import (
        DataFrameSchema as _PandasDataFrameSchema,
    )

    _PandasDataFrameSchema.register_default_backends(pd.DataFrame)
    SeriesSchema.register_default_backends(pd.Series)


@strategy_import_error
def dataframe_strategy(
    pandera_dtype=None,
    strategy: SearchStrategy | None = None,
    *,
    columns: dict | None = None,
    checks: list | None = None,
    unique: list | None = None,
    index=None,
    size: int | None = None,
    n_regex_columns: int = 1,
    target: str = "polars",
):
    """Generate dataframes for narwhals-backed schemas (polars / ibis).

    This is a thin wrapper around
    :func:`pandera.strategies.pandas_strategies.dataframe_strategy` that
    converts the generated pandas frame into the requested ``target``
    backend. The check-aware data generation logic lives entirely in the
    pandas strategies module.
    """
    from pandera.strategies import pandas_strategies as pst

    _ensure_pandas_check_backends_registered()

    columns = {} if columns is None else columns
    pandas_columns = _clone_columns_for_pandas(columns)
    pandas_dtype = _to_pandas_dtype(pandera_dtype) if pandera_dtype else None

    base_strategy = pst.dataframe_strategy(
        pandas_dtype,
        strategy,
        columns=pandas_columns,
        checks=checks,
        unique=unique,
        index=index,
        size=size,
        n_regex_columns=n_regex_columns,
    )

    @composite
    def _strategy(draw):
        df = draw(base_strategy)
        return _to_target(df, target)

    return _strategy()


# ---------------------------------------------------------------------------
# Helpers for the schema/.example() integration
# ---------------------------------------------------------------------------


def example(
    schema,
    *,
    target: str,
    size: int | None = None,
    n_regex_columns: int = 1,
):
    """Generate a single example for a narwhals-backed schema.

    Suppresses ``hypothesis.NonInteractiveExampleWarning`` so the
    behaviour mirrors the pandas implementation.
    """
    import hypothesis

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", category=hypothesis.errors.NonInteractiveExampleWarning
        )
        return strategy_for_schema(
            schema, target=target, size=size, n_regex_columns=n_regex_columns
        ).example()


def strategy_for_schema(
    schema,
    *,
    target: str,
    size: int | None = None,
    n_regex_columns: int = 1,
):
    """Build a hypothesis strategy from a narwhals-backed schema.

    Used by polars/ibis ``DataFrameSchema.strategy()`` to delegate to the
    pandas backend.
    """
    return dataframe_strategy(
        schema.dtype,
        columns=schema.columns,
        checks=schema.checks,
        unique=schema.unique,
        index=schema.index,
        size=size,
        n_regex_columns=n_regex_columns,
        target=target,
    )
