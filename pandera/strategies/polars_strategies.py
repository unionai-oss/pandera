"""Data synthesis for Polars schemas.

Native ``strategy()`` / ``example()`` support for the Polars backend, built on
the engine-agnostic value core in :mod:`pandera.strategies._values`. Realistic
(Mode A) ``FieldGenerator``s are reused unchanged; this module only bridges the
Polars dtype to a numpy-compatible form (the core's deterministic default
sampler works on numpy scalars) and assembles native ``pl.DataFrame`` objects.

See ``specs/pluggable-data-synthesis.md`` §6.
"""

from __future__ import annotations

from typing import Any

from pandera.api.synthesis.semantic import SemanticType
from pandera.strategies._values import compile_constraints_to_sampler
from pandera.strategies.pandas_strategies import (
    _aggregate_constraints_for_sampler,
)


def polars_to_numpy_dtype(polars_dtype: Any):
    """Convert a Polars dtype to a numpy dtype for the engine-agnostic core."""
    import polars as pl

    series = pl.Series([], dtype=polars_dtype)
    try:
        out = series.to_numpy().dtype
    except Exception:  # pragma: no cover - object/unknown dtypes
        import numpy as np

        out = np.dtype(object)
    # Polars string dtypes come back as object; keep them as text for
    # generation (the final Series is built with the Polars dtype).
    return out


def _column_values(
    polars_dtype: Any,  # native polars dtype class (e.g. pl.Int64)
    checks: Any,
    *,
    semantic: SemanticType | None,
    synthesis_backend: str | None,
    seed: int | None,
    size: int,
) -> list:
    """Draw ``size`` native scalar values for one Polars column."""
    constraints = _aggregate_constraints_for_sampler(list(checks or []))
    np_dtype = polars_to_numpy_dtype(polars_dtype)
    sampler = compile_constraints_to_sampler(
        np_dtype,
        constraints,
        semantic=semantic,
        synthesis_backend=synthesis_backend,
        seed=seed,
    )
    return [next(sampler) for _ in range(size)]


class PolarsDataFrameStrategy:
    """Thin helper exposing ``strategy().example()`` semantics for Polars.

    Pandas's ``strategy()`` returns a hypothesis ``SearchStrategy``; Polars has
    no such analog, so this object carries a zero-argument builder and returns
    a native ``pl.DataFrame`` from ``.example()``.
    """

    def __init__(self, builder):
        self._builder = builder

    def example(self):
        """Build and return a native ``pl.DataFrame``."""
        return self._builder()


def dataframe_strategy(
    columns: dict,
    *,
    size: int | None = None,
    n_regex_columns: int = 1,
    synthesis_backend: str | None = None,
    seed: int | None = None,
) -> PolarsDataFrameStrategy:
    """Return a ``strategy`` producing a native Polars DataFrame."""
    del n_regex_columns  # regex columns not specialised here

    def _build():
        import polars as pl

        n = size if size is not None else 5
        series = {}
        for name, col in columns.items():
            if col.regex:
                raise NotImplementedError(
                    "Data synthesis for Polars regex columns is not supported."
                )
            col_seed = (
                None if seed is None else (seed + hash(name)) & 0xFFFFFFFF
            )
            values = _column_values(
                col.dtype.type,
                col.checks,
                semantic=getattr(col, "semantic", None),
                synthesis_backend=synthesis_backend,
                seed=col_seed,
                size=n,
            )
            series[name] = pl.Series(name, values, dtype=col.dtype.type)

        return pl.DataFrame(series)

    return PolarsDataFrameStrategy(_build)


__all__ = ["dataframe_strategy", "polars_to_numpy_dtype"]
