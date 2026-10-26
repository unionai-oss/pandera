"""Data synthesis for Ibis schemas.

Native ``strategy()`` / ``example()`` support for the Ibis backend, built on the
engine-agnostic value core in :mod:`pandera.strategies._values`. Realistic
(Mode A) ``FieldGenerator``s are reused unchanged; this module bridges the Ibis
dtype to a numpy-compatible form and assembles an ``ibis.Table`` via
``ibis.memtable``. DataFusion uses the same engine-agnostic core via
:mod:`pandera.strategies.datafusion_strategies`.

See ``specs/pluggable-data-synthesis.md`` §6.
"""

from __future__ import annotations

from typing import Any

from pandera.api.synthesis.semantic import SemanticType
from pandera.strategies._values import compile_constraints_to_sampler
from pandera.strategies.pandas_strategies import (
    _aggregate_constraints_for_sampler,
)


def ibis_to_numpy_dtype(ibis_dtype: Any):
    """Convert an Ibis dtype to a numpy dtype for the value core."""
    import numpy as np

    try:
        return ibis_dtype().to_numpy()
    except Exception:  # pragma: no cover - fall back to object
        s = str(ibis_dtype).lower()
        if s in {"string", "str", "object"}:
            return np.dtype(object)
        if s.startswith(("list", "struct", "map")):
            return np.dtype(object)
        try:
            return np.dtype(s)
        except TypeError:
            return np.dtype(object)


def _column_values(
    ibis_dtype: Any,
    checks: Any,
    *,
    semantic: SemanticType | None,
    synthesis_backend: str | None,
    seed: int | None,
    size: int,
) -> list:
    """Draw ``size`` native scalar values for one Ibis column."""
    constraints = _aggregate_constraints_for_sampler(list(checks or []))
    np_dtype = ibis_to_numpy_dtype(ibis_dtype)
    sampler = compile_constraints_to_sampler(
        np_dtype,
        constraints,
        semantic=semantic,
        synthesis_backend=synthesis_backend,
        seed=seed,
    )
    return [next(sampler) for _ in range(size)]


class IbisTableStrategy:
    """Thin helper exposing ``strategy().example()`` semantics for Ibis."""

    def __init__(self, builder):
        self._builder = builder

    def example(self):
        return self._builder()


def table_strategy(
    schema,  # the DataFrameSchema/IbisSchema
    *,
    size: int | None = None,
    n_regex_columns: int = 1,
    synthesis_backend: str | None = None,
    seed: int | None = None,
) -> IbisTableStrategy:
    """Return a ``strategy`` producing a native ``ibis.Table``."""
    del n_regex_columns

    def _build():
        import pandas as pd

        n = size if size is not None else 5
        series = {}
        for name, col in schema.columns.items():
            if col.regex:
                raise NotImplementedError(
                    "Data synthesis for Ibis regex columns is not supported."
                )
            col_seed = (
                None if seed is None else (seed + hash(name)) & 0xFFFFFFFF
            )
            series[name] = _column_values(
                col.dtype.type if hasattr(col.dtype, "type") else col.dtype,
                col.checks,
                semantic=getattr(col, "semantic", None),
                synthesis_backend=synthesis_backend,
                seed=col_seed,
                size=n,
            )
        pdf = pd.DataFrame(series)
        import ibis

        return ibis.memtable(pdf)

    return IbisTableStrategy(_build)


__all__ = ["table_strategy", "ibis_to_numpy_dtype"]
