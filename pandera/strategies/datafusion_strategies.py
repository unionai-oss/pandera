"""Data synthesis for DataFusion schemas.

Native ``strategy()`` / ``example()`` support for the DataFusion backend, built
on the engine-agnostic value core in :mod:`pandera.strategies._values`.
Realistic (Mode A) ``FieldGenerator``s are reused unchanged; this module only
bridges the DataFusion dtype (a pandera narwhals-engine dtype, pyarrow-backed)
to a numpy-compatible form and assembles a native ``datafusion.DataFrame`` via
``datafusion.SessionContext.from_pandas``.

See ``specs/pluggable-data-synthesis.md`` §6.
"""

from __future__ import annotations

from typing import Any

from pandera.api.synthesis.semantic import SemanticType
from pandera.strategies._values import compile_constraints_to_sampler
from pandera.strategies.pandas_strategies import (
    _aggregate_constraints_for_sampler,
)


def datafusion_to_numpy_dtype(pandera_dtype: Any):
    """Convert a DataFusion (pandera narwhals-engine) dtype to numpy.

    :param pandera_dtype: the ``.dtype`` of a
        :class:`~pandera.api.datafusion.components.Column`.
    :returns: a ``numpy.dtype`` suitable for the engine-agnostic value core.
    """
    import numpy as np

    name = str(pandera_dtype)
    bit = getattr(pandera_dtype, "bit_width", None)

    if "Boolean" in name:
        return np.dtype(bool)
    if "String" in name or "Binary" in name:
        # Text / bytes columns are drawn by the core as strings.
        return np.dtype(object)
    if "Datetime" in name or "Date" in name or "Time" in name:
        return np.dtype("datetime64[ns]")
    if "Float" in name:
        return np.dtype(f"float{bit or 64}")
    if "Decimal" in name:
        return np.dtype("float64")
    if "Int" in name:
        return np.dtype(f"int{bit or 64}")
    if "UInt" in name:
        return np.dtype(f"uint{bit or 64}")
    if "List" in name or "Struct" in name or "Map" in name:
        return np.dtype(object)
    # Fall back to a numpy token per bit-width, else object.
    if bit is not None:
        try:
            return np.dtype(f"int{bit}")
        except TypeError:  # pragma: no cover - unknown bit width
            pass
    return np.dtype(object)


def _column_values(
    np_dtype: Any,
    checks: Any,
    *,
    semantic: SemanticType | None,
    synthesis_backend: str | None,
    seed: int | None,
    size: int,
) -> list:
    """Draw ``size`` native scalar values for one DataFusion column."""
    constraints = _aggregate_constraints_for_sampler(list(checks or []))
    sampler = compile_constraints_to_sampler(
        np_dtype,
        constraints,
        semantic=semantic,
        synthesis_backend=synthesis_backend,
        seed=seed,
    )
    return [next(sampler) for _ in range(size)]


class DataFusionDataFrameStrategy:
    """Thin helper exposing ``strategy().example()`` for DataFusion."""

    def __init__(self, builder):
        self._builder = builder

    def example(self):
        return self._builder()


def dataframe_strategy(
    schema,  # the DataFusion DataFrameSchema
    *,
    size: int | None = None,
    n_regex_columns: int = 1,
    synthesis_backend: str | None = None,
    seed: int | None = None,
) -> DataFusionDataFrameStrategy:
    """Return a ``strategy`` producing a native ``datafusion.DataFrame``."""
    del n_regex_columns

    def _build():
        import pandas as pd

        n = size if size is not None else 5
        series = {}
        for name, col in schema.columns.items():
            if col.regex:
                raise NotImplementedError(
                    "Data synthesis for DataFusion regex columns is not "
                    "supported."
                )
            col_seed = (
                None if seed is None else (seed + hash(name)) & 0xFFFFFFFF
            )
            series[name] = _column_values(
                datafusion_to_numpy_dtype(col.dtype),
                col.checks,
                semantic=getattr(col, "semantic", None),
                synthesis_backend=synthesis_backend,
                seed=col_seed,
                size=n,
            )
        pdf = pd.DataFrame(series)

        import datafusion

        session = datafusion.SessionContext()
        return session.from_pandas(pdf)

    return DataFusionDataFrameStrategy(_build)


__all__ = ["dataframe_strategy", "datafusion_to_numpy_dtype"]
