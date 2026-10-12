"""Data synthesis for PySpark schemas.

Native ``strategy()`` / ``example()`` support for the PySpark backend, built on
the engine-agnostic value core in :mod:`pandera.strategies._values` and PR6 of
``specs/pluggable-data-synthesis.md`` §6.4.

Counters to the distributed nature of Spark via **driver materialization**
(default): scalar values are drawn on the driver with the deterministic sampler
and materialized with ``spark.createDataFrame(pd_frame)``. This is
byte-reproducible from a ``seed`` and semantically identical to every other
backend.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pandera.api.synthesis.semantic import SemanticType
from pandera.strategies._values import compile_constraints_to_sampler
from pandera.strategies.pandas_strategies import (
    _aggregate_constraints_for_sampler,
)


def spark_dtype_to_numpy(spark_dtype: Any):
    """Convert a PySpark ``DataType`` to a numpy dtype for the value core."""
    from pyspark.sql import types

    if isinstance(spark_dtype, (types.StringType, types.ByteType)):
        return np.dtype(str)
    if isinstance(spark_dtype, types.BooleanType):
        return np.dtype(bool)
    if isinstance(spark_dtype, types.TimestampType):
        return np.dtype("datetime64[ns]")
    if isinstance(spark_dtype, types.DateType):
        return np.dtype("datetime64[D]")
    if isinstance(spark_dtype, types.DoubleType):
        return np.dtype("float64")
    if isinstance(
        spark_dtype, (types.IntegerType, types.LongType, types.ShortType)
    ):
        return np.dtype("int64")
    return np.dtype("object")


def _column_values(
    np_dtype: np.dtype,
    checks: Any,
    *,
    semantic: SemanticType | None,
    synthesis_backend: str | None,
    seed: int | None,
    size: int,
) -> list:
    """Draw ``size`` native scalar values for one PySpark column."""
    constraints = _aggregate_constraints_for_sampler(list(checks or []))
    sampler = compile_constraints_to_sampler(
        np_dtype,
        constraints,
        semantic=semantic,
        synthesis_backend=synthesis_backend,
        seed=seed,
    )
    return [next(sampler) for _ in range(size)]


class PySparkDataFrameStrategy:
    """Thin helper exposing ``strategy().example()`` for PySpark."""

    def __init__(self, builder):
        self._builder = builder

    def example(self):
        return self._builder()


def dataframe_strategy(
    schema,  # the PySpark DataFrameSchema
    *,
    size: int | None = None,
    n_regex_columns: int = 1,
    synthesis_backend: str | None = None,
    seed: int | None = None,
) -> PySparkDataFrameStrategy:
    """Return a ``strategy`` producing a native PySpark DataFrame."""
    del n_regex_columns

    def _build():
        from pyspark.sql import SparkSession

        spark_dtypes = {
            name: col.dtype.type for name, col in schema.columns.items()
        }
        pdf = {}
        for name, col in schema.columns.items():
            col_seed = (
                None if seed is None else (seed + hash(name)) & 0xFFFFFFFF
            )
            pdf[name] = _column_values(
                spark_dtype_to_numpy(spark_dtypes[name]),
                col.checks,
                semantic=getattr(col, "semantic", None),
                synthesis_backend=synthesis_backend,
                seed=col_seed,
                size=size if size is not None else 5,
            )

        import pandas as pd

        pdf = pd.DataFrame(pdf)
        spark = SparkSession.getActiveSession() or (
            SparkSession.builder.master("local[1]")
            .appName("pandera-data-synthesis")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        return spark.createDataFrame(pdf, schema=list(spark_dtypes.keys()))

    return PySparkDataFrameStrategy(_build)


__all__ = ["dataframe_strategy", "spark_dtype_to_numpy"]
