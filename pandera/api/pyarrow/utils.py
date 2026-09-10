"""Utilities for the pyarrow schema API."""

from __future__ import annotations

from typing import Any

import narwhals.stable.v1 as nw
import pyarrow as pa

from pandera.config import (
    ValidationDepth,
    get_config_context,
    get_config_global,
)
from pandera.engines import narwhals_engine


def pyarrow_dtype_to_narwhals(dtype: pa.DataType) -> Any:
    """Translate a native ``pyarrow.DataType`` into its narwhals equivalent.

    ``narwhals_engine`` resolves strings, narwhals dtypes and abstract pandera
    dtypes, but only a handful of pyarrow types happen to stringify into
    something it recognises: ``pa.int64()`` renders as ``"int64"`` and
    resolves, while ``pa.float64()`` renders as ``"double"`` and
    ``pa.date32()`` as ``"date32[day]"``, neither of which do. Round-tripping a
    zero-length array through narwhals yields an exact translation for every
    pyarrow type, including parametrized ones (timestamps, decimals, lists,
    structs).
    """
    empty = pa.table({"x": pa.array([], type=dtype)})
    return nw.from_native(empty, eager_only=True).schema["x"]


def resolve_dtype(value: Any):
    """Resolve a user-supplied dtype into a narwhals engine ``DataType``.

    Accepts anything ``narwhals_engine`` understands plus native pyarrow
    types, which are translated first.
    """
    if value is None:
        return None
    if isinstance(value, pa.DataType):
        value = pyarrow_dtype_to_narwhals(value)
    return narwhals_engine.Engine.dtype(value)


def get_validation_depth(check_obj: pa.Table) -> ValidationDepth:
    """Get the validation depth for a pyarrow Table.

    Unlike ``pl.LazyFrame`` or ``ibis.Table``, a ``pyarrow.Table`` is always
    fully materialized in memory, so running data-level checks by default
    costs nothing extra — it gets the same treatment as ``pl.DataFrame``.
    Explicit context or global configuration still wins.
    """
    config_ctx = get_config_context(validation_depth_default=None)
    if config_ctx.validation_depth is not None:
        return config_ctx.validation_depth

    config_global = get_config_global()
    if config_global.validation_depth is not None:
        return config_global.validation_depth

    return ValidationDepth.SCHEMA_AND_DATA
