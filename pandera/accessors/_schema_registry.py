"""Weakref-backed store for pandera schemas on pandas/dask objects."""

from __future__ import annotations

import weakref
from typing import Any

_reg: dict[int, Any] = {}
_refs: dict[int, weakref.ref] = {}


def register_schema(obj: object, schema: Any) -> None:
    """Associate schema with obj without using df.attrs (serializable I/O)."""
    oid = id(obj)
    _reg[oid] = schema
    if oid not in _refs:

        def cleanup(_: weakref.ref) -> None:
            _reg.pop(oid, None)
            _refs.pop(oid, None)

        _refs[oid] = weakref.ref(obj, cleanup)


def get_schema(obj: object) -> Any | None:
    """Return the schema registered for obj, if any."""
    return _reg.get(id(obj))
