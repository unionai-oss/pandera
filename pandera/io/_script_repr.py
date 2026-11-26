"""Render values as source expressions for emitted schema scripts."""

from __future__ import annotations

import math
from typing import Any

_CONTAINERS = (list, tuple, set, frozenset)


def _has_non_finite_float(value: Any) -> bool:
    """Whether ``value`` is, or holds, an infinity or a NaN."""
    if isinstance(value, bool):
        return False
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, _CONTAINERS):
        return any(_has_non_finite_float(item) for item in value)
    if isinstance(value, dict):
        return any(
            _has_non_finite_float(key) or _has_non_finite_float(val)
            for key, val in value.items()
        )
    return False


def script_repr(value: Any) -> str:
    """Source expression for ``value``, for use in a generated script.

    ``repr()`` writes the special floats as ``inf``, ``-inf`` and ``nan``.
    Those are names, not literals, so a script that carries one raises
    ``NameError`` when it is imported; they are written as ``float("inf")``
    and so on instead. A value that holds no such float is left to
    ``repr()``, so the emitted script is unchanged for everything else.
    """
    if not _has_non_finite_float(value):
        return repr(value)
    if isinstance(value, float):
        return f'float("{value}")'
    if isinstance(value, dict):
        items = ", ".join(
            f"{script_repr(key)}: {script_repr(val)}"
            for key, val in value.items()
        )
        return f"{{{items}}}"
    if isinstance(value, list):
        return f"[{', '.join(script_repr(item) for item in value)}]"
    if isinstance(value, tuple):
        items = ", ".join(script_repr(item) for item in value)
        return f"({items},)" if len(value) == 1 else f"({items})"
    items = ", ".join(script_repr(item) for item in value)
    if isinstance(value, frozenset):
        return f"frozenset({{{items}}})"
    return f"{{{items}}}"
