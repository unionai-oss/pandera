"""Registry and entry-point loader for data-synthesis backends.

Backends are registered under a string name either programmatically via
:func:`register_synthesis_backend` or declaratively through the
``pandera.synthesis_backends`` packaging entry-point group. Discovery is lazy:
entry points are only scanned when a requested name is not already registered,
so importing pandera never imports third-party backends.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from pandera.api.synthesis.protocol import SynthesisBackend

_SYNTHESIS_BACKENDS: dict[str, SynthesisBackend] = {}
_ENTRY_POINTS_LOADED = False


def register_synthesis_backend(name: str, backend: SynthesisBackend) -> None:
    """Register a synthesis backend under a string name (additive, opt-in).

    Raises ``ValueError`` if ``name`` is already registered by a *different*
    object. Re-registering the identical object is a no-op (idempotent), so
    importing a schema module twice does not raise.

    :param name: backend name used in ``synthesis_backend="..."``.
    :param backend: a :class:`~pandera.api.synthesis.protocol.SynthesisBackend`.
    :raises ValueError: if ``name`` maps to a different backend object.
    """
    existing = _SYNTHESIS_BACKENDS.get(name)
    if existing is not None and existing is not backend:
        raise ValueError(
            f"Synthesis backend {name!r} is already registered as {existing!r}."
        )
    _SYNTHESIS_BACKENDS[name] = backend


def _load_entry_point_backends() -> None:
    """Load all installed ``pandera.synthesis_backends`` entry points once."""
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - py<3.10 fallback
        return
    eps = entry_points(group="pandera.synthesis_backends")
    for ep in eps:
        if ep.name in _SYNTHESIS_BACKENDS:  # pragma: no cover - idempotent
            continue
        try:
            obj = ep.load()  # lazy; never at import time
        except Exception as exc:  # broken third-party backend
            warnings.warn(
                f"Failed to load synthesis backend {ep.name}: {exc}",
                stacklevel=2,
            )
            continue
        _SYNTHESIS_BACKENDS[ep.name] = obj


def get_synthesis_backend(name: str | None) -> SynthesisBackend | None:
    """Resolve a backend by name.

    ``None`` selects the default (pure-``hypothesis``) path. Unknown names are
    resolved against installed third-party entry points once, then raise
    :class:`~pandera.errors.SynthesisBackendNotFoundError` if still unknown.

    :param name: backend name, or ``None`` for the default path.
    :returns: the resolved backend, or ``None``.
    :raises pandera.errors.SynthesisBackendNotFoundError: if ``name`` is
        unknown after scanning installed entry points.
    """
    if name is None:
        return None
    backend = _SYNTHESIS_BACKENDS.get(name)
    if backend is not None:
        return backend
    _load_entry_point_backends()
    backend = _SYNTHESIS_BACKENDS.get(name)
    if backend is None:
        from pandera.errors import SynthesisBackendNotFoundError

        raise SynthesisBackendNotFoundError(
            f"No data synthesis backend named {name!r} is registered. "
            "Install the corresponding package (e.g. pandera[strategy-faker]) "
            "or register it with "
            "pandera.api.synthesis.register_synthesis_backend."
        )
    return backend


def _registered_backend_names() -> set[str]:
    """Internal helper returning currently-registered backend names."""
    return set(_SYNTHESIS_BACKENDS)


def _reset_registry_for_tests() -> None:
    """Internal helper: clear the registry so tests start from a clean slate."""
    global _ENTRY_POINTS_LOADED
    _SYNTHESIS_BACKENDS.clear()
    _ENTRY_POINTS_LOADED = False


__all__ = [
    "register_synthesis_backend",
    "get_synthesis_backend",
]
