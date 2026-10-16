"""Pluggable data-synthesis backends.

This package defines the pluggable, opt-in synthesis layer described in
``specs/pluggable-data-synthesis.md``. The value-generating path of
``Schema.strategy()`` / ``Schema.example()`` is no longer hard-wired to
``hypothesis`` alone: a thin structural backend interface sits in front of the
leaf where base values are drawn, letting realistic generators (e.g. Faker) and
future whole-table statistical synthesizers (e.g. SDV) hook in without
changing the default, byte-for-byte-compatible behavior.

Public API:
- :func:`register_synthesis_backend` — programmatically register a backend.
- :func:`get_synthesis_backend` — resolve a backend by name.
- :class:`SynthesisBackend` / :class:`FieldGenerator` — structural protocols.
- :class:`SemanticType` — semantic categories driving realistic generation.
"""

from pandera.api.synthesis.protocol import FieldGenerator, SynthesisBackend
from pandera.api.synthesis.registry import (
    get_synthesis_backend,
    register_synthesis_backend,
)
from pandera.api.synthesis.semantic import SemanticType, coerce_semantic

__all__ = [
    "FieldGenerator",
    "SynthesisBackend",
    "SemanticType",
    "coerce_semantic",
    "get_synthesis_backend",
    "register_synthesis_backend",
]
