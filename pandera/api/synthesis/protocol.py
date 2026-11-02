"""Protocols for pluggable data-synthesis backends.

Two small, duck-typed (structural) protocols define the pluggable synthesis
layer described in ``specs/pluggable-data-synthesis.md``:

- :class:`FieldGenerator` — the Mode A building block: produces realistic
  *base values* for a single field.
- :class:`SynthesisBackend` — a named, registered backend (Mode A, Mode B, or
  both). A backend advertises a capability by implementing the corresponding
  protocol method; unimplemented capabilities are simply not used.

``@runtime_checkable`` lets pandera detect capabilities via ``isinstance`` /
``hasattr``-style probing without forcing inheritance, so third parties
implement only the methods relevant to the synthesis mode they support.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Protocol, runtime_checkable

from pandera.api.synthesis.semantic import SemanticType
from pandera.strategies.constraints import FieldConstraints


@runtime_checkable
class FieldGenerator(Protocol):
    """Generates realistic *base values* for a single field.

    Given a pandera dtype, the schema ``FieldConstraints`` (so it can decide
    feasibility without a trial-and-error loop) and an optional explicit
    ``semantic``, it produces either a hypothesis ``SearchStrategy`` yielding
    realistic values, a concrete value or iterable of values, or ``None`` to
    signal "no opinion — use the default hypothesis path".
    """

    def generate(
        self,
        pandera_dtype: Any,
        constraints: FieldConstraints,
        *,
        semantic: SemanticType | None = None,
        seed: int | None = None,
    ) -> Any:
        """Return a generator, a single value, an iterable, or ``None``.

        The return value must be one of:

        - a ``hypothesis.strategies.SearchStrategy`` (accepted as-is),
        - a single concrete value (wrapped in ``st.just``),
        - an iterable of candidate values (wrapped in ``st.sampled_from``),
        - ``None``, meaning "use the default hypothesis path".
        """
        ...


@runtime_checkable
class SynthesisBackend(Protocol):
    """A named, registered data-synthesis backend.

    Backends are registered under a string name and selected via
    ``schema.example(..., synthesis_backend=...)`` or the
    ``PANDERA_SYNTHESIS_BACKEND`` config. Only the ``name`` attribute is
    required; capabilities are advertised by implementing the corresponding
    optional method.
    """

    name: str

    # Mode A (field-level realistic base values). Optional.
    def field_generator(self) -> FieldGenerator: ...

    # Mode B (whole-table statistical synthesis, future). Optional.
    def synthesize(
        self,
        schema: Any,
        *,
        reference_data: Any = None,
        size: int | None = None,
        seed: int | None = None,
    ) -> Any: ...


# Convenience alias for the duck-typed callable shape some backends expose.
FieldStrategyFn = Callable[..., Any]
Sampler = Iterator


__all__ = [
    "FieldGenerator",
    "SynthesisBackend",
    "FieldStrategyFn",
    "Sampler",
]
