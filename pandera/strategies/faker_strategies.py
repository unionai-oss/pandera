"""Built-in Faker-backed data-synthesis backend (Mode A, realistic field values).

The first production backend for the pluggable synthesis layer described in
``specs/pluggable-data-synthesis.md``. It maps a field's ``semantic`` tag to a
realistic `Faker <https://faker.readthedocs.io/>`__ provider and produces a
realistic *base value*; the merged ``FieldConstraints`` are still enforced by
the normal hypothesis machinery (or the engine-agnostic sampler in
:mod:`pandera.strategies._values`), so generated data always satisfies the
schema.

Faker is an **optional** dependency (``pandera[strategy-faker]``); this module
is only imported on demand when the ``"faker"`` backend is requested.
"""

from __future__ import annotations

from typing import Any, cast

from pandera.api.synthesis import register_synthesis_backend
from pandera.api.synthesis.protocol import FieldGenerator, SynthesisBackend
from pandera.api.synthesis.semantic import SemanticType, coerce_semantic
from pandera.strategies.constraints import UNSET, FieldConstraints

try:
    import hypothesis.strategies as _st
except ImportError:  # pragma: no cover - hypothesis is a strategies dep
    _st = None  # type: ignore[assignment]


#: semantic -> Faker provider method name
_SEMANTIC_TO_FAKER: dict[SemanticType, str] = {
    SemanticType.NAME: "name",
    SemanticType.FIRST_NAME: "first_name",
    SemanticType.LAST_NAME: "last_name",
    SemanticType.EMAIL: "email",
    SemanticType.PHONE: "phone_number",
    SemanticType.ADDRESS: "address",
    SemanticType.CITY: "city",
    SemanticType.STATE: "state",
    SemanticType.POSTAL_CODE: "postcode",
    SemanticType.COUNTRY: "country",
    SemanticType.COMPANY: "company",
    SemanticType.JOB: "job",
    SemanticType.BOOLEAN: "boolean",
    SemanticType.UUID: "uuid4",
    SemanticType.CURRENCY: "pricetag",
    SemanticType.DATE: "date_object",
    SemanticType.DATETIME: "date_time",
}

#: Candidate pool size used when an ``isin`` constraint is present: we pre-draw
#: realistic candidates and keep only those inside the allowed set (mirroring
#: the "merge first, generate once" philosophy, no filter chain forms).
_ISIN_POOL_SIZE = 500

#: Canonical email regex, used by the narrow, documented inference channel
#: (§5.4 of the spec): if a ``str_matches`` constraint equals this shape, pair
#: the ``email`` semantic automatically (never overrides an explicit ``semantic``).
_EMAIL_REGEX = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


class _FakerFieldGenerator(FieldGenerator):
    """Field-level realistic generator built on a shared, seedable Faker."""

    def __init__(self, locale: str | list[str] | None = None):
        from faker import Faker

        self._locale = locale
        self._factory = Faker(locale=locale) if locale is not None else Faker()

    def generate(
        self,
        pandera_dtype: Any,
        constraints: FieldConstraints,
        *,
        semantic: SemanticType | str | None = None,
        seed: int | None = None,
    ):
        """Return a realistic strategy / candidate pool, or ``None``.

        ``None`` means "no opinion — use the default hypothesis path".
        """
        semantic = coerce_semantic(semantic)
        if semantic is None:
            semantic = self._infer_semantic(constraints)
        if semantic is None or semantic not in _SEMANTIC_TO_FAKER:
            return None

        if constraints.eq is not UNSET:
            # A provider cannot honour an arbitrary ``eq`` value.
            return None

        if seed is not None:
            self._factory.seed_instance((seed + hash(semantic)) & 0xFFFFFFFF)

        provider_name = _SEMANTIC_TO_FAKER[semantic]
        provider = getattr(self._factory, provider_name)

        # ``isin``: pre-compute a bounded candidate pool inside the allowed set
        # so no filter chain forms downstream.
        if constraints.isin is not None:
            pool = self._candidates_in_isin(provider, constraints.isin)
            if not pool:
                return None
            return pool

        if _st is not None:
            return _st.builds(provider)
        return None

    @staticmethod
    def _candidates_in_isin(provider, isin: frozenset) -> list:
        """Draw a bounded pool of realistic candidates lying inside ``isin``."""
        seen = set()
        pool: list = []
        for _ in range(_ISIN_POOL_SIZE):
            value = provider()
            if value in isin and value not in seen:
                seen.add(value)
                pool.append(value)
        return pool

    @staticmethod
    def _infer_semantic(constraints: FieldConstraints) -> SemanticType | None:
        """Narrow, unambiguous inference from the schema (§5.4).

        Currently only the canonical email regex is recognised. This is opt-in
        per backend and never overrides an explicit ``semantic``.
        """
        if constraints.regex_fullmatch:
            if all(p == _EMAIL_REGEX for p in constraints.regex_fullmatch):
                return SemanticType.EMAIL
        return None


class FakerBackend:
    """Named ``"faker"`` synthesis backend exposing a Faker field generator."""

    name = "faker"

    def field_generator(self, locale: str | list[str] | None = None):
        """Return a :class:`_FakerFieldGenerator` bound to this backend."""
        return _FakerFieldGenerator(locale=locale)


#: Singleton backend instance, registered on import (the module is only
#: imported on demand, so this never runs at pandera import time).
_faker_backend = FakerBackend()

register_synthesis_backend(
    "faker",
    cast(SynthesisBackend, _faker_backend),  # Mode-A only; no synthesize()
)

#: Public alias so the packaging entry point
#: ``pandera.strategies.faker_strategies:faker_backend`` resolves to the
#: registered singleton.
faker_backend = _faker_backend

__all__ = ["FakerBackend", "faker_backend"]
