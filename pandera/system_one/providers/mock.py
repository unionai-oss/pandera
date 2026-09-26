"""Deterministic providers for tests, docs and offline development.

None of these touch the network. ``MockProvider`` answers from a seeded hash of
the state, so the same input always yields the same decision without an API key;
``RecordingProvider``/``ReplayProvider`` capture and serve real answers.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from pandera.errors import SchemaInitError
from pandera.system_one.primitives import (
    Choice,
    Decision,
    Noul,
    ProviderCapabilities,
    ProviderLimits,
    Question,
    Score,
)
from pandera.system_one.providers.base import capabilities_of


def _stable_unit(*parts: Any) -> float:
    """A deterministic float in [0, 1) derived from the given parts."""
    payload = json.dumps(parts, sort_keys=True, default=repr).encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


class MockProvider:
    """Answers deterministically from a hash of the state.

    Answers are meaningless but stable, valid for the question's domain, and
    free -- which is what documentation examples and CI need. Use
    ``ReplayProvider`` when a test depends on what a real model would say.
    """

    def __init__(
        self,
        seed: int = 0,
        model_version: str = "mock-1",
        *,
        capabilities: ProviderCapabilities | None = None,
        limits: ProviderLimits | None = None,
    ):
        self.seed = seed
        self.id = f"mock:{seed}"
        self.model_version = model_version
        self.calls = 0
        # Configurable so tests can stand in for a narrower model -- a small
        # local one, say -- without a network or a real model.
        self.capabilities = capabilities or ProviderCapabilities()
        self._limits = limits or ProviderLimits(max_concurrency=8)

    @property
    def limits(self) -> ProviderLimits:
        return self._limits

    def compile(self, questions: Mapping[str, Question]) -> Any:
        return dict(questions)

    async def decide(
        self, state: Any, compiled: Mapping[str, Question]
    ) -> Mapping[str, Decision]:
        self.calls += 1
        return {
            name: self._answer(name, question, state)
            for name, question in compiled.items()
        }

    def _answer(self, name: str, question: Question, state: Any) -> Decision:
        unit = _stable_unit(self.seed, name, state)

        if isinstance(question, Noul):
            return Decision(value=unit, confidence=None, raw=unit)

        if isinstance(question, Choice):
            options = question.options
            index = int(unit * len(options))
            chosen = options[min(index, len(options) - 1)]
            share = 1.0 / len(options)
            return Decision(
                value=chosen,
                confidence=round(0.5 + unit / 2, 4),
                probabilities={option: share for option in options},
                raw=chosen,
            )

        if isinstance(question, Score):
            top = len(question.criteria) - 1
            score = unit * top
            return Decision(
                value=score,
                confidence=round(0.5 + unit / 2, 4),
                probabilities={
                    level: 1.0 / len(question.criteria)
                    for level in range(len(question.criteria))
                },
                raw=score,
            )

        raise SchemaInitError(  # pragma: no cover - defensive
            f"MockProvider cannot answer {type(question).__name__}."
        )


class RecordingProvider:
    """Wraps a real provider and records every answer to a cassette."""

    def __init__(self, inner: Any, cassette: dict[str, Any] | None = None):
        self.inner = inner
        self.cassette: dict[str, Any] = {} if cassette is None else cassette
        self.id = inner.id
        self.model_version = inner.model_version

    @property
    def limits(self) -> ProviderLimits:
        return self.inner.limits

    @property
    def capabilities(self) -> ProviderCapabilities:
        return capabilities_of(self.inner)

    def compile(self, questions: Mapping[str, Question]) -> Any:
        self._questions = dict(questions)
        return self.inner.compile(questions)

    async def decide(
        self, state: Any, compiled: Any
    ) -> Mapping[str, Decision]:
        decisions = await self.inner.decide(state, compiled)
        self.cassette[cassette_key(self.id, self.model_version, state)] = {
            name: {
                "value": decision.value,
                "confidence": decision.confidence,
                "probabilities": decision.probabilities,
            }
            for name, decision in decisions.items()
        }
        return decisions


class ReplayProvider:
    """Serves answers from a cassette; never touches the network.

    A state with no recorded answer raises rather than inventing one, so a
    test that drifts from its cassette fails loudly.
    """

    def __init__(
        self,
        cassette: dict[str, Any],
        id: str = "replay",
        model_version: str = "replay-1",
    ):
        self.cassette = cassette
        self.id = id
        self.model_version = model_version

    @property
    def limits(self) -> ProviderLimits:
        return ProviderLimits(max_concurrency=8)

    def compile(self, questions: Mapping[str, Question]) -> Any:
        return dict(questions)

    async def decide(
        self, state: Any, compiled: Any
    ) -> Mapping[str, Decision]:
        key = cassette_key(self.id, self.model_version, state)
        try:
            recorded = self.cassette[key]
        except KeyError:
            raise KeyError(
                f"no recorded answer for state {state!r}. Re-record the "
                "cassette, or use MockProvider if the exact answer does not "
                "matter."
            ) from None
        return {
            name: Decision(
                value=answer["value"],
                confidence=answer.get("confidence"),
                probabilities=answer.get("probabilities", {}),
                raw=answer,
            )
            for name, answer in recorded.items()
        }


def cassette_key(provider_id: str, model_version: str, state: Any) -> str:
    """Stable key for one (provider, model version, state) triple."""
    payload = json.dumps(
        [provider_id, model_version, state], sort_keys=True, default=repr
    ).encode()
    return hashlib.sha256(payload).hexdigest()
