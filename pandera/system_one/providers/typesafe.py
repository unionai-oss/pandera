"""TypeSafe (Jev) provider.

Translates pandera's provider-neutral questions into ``typesafe_sdk`` ones and
normalizes the answers back. Everything vendor-specific lives here: nothing
else in pandera imports ``typesafe_sdk``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pandera.errors import SchemaInitError
from pandera.system_one.questions import (
    Choice,
    Decision,
    Noul,
    ProviderLimits,
    Question,
    Score,
)

_INSTALL_HINT = (
    "The TypeSafe provider requires the typesafe-ai extra: "
    "`pip install 'pandera[typesafe-ai]'`."
)

# Published limits, used to pace requests. Conservative by design: exceeding
# them costs a 429 and a backoff, under-running them costs a little latency.
_LIMITS = ProviderLimits(
    max_concurrency=16,
    requests_per_minute=1200,
    tokens_per_second=250_000,
    max_state_tokens=32_000,
)


class TypeSafeProvider:
    """Answers questions with a TypeSafe System One model."""

    def __init__(
        self,
        model: str = "jev-latest",
        *,
        client: Any = None,
        api_key: str | None = None,
        max_concurrency: int | None = None,
    ):
        self._client = client
        self._api_key = api_key
        self._max_concurrency = max_concurrency
        self.id = f"typesafe:{model}"
        #: The resolved model version. ``jev-latest`` is a moving target, so
        #: anything keying on the model -- caches, cassettes -- should pin a
        #: version instead.
        self.model_version = model

    @property
    def limits(self) -> ProviderLimits:
        if self._max_concurrency is None:
            return _LIMITS
        return ProviderLimits(
            max_concurrency=self._max_concurrency,
            requests_per_minute=_LIMITS.requests_per_minute,
            tokens_per_second=_LIMITS.tokens_per_second,
            max_state_tokens=_LIMITS.max_state_tokens,
        )

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = _build_client(self._api_key)
        return self._client

    def compile(self, questions: Mapping[str, Question]) -> Any:
        sdk = _import_sdk()
        compiled: dict[str, Any] = {}
        for name, question in questions.items():
            if isinstance(question, Noul):
                criteria = None
                if question.true_description or question.false_description:
                    criteria = {
                        "true": question.true_description,
                        "false": question.false_description,
                    }
                compiled[name] = sdk.Noul(
                    instructions=question.instructions, criteria=criteria
                )
            elif isinstance(question, Choice):
                compiled[name] = sdk.Choice(
                    instructions=question.instructions,
                    criteria={
                        label: description
                        for label, description in question.criteria
                    },
                )
            elif isinstance(question, Score):
                compiled[name] = sdk.Score(
                    instructions=question.instructions,
                    criteria=list(question.criteria),
                )
            else:  # pragma: no cover - defensive
                raise SchemaInitError(
                    f"TypeSafeProvider cannot pose {type(question).__name__}."
                )
        return compiled

    async def decide(
        self, state: Any, compiled: Mapping[str, Any]
    ) -> Mapping[str, Decision]:
        response = await self.client.system_one(
            state, compiled, model=self.model_version
        )
        return {
            name: _to_decision(answer)
            for name, answer in response.answers.items()
        }


def _to_decision(answer: Any) -> Decision:
    """Normalize an SDK answer into a provider-neutral decision."""
    kind = getattr(answer, "type", None)
    if kind == "noul":
        # A noul is a probability; it carries no separate confidence.
        return Decision(value=answer.noul, confidence=None, raw=answer)
    if kind == "choice":
        return Decision(
            value=answer.choice,
            confidence=answer.confidence,
            probabilities=dict(answer.probabilities),
            raw=answer,
        )
    if kind == "score":
        return Decision(
            value=answer.score,
            confidence=answer.confidence,
            probabilities=dict(answer.probabilities),
            raw=answer,
        )
    raise SchemaInitError(  # pragma: no cover - forward compatibility
        f"unrecognized TypeSafe answer type {kind!r}."
    )


def _import_sdk() -> Any:
    try:
        import typesafe_sdk
    except ImportError as exc:  # pragma: no cover - exercised by env
        raise ImportError(_INSTALL_HINT) from exc
    return typesafe_sdk


def _build_client(api_key: str | None) -> Any:
    sdk = _import_sdk()
    if api_key is None:
        return sdk.AsyncTypeSafeClient()
    return sdk.AsyncTypeSafeClient(api_key=api_key)
