"""TypeSafe (Jev) provider, and the base for anything speaking its wire format.

Translates pandera's provider-neutral questions into ``typesafe_sdk`` ones and
normalizes the answers back. Everything vendor-specific lives here: nothing
else in pandera imports ``typesafe_sdk``.

The wire format -- one ``state`` plus named ``questions`` in, typed ``answers``
out -- is not TypeSafe's alone: open-weight decision models served locally
(see :mod:`~pandera.system_one.providers.ollaya`) implement the same endpoint,
and the SDK works against them by changing ``base_url``. So this class takes
``base_url`` and is written to be subclassed by a provider that only differs in
where it points and what its model can do.
"""

from __future__ import annotations

import dataclasses
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
    """Answers questions with a TypeSafe System One model.

    :param model: the model to ask. ``jev-latest`` is a moving target, so pin a
        version wherever answers are cached or recorded.
    :param base_url: where to send requests, for any server speaking the same
        wire format. Defaults to the hosted TypeSafe API.
    :param retry: the SDK's own ``RetryPolicy``. Transport retries belong to
        the provider, which knows which of its errors are transient; pandera
        only paces requests.
    :param capabilities: narrows what the model can be asked, when it is
        narrower than the shared vocabulary.
    :param client_options: anything else the SDK's client takes -- ``timeout``,
        ``headers``, ``transport``. A local model on a CPU may want a longer
        ``timeout`` than the SDK's default.
    """

    #: Prefix of :attr:`id`. Subclasses set their own.
    _id_prefix = "typesafe"
    #: Published limits, used to pace requests. Conservative by design:
    #: exceeding them costs a 429 and a backoff, under-running them costs a
    #: little latency.
    _default_limits = _LIMITS
    _default_capabilities = ProviderCapabilities()

    def __init__(
        self,
        model: str = "jev-latest",
        *,
        client: Any = None,
        api_key: str | None = None,
        base_url: str | None = None,
        retry: Any = None,
        max_concurrency: int | None = None,
        capabilities: ProviderCapabilities | None = None,
        **client_options: Any,
    ):
        self._client = client
        self._client_options = client_options
        self._api_key = api_key
        self._base_url = base_url
        self._retry = retry
        self._max_concurrency = max_concurrency
        self._capabilities = capabilities
        self.id = f"{self._id_prefix}:{model}"
        #: The resolved model version. ``jev-latest`` is a moving target, so
        #: anything keying on the model -- caches, cassettes -- should pin a
        #: version instead.
        self.model_version = model

    @property
    def limits(self) -> ProviderLimits:
        if self._max_concurrency is None:
            return self._default_limits
        return dataclasses.replace(
            self._default_limits, max_concurrency=self._max_concurrency
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        if self._capabilities is not None:
            return self._capabilities
        return self._default_capabilities

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = _build_client(
                api_key=self._api_key,
                base_url=self._base_url,
                retry=self._retry,
                **self._client_options,
            )
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


def _build_client(**options: Any) -> Any:
    sdk = _import_sdk()
    # Only what was given, so the SDK's own defaults and environment variables
    # (``TYPESAFE_API_KEY``, ``TYPESAFE_BASE_URL``) still apply to the rest.
    given = {
        name: value for name, value in options.items() if value is not None
    }
    return sdk.AsyncTypeSafeClient(**given)
