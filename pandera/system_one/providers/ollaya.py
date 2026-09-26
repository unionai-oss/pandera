"""Ollaya provider: open-weight decision models served locally.

`Ollaya <https://ollaya.dev>`_ runs decision models -- Laya, Decider, Kev and
others -- on your own hardware behind the same ``/v1/systemone`` endpoint
TypeSafe serves, so the TypeSafe SDK talks to it unchanged. That makes this a
thin subclass: the same wire format, pointed at a local server, with the
limits and capabilities of a local model instead of a hosted one.

Two differences from a hosted provider matter to pandera:

- **No rate limit and no bill.** Requests are paced by how many the local
  queue will take, not by a published quota, and there is no per-token price.
- **Models differ.** Which question kinds are answered and how many options a
  question may carry depend on the model, so capabilities are per model and can
  be overridden.
"""

from __future__ import annotations

import os
from typing import Any

from pandera.system_one.primitives import ProviderCapabilities, ProviderLimits
from pandera.system_one.providers.typesafe import TypeSafeProvider

HOST_ENV_VAR = "OLLAYA_HOST"
API_KEY_ENV_VAR = "OLLAYA_API_KEY"
DEFAULT_HOST = "127.0.0.1:11435"

# A local queue, not a quota. Ollaya answers QUEUE_FULL when it is saturated,
# so a small fan-out is faster than a large one that spends its time retrying.
_LIMITS = ProviderLimits(max_concurrency=4, max_state_tokens=65_536)


def _capabilities_for(model: str) -> ProviderCapabilities:
    """Ollaya's published per-model option budget.

    ``laya:en`` takes about 125 options per question and ``laya:multilingual``
    about 250; the bare ``laya`` alias routes between them per request, so it
    takes the smaller. Other models get the shared vocabulary's bound, and the
    server's ``TOO_MANY_OPTIONS`` is the backstop if that is optimistic.
    """
    family, _, variant = model.partition(":")
    if family == "laya":
        return ProviderCapabilities(
            max_options=250 if variant == "multilingual" else 125,
            max_questions=256,
            price_per_million_input_tokens=0.0,
        )
    return ProviderCapabilities(
        max_questions=256, price_per_million_input_tokens=0.0
    )


def _default_base_url() -> str:
    host = os.environ.get(HOST_ENV_VAR) or DEFAULT_HOST
    return host if "://" in host else f"http://{host}"


class OllayaProvider(TypeSafeProvider):
    """Answers questions with a decision model served by a local Ollaya.

    ::

        system_one.set_provider("ollaya:laya")

        # or, for a server elsewhere
        system_one.set_provider(
            system_one.OllayaProvider("decider:2b", base_url="http://gpu-box:11435")
        )

    :param model: an Ollaya model name, e.g. ``laya`` or ``decider:2b``.
    :param base_url: the server. Defaults to ``OLLAYA_HOST``, then
        ``127.0.0.1:11435``.
    :param api_key: sent as a bearer token. Defaults to ``OLLAYA_API_KEY``,
        then a placeholder -- Ollaya only checks it when the server was
        started with one.
    """

    _id_prefix = "ollaya"
    _default_limits = _LIMITS

    def __init__(
        self,
        model: str = "laya",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        capabilities: ProviderCapabilities | None = None,
        **options: Any,
    ):
        super().__init__(
            model,
            base_url=base_url or _default_base_url(),
            api_key=api_key or os.environ.get(API_KEY_ENV_VAR) or "local",
            capabilities=capabilities or _capabilities_for(model),
            **options,
        )
