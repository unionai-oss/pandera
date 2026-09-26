"""The provider protocol and the runtime configuration that resolves one.

A schema says *what to ask*; who answers is configured out of band. That split
is what lets the same schema run against a recorded cassette in CI and a live
model in production without being edited, and it is why no provider ever
appears in a serialized schema.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Protocol, Union, runtime_checkable

from pandera.errors import SchemaInitError
from pandera.system_one.primitives import (
    Decision,
    ProviderCapabilities,
    ProviderLimits,
    Question,
)

PROVIDER_ENV_VAR = "PANDERA_SYSTEM_ONE_PROVIDER"
ENABLED_ENV_VAR = "PANDERA_SYSTEM_ONE_ENABLED"


class SystemOneConfigError(SchemaInitError):
    """Raised when a System One column is validated with no provider set."""


class ProviderCapabilityError(SchemaInitError):
    """Raised when a provider's model cannot be asked a schema's question."""


@runtime_checkable
class DecisionProvider(Protocol):
    """Something that can answer a set of typed questions about a state."""

    id: str
    """Stable identifier, e.g. ``"typesafe:jev-1.13.0"``."""

    model_version: str
    """Resolved model version. Part of the cache key, so it must be concrete."""

    def compile(self, questions: Mapping[str, Question]) -> Any:
        """Translate questions into the provider's own representation.

        Raises :class:`~pandera.errors.SchemaInitError` for a question the
        provider cannot pose, before any request is made.
        """
        ...  # pragma: no cover

    async def decide(
        self, state: Any, compiled: Any
    ) -> Mapping[str, Decision]:
        """Answer the compiled questions about one state."""
        ...  # pragma: no cover

    @property
    def limits(self) -> ProviderLimits:
        """Throughput limits used to pace requests."""
        ...  # pragma: no cover

    # ``capabilities`` is optional and deliberately not part of the protocol:
    # a provider that says nothing is taken to answer the whole shared
    # vocabulary. Declare it as a :class:`ProviderCapabilities` property to
    # narrow that -- see :func:`capabilities_of`.


def capabilities_of(provider: Any) -> ProviderCapabilities:
    """What a provider can be asked; the shared vocabulary if it does not say."""
    declared = getattr(provider, "capabilities", None)
    return declared if declared is not None else ProviderCapabilities()


def verify_questions(
    questions: Mapping[str, Question],
    provider: DecisionProvider,
    *,
    needs_confidence: Sequence[str] = (),
) -> None:
    """Check a question set against a provider before any request is made.

    Which model answers is runtime configuration, so this cannot happen when
    the schema is built. It happens as soon as the provider is known, which is
    still before the first request, and it names the column and the limit.

    ``needs_confidence`` names columns whose answer's confidence is used --
    for abstention, or reported in a column of its own. A provider that does
    not report one for that kind is refused here: an abstention floor that
    silently never fires is worse than an error.
    """
    caps = capabilities_of(provider)
    who = f"provider {provider.id!r}"

    if caps.max_questions is not None and len(questions) > caps.max_questions:
        raise ProviderCapabilityError(
            f"this request asks {len(questions)} questions, but {who} accepts "
            f"at most {caps.max_questions} per request. Give some of the "
            "columns a different source so they form separate requests."
        )

    for name in needs_confidence:
        kind = questions[name].kind
        if kind in caps.kinds and kind not in caps.reports_confidence:
            raise ProviderCapabilityError(
                f"column '{name}' needs the model's confidence, but {who} "
                f"does not report one for {kind} questions."
                + (
                    " A noul is already a calibrated probability: threshold "
                    "it (`Noul(threshold=...)`) or keep it as a float column "
                    "instead of abstaining."
                    if kind == "noul"
                    else ""
                )
            )

    for name, question in questions.items():
        kind = question.kind
        if kind not in caps.kinds:
            raise ProviderCapabilityError(
                f"column '{name}' asks a {kind} question, which {who} does not "
                f"answer (it answers: {', '.join(sorted(caps.kinds))}). Use a "
                "provider or model that supports it."
            )
        if kind == "choice":
            count = len(question.criteria)  # type: ignore[union-attr]
            if caps.max_options is not None and count > caps.max_options:
                raise ProviderCapabilityError(
                    f"column '{name}' offers {count} options, but {who} "
                    f"accepts at most {caps.max_options} per question. Narrow "
                    "the column's type, or use a model with a larger budget."
                )
        elif kind == "score":
            count = len(question.criteria)  # type: ignore[union-attr]
            if caps.max_levels is not None and count > caps.max_levels:
                raise ProviderCapabilityError(
                    f"column '{name}' has {count} levels, but {who} accepts "
                    f"at most {caps.max_levels} per question."
                )


# The active provider. A ContextVar rather than a module global so that
# ``with provider(...)`` nests correctly and is safe under concurrency.
_provider: contextvars.ContextVar[DecisionProvider | None] = (
    contextvars.ContextVar("pandera_system_one_provider", default=None)
)


def set_provider(value: Union[str, DecisionProvider, None]) -> None:
    """Set the provider for the current process.

    Pass ``None`` to unset, which restores the "no provider configured" state
    where validating a System One column raises rather than calling out.
    """
    _provider.set(None if value is None else _coerce(value))


def get_provider() -> DecisionProvider | None:
    """Return the active provider, or ``None`` if none is configured."""
    current = _provider.get()
    if current is not None:
        return current
    from_env = os.environ.get(PROVIDER_ENV_VAR)
    return _coerce(from_env) if from_env else None


@contextlib.contextmanager
def provider(
    value: Union[str, DecisionProvider, None],
) -> Iterator[DecisionProvider | None]:
    """Use a provider for the duration of a block.

    The scoped form tests are expected to use::

        with system_one.provider(ReplayProvider(cassette)):
            Triage.validate(df)
    """
    resolved = None if value is None else _coerce(value)
    token = _provider.set(resolved)
    try:
        yield resolved
    finally:
        _provider.reset(token)


def require_provider(columns: Sequence[str]) -> DecisionProvider:
    """Return the active provider, or explain how to configure one.

    There is deliberately no default. Filling a column by asking a model costs
    money and time, so reaching that state must require having said so.
    """
    resolved = get_provider()
    if resolved is None:
        named = ", ".join(repr(column) for column in columns)
        raise SystemOneConfigError(
            f"column(s) {named} are filled by a System One model, but no "
            "provider is configured. Set one with "
            '`pandera.system_one.set_provider("typesafe:jev-1.13.0")`, scope '
            "one with `with pandera.system_one.provider(...):`, or set the "
            f"{PROVIDER_ENV_VAR} environment variable."
        )
    return resolved


def enabled() -> bool:
    """Whether System One calls are permitted at all.

    Set ``PANDERA_SYSTEM_ONE_ENABLED=0`` to make schemas carrying semantic
    checks run offline, degrading to a skip instead of an error.
    """
    return os.environ.get(ENABLED_ENV_VAR, "1").lower() not in (
        "0",
        "false",
        "no",
    )


_FACTORIES: dict[str, Callable[[str], DecisionProvider]] = {}


def register_provider(
    prefix: str, factory: Callable[[str], DecisionProvider]
) -> None:
    """Make ``"<prefix>:<model>"`` resolve to a provider.

    This is how a provider that pandera does not ship becomes reachable from
    ``set_provider``, ``PANDERA_SYSTEM_ONE_PROVIDER`` and ``provider(...)``::

        system_one.register_provider("acme", lambda model: AcmeProvider(model))
        system_one.set_provider("acme:decider-2b")

    ``factory`` receives whatever follows the first colon (``""`` if nothing).
    """
    _FACTORIES[prefix] = factory


def _typesafe(model: str) -> DecisionProvider:
    from pandera.system_one.providers.typesafe import TypeSafeProvider

    return TypeSafeProvider(model or "jev-latest")


def _ollaya(model: str) -> DecisionProvider:
    from pandera.system_one.providers.ollaya import OllayaProvider

    return OllayaProvider(model or "laya")


def _mock(model: str) -> DecisionProvider:
    from pandera.system_one.providers.mock import MockProvider

    return MockProvider(seed=int(model) if model else 0)


_BUILTIN_FACTORIES: dict[str, Callable[[str], DecisionProvider]] = {
    "typesafe": _typesafe,
    "ollaya": _ollaya,
    "mock": _mock,
}


def _coerce(value: Union[str, DecisionProvider]) -> DecisionProvider:
    """Turn a provider string into a provider instance."""
    if not isinstance(value, str):
        return value

    name, _, model = value.partition(":")
    # registered providers win, so a built-in prefix can be redirected
    factory = _FACTORIES.get(name) or _BUILTIN_FACTORIES.get(name)
    if factory is not None:
        return factory(model)

    known = ", ".join(
        f"'{p}:'" for p in sorted({*_BUILTIN_FACTORIES, *_FACTORIES})
    )
    raise SystemOneConfigError(
        f"unknown System One provider {value!r}. Known prefixes: {known}. "
        "Pass a DecisionProvider instance, or add a prefix with "
        "`pandera.system_one.register_provider`."
    )
