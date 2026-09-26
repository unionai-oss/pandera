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
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Protocol, Union, runtime_checkable

from pandera.errors import SchemaInitError
from pandera.system_one.primitives import Decision, ProviderLimits, Question

PROVIDER_ENV_VAR = "PANDERA_SYSTEM_ONE_PROVIDER"
ENABLED_ENV_VAR = "PANDERA_SYSTEM_ONE_ENABLED"


class SystemOneConfigError(SchemaInitError):
    """Raised when a System One column is validated with no provider set."""


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


def _coerce(value: Union[str, DecisionProvider]) -> DecisionProvider:
    """Turn a provider string into a provider instance."""
    if not isinstance(value, str):
        return value

    name, _, model = value.partition(":")
    if name == "typesafe":
        from pandera.system_one.providers.typesafe import TypeSafeProvider

        return TypeSafeProvider(model or "jev-latest")
    if name == "mock":
        from pandera.system_one.providers.mock import MockProvider

        return MockProvider(seed=int(model) if model else 0)
    raise SystemOneConfigError(
        f"unknown System One provider {value!r}. Known prefixes: "
        "'typesafe:' (e.g. 'typesafe:jev-1.13.0') and 'mock:'. Pass a "
        "DecisionProvider instance for anything else."
    )
