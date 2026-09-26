"""Provider-neutral question and answer types.

A System One model answers typed questions and can only return values from the
schema it was given, so these three question shapes are the entire vocabulary.
They are deliberately independent of any provider SDK: a
:class:`~pandera.system_one.providers.base.DecisionProvider` translates them on
the way out and normalizes answers on the way back.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Union


@dataclasses.dataclass(frozen=True)
class Noul:
    """A yes/no question, answered with a calibrated probability."""

    instructions: str
    true_description: str | None = None
    false_description: str | None = None

    kind: str = dataclasses.field(default="noul", init=False)


@dataclasses.dataclass(frozen=True)
class Choice:
    """A question that selects one of a set of named options."""

    instructions: str
    criteria: tuple[tuple[str, str | None], ...]
    """Option label paired with its description, in declaration order."""

    values: tuple[Any, ...] = ()
    """The column value each label maps back to, positionally."""

    allow_none: bool = False
    """Whether abstention is a valid answer."""

    kind: str = dataclasses.field(default="choice", init=False)

    @property
    def options(self) -> tuple[str, ...]:
        return tuple(label for label, _ in self.criteria)


@dataclasses.dataclass(frozen=True)
class Score:
    """A question that places the state on an ordered scale.

    ``criteria`` is one description per level, starting at zero -- the wire
    format every System One provider uses. ``levels`` carries the value each
    position maps back to, so an ``IntEnum`` whose members are not ``0..n-1``
    still round-trips.
    """

    instructions: str
    criteria: tuple[str, ...]
    levels: tuple[Any, ...]

    kind: str = dataclasses.field(default="score", init=False)


Question = Union[Noul, Choice, Score]


@dataclasses.dataclass(frozen=True)
class Decision:
    """One answer, normalized across providers."""

    value: Any
    """The answer in the question's own terms: the option label for a
    :class:`Choice`, the (possibly fractional) level position for a
    :class:`Score`, the probability of yes for a :class:`Noul`. Mapping it onto
    the column's value domain is the parser's job, not the provider's."""

    confidence: float | None = None
    """How certain the model is, from 0 to 1, when the provider reports it."""

    probabilities: dict[Any, float] = dataclasses.field(default_factory=dict)
    """Probability of each option, when the provider reports them."""

    raw: Any = None
    """The provider's unmodified answer, for debugging."""


@dataclasses.dataclass(frozen=True)
class ProviderLimits:
    """Throughput limits a provider publishes, used to pace requests.

    Every field is optional, because a local model has no rate limit and a
    hosted one may publish only some. ``None`` means "not limited".
    """

    max_concurrency: int = 16
    requests_per_minute: int | None = None
    tokens_per_second: int | None = None
    max_state_tokens: int | None = None


@dataclasses.dataclass(frozen=True)
class ProviderCapabilities:
    """What a provider's model can be *asked*, as opposed to how fast.

    Decision models differ here even when they share a wire format: a small
    local model may take fewer options per question than a hosted one, and a
    classifier-only model may not answer scores at all. The defaults are the
    question vocabulary's own bounds -- two to 255 options, two to ten levels
    -- which every provider speaking the shared wire format inherits, so a
    provider only declares where it is *narrower*.

    A provider is checked against these before any request is made, so an
    unsupported question is an error naming the column rather than a failure
    from the server on row one.
    """

    kinds: frozenset[str] = frozenset({"choice", "score", "noul"})
    """Question kinds the model answers."""

    max_options: int | None = 255
    """Most options a single :class:`Choice` may offer."""

    max_levels: int | None = 10
    """Most levels a single :class:`Score` may have."""

    max_questions: int | None = None
    """Most questions one request may carry, when the provider caps it."""
