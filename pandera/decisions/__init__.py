"""Decision parsing: fill columns by asking a decision model.

A decision model does not generate text -- it answers typed questions and
can only return values from the schema it was given. That makes it a natural
fit for a derived column: the column's declared type *is* the answer domain.

    import pandera.pandas as pa
    import pandera.decisions as decisions

    class Triage(pa.DataFrameModel):
        ticket_body: str
        department: Department = pa.ParsedField(
            description="Which team should handle this ticket",
            parser=decisions.Choice(),
        )
        is_urgent: bool = pa.ParsedField(
            description="The message conveys time-sensitivity",
            parser=decisions.Noul(),
        )

        class Config:
            parser_source = "ticket_body"

    decisions.set_provider("typesafe:jev-1.13.0")
    triaged = Triage.validate(tickets_df)

Columns sharing a provider and a source are filled by one request per row, not
one per column. The provider is runtime configuration, never part of the
schema, so the same model class runs against a cassette in CI and a live model
in production without being edited.
"""

from pandera.decisions.cache import MemoryCache, SQLiteCache
from pandera.decisions.checks import Holds, holds
from pandera.decisions.inspection import Plan, plan, questions
from pandera.decisions.parsers import (
    Choice,
    Confidence,
    Noul,
    Score,
    stats,
)
from pandera.decisions.primitives import (
    Decision,
    ProviderLimits,
    Question,
)
from pandera.decisions.providers.base import (
    DecisionProvider,
    DecisionsConfigError,
    enabled,
    get_provider,
    provider,
    set_provider,
)
from pandera.decisions.providers.mock import (
    MockProvider,
    RecordingProvider,
    ReplayProvider,
)

__all__ = [
    "Choice",
    "Confidence",
    "Decision",
    "DecisionProvider",
    "Holds",
    "MemoryCache",
    "MockProvider",
    "Noul",
    "Plan",
    "ProviderLimits",
    "Question",
    "RecordingProvider",
    "ReplayProvider",
    "SQLiteCache",
    "Score",
    "DecisionsConfigError",
    "enabled",
    "get_provider",
    "holds",
    "plan",
    "provider",
    "questions",
    "set_provider",
    "stats",
]


def __getattr__(name: str):
    # Imported lazily so that ``import pandera.decisions`` works without the
    # decisions extra installed.
    if name == "TypeSafeProvider":
        from pandera.decisions.providers.typesafe import TypeSafeProvider

        return TypeSafeProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
