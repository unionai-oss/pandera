"""System One parsing: fill columns by asking a decision model.

A System One model does not generate text -- it answers typed questions and
can only return values from the schema it was given. That makes it a natural
fit for a derived column: the column's declared type *is* the answer domain.

    import pandera.pandas as pa
    import pandera.system_one as system_one

    class Triage(pa.DataFrameModel):
        ticket_body: str
        department: Department = pa.ParsedField(
            description="Which team should handle this ticket",
            parser=system_one.Choice(),
        )
        is_urgent: bool = pa.ParsedField(
            description="The message conveys time-sensitivity",
            parser=system_one.Noul(),
        )

        class Config:
            parser_source = "ticket_body"

    system_one.set_provider("typesafe:jev-1.13.0")
    triaged = Triage.validate(tickets_df)

Columns sharing a provider and a source are filled by one request per row, not
one per column. The provider is runtime configuration, never part of the
schema, so the same model class runs against a cassette in CI and a live model
in production without being edited.
"""

from pandera.system_one.cache import MemoryCache, SQLiteCache
from pandera.system_one.checks import Holds, holds
from pandera.system_one.inspection import Plan, plan, questions
from pandera.system_one.parsers import (
    Choice,
    Confidence,
    Noul,
    Score,
    stats,
)
from pandera.system_one.primitives import (
    Decision,
    ProviderCapabilities,
    ProviderLimits,
    Question,
)
from pandera.system_one.providers.base import (
    DecisionProvider,
    ProviderCapabilityError,
    SystemOneConfigError,
    capabilities_of,
    enabled,
    get_provider,
    provider,
    register_provider,
    set_provider,
)
from pandera.system_one.providers.mock import (
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
    "ProviderCapabilities",
    "ProviderCapabilityError",
    "ProviderLimits",
    "Question",
    "RecordingProvider",
    "ReplayProvider",
    "SQLiteCache",
    "Score",
    "SystemOneConfigError",
    "capabilities_of",
    "enabled",
    "get_provider",
    "holds",
    "plan",
    "provider",
    "questions",
    "register_provider",
    "set_provider",
    "stats",
]


def __getattr__(name: str):
    # Imported lazily so that ``import pandera.system_one`` works without the
    # typesafe-ai extra installed.
    if name == "TypeSafeProvider":
        from pandera.system_one.providers.typesafe import TypeSafeProvider

        return TypeSafeProvider
    if name == "OllayaProvider":
        from pandera.system_one.providers.ollaya import OllayaProvider

        return OllayaProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
