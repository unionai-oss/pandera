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

from pandera.system_one.parsers import Choice, Noul, Score
from pandera.system_one.providers.base import (
    DecisionProvider,
    SystemOneConfigError,
    enabled,
    get_provider,
    provider,
    set_provider,
)
from pandera.system_one.providers.mock import (
    MockProvider,
    RecordingProvider,
    ReplayProvider,
)
from pandera.system_one.questions import (
    Decision,
    ProviderLimits,
    Question,
)

__all__ = [
    "Choice",
    "Decision",
    "DecisionProvider",
    "MockProvider",
    "Noul",
    "ProviderLimits",
    "Question",
    "RecordingProvider",
    "ReplayProvider",
    "Score",
    "SystemOneConfigError",
    "enabled",
    "get_provider",
    "provider",
    "set_provider",
]


def __getattr__(name: str):
    # Imported lazily so that ``import pandera.system_one`` works without the
    # typesafe-ai extra installed.
    if name == "TypeSafeProvider":
        from pandera.system_one.providers.typesafe import TypeSafeProvider

        return TypeSafeProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
