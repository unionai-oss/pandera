---
file_format: mystnb
---

(decision-parsing-tutorial)=

# Parse unstructured data with decision models

Support tickets, help-desk messages, and free-text forms arrive as unstructured
prose, but a pipeline usually needs that text as typed columns — a department,
a severity, a yes/no. This tutorial uses Pandera's
{ref}`decision parsing <decisions>` to read one column of free text and fill
typed columns from it, using {class}`~pandera.decisions.MockProvider` so every
answer is deterministic, offline, and free.

```{code-cell} python
import enum
import pandas as pd
import pandera.pandas as pa
import pandera.decisions as decisions
```

Install the pandas and decisions extras:

```bash
pip install 'pandera[pandas,decisions]'
```

## Declare the columns to derive

Start from the text and the types you want out of it. A
{class}`~pandera.api.pandas.components.ParsedColumn` (or a
{func}`~pandera.api.dataframe.model_components.ParsedField` in a model) is an
ordinary typed column that additionally declares how its values are produced.
When the values are produced by *asking a decision model*, the parser is one of
the {ref}`decision question types <decisions>`: `Choice` for one option out of
a set, `Score` for a position on an ordered scale, and `Noul` for a yes/no.

The options and rubric come from the column's own type. Enum members map to
`Choice` options and `Score` levels, and their docstrings become the criteria
the model is asked against:

```{code-cell} python
class Department(str, enum.Enum):
    billing = "billing"
    """Payment, invoices or subscription issues."""
    technical = "technical"
    """Bugs, outages or integration problems."""
    sales = "sales"
    """Pricing, plans or account expansion."""

class Severity(enum.IntEnum):
    low = 0
    """Minor or informational."""
    medium = 1
    """Blocking but with a workaround."""
    high = 2
    """Service is down or data is at risk."""
```

Each derived column puts the provenance on the column itself: `parser` says
how the column is filled, `description` becomes the question the model is
asked, and `Config.parser_source` says which existing column the question is
about:

```{code-cell} python
class SupportTicket(pa.DataFrameModel):
    message: str
    department: Department = pa.ParsedField(
        description="Which team should handle this ticket",
        parser=decisions.Choice(),
    )
    severity: Severity = pa.ParsedField(
        description="How severe the reported problem is",
        parser=decisions.Score(),
    )
    needs_escalation: bool = pa.ParsedField(
        description="The message requires urgent attention",
        parser=decisions.Noul(),
    )

    class Config:
        parser_source = "message"
```

Because everything is declared on the schema, validating the model *is* the
parsing step — the derived columns are filled and returned alongside the raw
text. What the model will actually be asked is reviewable before any request,
via {func}`~pandera.decisions.questions`, which applies the full inference
chain from the column types and docstrings:

```{code-cell} python
decisions.questions(SupportTicket)
```

## Validate (and parse) with `MockProvider`

Who answers the questions is runtime configuration, never part of the schema.
For a deterministic example we configure the shipped {class}`~pandera.decisions.MockProvider`,
which derives an answer from a hash of the state — always valid for the
question's domain, always the same for the same input, and free:

```{code-cell} python
tickets = pd.DataFrame({
    "message": [
        "my card was declined for no reason",
        "the api is returning 500s and the site is down",
        "what does the pro plan cost",
        "i was double charged last month",
        "login is broken after the update",
    ],
})

with decisions.provider(decisions.MockProvider(seed=7)):
    parsed = SupportTicket.validate(tickets)

parsed
```

The three derived columns (`department`, `severity`, `needs_escalation`) are
now ordinary typed column values, so every normal pandera behavior applies:
coercion, checks, `strict=True`, lazy validation. The provider here returns
domain-valid but arbitrary answers; point Pandera at a real model and the same
schema returns meaningful ones:

```python
# a runtime setting, not a schema change:
decisions.set_provider("typesafe:jev-1.13.0")
decisions.set_provider("ollaya:decider:2b")
```

Switching providers is a change of configuration, never of the model class.

## See the request and the cost before paying

Before running, {func}`~pandera.decisions.plan` reports how many requests a
validation would make and their estimated cost; after running,
{func}`~pandera.decisions.stats` reports what actually happened:

```{code-cell} python
with decisions.provider(decisions.MockProvider(seed=7)):
    plan = decisions.plan(SupportTicket, tickets)
    parsed = SupportTicket.validate(tickets)

print(plan)
print(decisions.stats(parsed))
```

Because the three columns share a provider and a source, they are filled by a
single request per row rather than one per column.

## Use the model's confidence

A decision model reports how certain it was, and that is a column like any
other. `Confidence` reads the answer a named column already paid for, so it
adds no request; `abstain_below` turns low-confidence answers into missing
values instead of guesses (the column must be nullable):

```{code-cell} python
class SupportTicket(pa.DataFrameModel):
    message: str
    department: Department = pa.ParsedField(
        description="Which team should handle this ticket",
        parser=decisions.Choice(abstain_below=0.6),
        nullable=True,
    )
    department_confidence: float = pa.ParsedField(
        parser=decisions.Confidence("department"),
        ge=0.2,
    )

    class Config:
        parser_source = "message"


with decisions.provider(decisions.MockProvider(seed=7)):
    parsed = SupportTicket.validate(tickets)

parsed[["message", "department", "department_confidence"]]
```

An abstained answer becomes `None` (a missing value, never a forced "no"), so a
confidence column stays honest — check its mean, its `ge`, or the abstention
rate with an ordinary check.

## Judging existing values

Decision models are not only for parsing: a model can also *judge* a column
that already exists, as a semantic {func}`~pandera.decisions.Holds` check —
for example that a product description matches its category. Because `Holds`
is an ordinary `Check`, failure cases and lazy validation work untouched. See
{ref}`decisions <decisions>` for the full details.

## Next steps

- The {ref}`decision parsing guide <decisions>` covers every question type,
  real providers, caching, and per-column batching in depth.
- The {ref}`API reference <api-decisions>` documents the whole
  `pandera.decisions` module.
