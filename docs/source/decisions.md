---
file_format: mystnb
---

% pandera documentation for decision parsing

```{currentmodule} pandera.decisions
```

(decisions)=

# Decision parsing

A decision model does not generate text. It answers typed questions and can
only return values from the schema it was given, so producing an out-of-domain
answer is not representable. That makes it a natural fit for a derived column:
the column's declared type **is** the answer domain.

Install with the extra:

```bash
pip install 'pandera[decisions]'
```

## Declaring a column

A decision column is declared exactly like any other derived column
({ref}`derived-columns`) — with a parser object instead of a callable:

```{code-cell} python
import enum
import pandas as pd
import pandera.pandas as pa
import pandera.decisions as decisions

class Department(enum.StrEnum):
    billing = "billing"
    """Payment, invoices or subscription issues."""
    technical = "technical"
    """Bugs, outages or integration problems."""
    sales = "sales"
    """Pricing, plans or account expansion."""

class Frustration(enum.IntEnum):
    calm = 0
    """Calm, simply stating facts."""
    annoyed = 1
    """Frustrated but civil."""
    angry = 2
    """Very angry, strong language."""

class Triage(pa.DataFrameModel):
    ticket_body: str
    department: Department = pa.ParsedField(
        description="Which team should handle this ticket",
        parser=decisions.Choice(),
    )
    frustration: Frustration = pa.ParsedField(
        description="How frustrated the customer appears",
        parser=decisions.Score(),
    )
    is_urgent: bool = pa.ParsedField(
        description="The message conveys time-sensitivity",
        parser=decisions.Noul(),
    )

    class Config:
        parser_source = "ticket_body"
```

Nothing here is repeated: the options come from `Department`, the rubric from
`Frustration`'s member docstrings, and the questions from each field's
`description`.

## What gets inferred

| Question part | Explicit | Inferred from |
|---|---|---|
| instructions | `instructions=` | the field's `description` |
| `Choice` options | — | the column type's members |
| `Choice` criteria | `criteria={option: text}` | enum member docstrings |
| `Score` levels | — | the ordered type's members, in value order |
| `Score` rubric | `criteria=[...]` | enum member docstrings |
| abstention | — | `nullable=True` on the field |

Anything given explicitly wins. Anything that can be neither given nor inferred
raises `SchemaInitError` naming the column — before any request is made.

Criteria quality is the single biggest lever on answer quality, so prefer
docstrings over bare member names. Keeping them on the enum keeps them in the
schema, versioned alongside the type.

## Choosing a question type

| Parser | Valid column type | Answer |
|---|---|---|
| `Choice()` | `Enum`, `Literal`, categorical (2–255 options) | the selected option |
| `Score()` | an *ordered* type (typically `IntEnum`), 2–10 levels | nearest level, or the raw position for a `float` column |
| `Noul()` | `bool`, or `float` | `True` iff `p >= threshold`, or the probability itself |

Naming the question type rather than inferring it from the column is
deliberate. `Score()` on an unordered type is an error that tells you to use
`Choice()` or make the type ordered — where inferring would have silently asked
a different question than you meant.

## Providers

A schema says *what to ask*. Who answers is runtime configuration, so the same
model class runs against a cassette in CI and a live model in production
without being edited:

```python
decisions.set_provider("typesafe:jev-1.13.0")       # process-wide

with decisions.provider(decisions.MockProvider()):  # scoped
    Triage.validate(tickets_df)
```

`PANDERA_DECISIONS_PROVIDER` is the environment-variable form.

**There is no default provider.** Filling a column by asking a model costs
money and time, so validating a decision schema without configuring one
raises `DecisionsConfigError` naming the columns rather than calling out.

Shipped providers:

- `MockProvider` — deterministic answers from a hash of the state. Meaningless
  but stable, valid for the question's domain, and free: what docs and CI need.
- `RecordingProvider` / `ReplayProvider` — capture real answers once, serve them
  offline. A state with no recording fails loudly rather than inventing one.
- `TypeSafeProvider` — the real thing, behind the `decisions` extra.

Any object with `compile`, `decide` and `limits` works; see
{class}`~pandera.decisions.DecisionProvider`.

## One request per row, not per column

A decision request is one state plus many questions, and a tenth question
costs tokens but almost no time. Columns that share a provider, a source and an
error policy are therefore filled by a **single request per row**:

```{code-cell} python
provider = decisions.MockProvider(seed=7)
frame = pd.DataFrame({"ticket_body": [
    "my card was declined",
    "the API is down",
    "what does the pro plan cost",
]})

with decisions.provider(provider):
    triaged = Triage.validate(frame)

print(triaged)
print(f"{provider.calls} requests for {len(frame)} rows x 3 columns")
```

Columns reading *different* sources form separate batches, one request each per
row.

## Validating the answers

The answers are ordinary column values, so ordinary checks apply — which is the
point. A decision model cannot return an out-of-domain value, so what is
worth checking is not the shape of an individual answer but the shape of the
distribution:

```python
class Triage(pa.DataFrameModel):
    ...

    @pa.dataframe_check
    def routing_is_not_degenerate(cls, df):
        return df["department"].value_counts(normalize=True).max() < 0.9
```
