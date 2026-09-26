---
file_format: mystnb
---

% pandera documentation for System One parsing

```{currentmodule} pandera.system_one
```

(system-one)=

# System One parsing

A *System One* model does not generate text. It answers typed questions and can
only return values from the schema it was given, so producing an out-of-domain
answer is not representable. That makes it a natural fit for a derived column:
the column's declared type **is** the answer domain.

Install with the extra:

```bash
pip install 'pandera[typesafe-ai]'
```

## Declaring a column

A System One column is declared exactly like any other derived column
({ref}`derived-columns`) — with a parser object instead of a callable:

```{code-cell} python
import enum
import pandas as pd
import pandera.pandas as pa
import pandera.system_one as system_one

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
        parser=system_one.Choice(),
    )
    frustration: Frustration = pa.ParsedField(
        description="How frustrated the customer appears",
        parser=system_one.Score(),
    )
    is_urgent: bool = pa.ParsedField(
        description="The message conveys time-sensitivity",
        parser=system_one.Noul(),
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
system_one.set_provider("typesafe:jev-1.13.0")       # process-wide

with system_one.provider(system_one.MockProvider()):  # scoped
    Triage.validate(tickets_df)
```

`PANDERA_SYSTEM_ONE_PROVIDER` is the environment-variable form.

**There is no default provider.** Filling a column by asking a model costs
money and time, so validating a System One schema without configuring one
raises `SystemOneConfigError` naming the columns rather than calling out.

Shipped providers:

- `MockProvider` — deterministic answers from a hash of the state. Meaningless
  but stable, valid for the question's domain, and free: what docs and CI need.
- `RecordingProvider` / `ReplayProvider` — capture real answers once, serve them
  offline. A state with no recording fails loudly rather than inventing one.
- `TypeSafeProvider` — the real thing, behind the `typesafe-ai` extra.

Any object with `compile`, `decide` and `limits` works; see
{class}`~pandera.system_one.DecisionProvider`.

## One request per row, not per column

A System One request is one state plus many questions, and a tenth question
costs tokens but almost no time. Columns that share a provider, a source and an
error policy are therefore filled by a **single request per row**:

```{code-cell} python
provider = system_one.MockProvider(seed=7)
frame = pd.DataFrame({"ticket_body": [
    "my card was declined",
    "the API is down",
    "what does the pro plan cost",
]})

with system_one.provider(provider):
    triaged = Triage.validate(frame)

print(triaged)
print(f"{provider.calls} requests for {len(frame)} rows x 3 columns")
```

Columns reading *different* sources form separate batches, one request each per
row.

## Validating the answers

The answers are ordinary column values, so ordinary checks apply — which is the
point. A System One model cannot return an out-of-domain value, so what is
worth checking is not the shape of an individual answer but the shape of the
distribution:

```python
class Triage(pa.DataFrameModel):
    ...

    @pa.dataframe_check
    def routing_is_not_degenerate(cls, df):
        return df["department"].value_counts(normalize=True).max() < 0.9
```

## Confidence

A System One model reports how certain it was. That is a column like any other,
so an ordinary check governs it:

```python
class Triage(pa.DataFrameModel):
    ticket_body: str
    department: Department = pa.ParsedField(
        description="Which team should handle this ticket",
        parser=system_one.Choice(abstain_below=0.55),
        nullable=True,
    )
    department_confidence: float = pa.ParsedField(
        parser=system_one.Confidence("department"),
        ge=0.70,
    )
```

`Confidence` asks nothing of its own -- it reads a decision the column it names
already paid for -- so it adds no request. It must therefore share that
column's request, which it does by default since both resolve the same source.
Naming a column answered in a *different* batch is an error.

Three independent levers:

1. `abstain_below` — per row: an answer the model was unsure of becomes null
   rather than a guess. The column must be `nullable`.
2. `ge=` on the confidence column — per row: a hard failure with ordinary
   failure-case reporting.
3. `@pa.dataframe_check` — per batch: mean confidence, abstention rate,
   distribution drift.

## Caching

Dataframe workloads re-run constantly. The cache key covers the provider, the
resolved model version, the state **and the questions**, so rewording a
`description` correctly invalidates its cached answers:

```python
parser=system_one.Choice(cache="sqlite:///.pandera_system_one.db")
```

`"memory"`, a `sqlite:///` path, or any object with `get`/`set`.

:::{note}
Pandera deep-copies columns into a schema, so a `MemoryCache` instance handed
to one model is not the object another model ends up using. Sharing a cache
across schemas needs shared storage — SQLite, or a custom cache over a global.
:::

Because `jev-latest` is a moving target, pin a version when caching.

## Seeing the cost

Neither of these contacts a provider:

```python
system_one.questions(Triage)
#> {'department': Choice(instructions='Which team should handle this ticket', ...)}

system_one.plan(Triage, tickets_df)
#> Plan(rows=10000, batches=1, requests=10000, questions=3, est_input_tokens=...)
```

`questions()` applies the full inference chain, so what the model will be asked
is reviewable in a test with no credentials. `plan()` reports how many requests
a validation would make before making them.

After validating, the actual cost is attached to the frame:

```python
system_one.stats(triaged)
#> {'rows': 10000, 'batches': 1, 'cached': 9412, 'called': 588,
#>  'seconds': 4.31, 'provider': 'typesafe:jev-1.13.0',
#>  'model_version': 'jev-1.13.0'}
```

## Semantic checks

Judging a column that already exists is a `Check`, not a parser -- it produces
a verdict about values rather than the values themselves:

```python
schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str),
        "category": pa.Column(str),
        "description": pa.Column(str),
    },
    checks=system_one.Holds(
        "The description is a coherent description of a product belonging "
        "to the stated category",
        context=["name", "category", "description"],
        min_probability=0.85,
    ),
)
```

Because it is an ordinary `Check`, failure cases, `lazy=True`,
`n_failure_cases` and `raise_warning` all work untouched.

Attached to a **column**, the column's own value is what gets judged. Attached
to the **dataframe** with `context`, those columns are sent instead -- the only
way to judge a value relative to another column.

Inside a `@pa.dataframe_check` method use `system_one.holds(...)`, which
returns the boolean Series rather than a `Check`:

```python
@pa.dataframe_check
def name_fits_category(cls, df):
    return system_one.holds(
        "The name fits the category",
        context=["name", "category"],
    )(df)
```

Setting `PANDERA_SYSTEM_ONE_ENABLED=0` makes semantic checks pass with a
warning instead of calling out, so a schema carrying them still runs offline.
