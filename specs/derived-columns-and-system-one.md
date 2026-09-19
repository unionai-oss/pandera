# Derived Columns and System One Parsing — Integration Spec

> **Status:** Draft / RFC
> **Author:** pandera maintainers
> **Install:** `pip install 'pandera[typesafe-ai]'`
> **Related:** [TypeSafe Jev](https://pydantic.dev/docs/ai/models/typesafe/)

---

## 0. TL;DR

Two layers. The lower one is a pandera feature with its own users; the upper one
is a plugin into it.

**Layer 1 — derived columns (pandera core).** Today a `Parser` transforms data
that already exists. Generalize it so a column can declare that it is *derived
from* another column:

```python
class Tickets(pa.DataFrameModel):
    body: str
    n_words: int = pa.ParsedField(source="body", parser=lambda s: s.str.split().str.len())
```

**Layer 2 — System One parsing (`pandera[typesafe-ai]`).** `parser=` accepts a
plain callable *or* a parser object. The System One question types are parser
objects, so asking a decision model is the same construct as any other
derivation:

```python
import enum
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

triaged = Triage.validate(tickets_df)
```

`Choice()`, `Score()`, and `Noul()` take `instructions` and `criteria`
explicitly. Given neither, they infer: instructions from the field's
`description`, options and criteria from the annotated type — `Department`'s
members and their docstrings, `Frustration`'s ordered levels and theirs. Plain
Python carries the type information; the field carries the question.

The three fields share a source and a provider, so the compiler batches them
into **one request per row**, not three (§5.1). The declarations are per column
because that is where they are legible; the batching is pandera's job.

---

## 1. Motivation

### 1.1 Layer 1 is missing from pandera independently

Deriving a column from another column is one of the most common things anyone
does to a dataframe, and pandera has no way to *say* it. You can compute it
before validating, or hide it in a parser closure, but the schema — the thing
that is supposed to describe the data — cannot express that `n_words` comes from
`body`. That costs:

- **Provenance.** A schema does not record which columns are derived, or from
  what. The YAML form is an incomplete description of the pipeline.
- **Error quality.** A parser that reads a column that isn't there raises a bare
  `KeyError: 'nope'` — no schema context, not even a `SchemaError` (§6.1).
- **Ordering.** Chained parsers work only because list order happens to be
  preserved; pandera cannot sort them or detect a cycle, because it does not
  know what any parser reads or writes.
- **Locality.** A derived column's definition lives in a schema-level `parsers`
  list, far from the `Column` it produces.

### 1.2 Layer 2 needs exactly that primitive

Filling `department` from `ticket_body` by asking a model is a derived column.
It differs from `n_words` only in how the values are computed. If layer 1
exists, layer 2 is a handful of parser objects — which is the test of whether
the abstraction is right.

Pandera already draws this line, in `docs/source/parsers.md`:

> Validation is the act of verifying whether data follows some set of
> constraints, whereas parsing transforms raw data into some desired set of
> constraints.

### 1.3 Why Jev fits the layer-2 slot

Jev is a **System One model**: it does not generate text. It answers typed
questions and can only return values from the schema it was given. Three
question types, which become the three parser objects:

| Question | Returns | Shape |
|---|---|---|
| `Noul(instructions=...)` | calibrated probability | `float` in `[0, 1]` |
| `Choice(instructions=..., criteria={opt: desc})` | one option | ≤ 255 options |
| `Score(instructions=..., criteria=[level, ...])` | position on a scale | 2–10 ordered levels |

Vendor-reported: 70–500 ms latency, 1,200 req/min, 250k tokens/sec, $0.042/M
input tokens with output free, a 0% structured-output error rate, and *"a tenth
question costs tokens but almost no time"* — which is what makes per-column
declarations affordable, since the compiler can batch them back together.

The 0% structured-output rate is the load-bearing fact for pandera. It **moves
validation up a level**. There is no question of whether the model returned
`"Billing "`, `"BILLING"`, or a two-paragraph apology; the dtype and the domain
are guaranteed by construction. What remains uncertain is whether the
*distribution* of decisions is sane:

- Is 60% of today's batch routing to one department when the baseline is 4%?
- Did mean confidence on `frustration` drop after the last model version bump?
- Do 30% of rows now fall below the abstention threshold?

Those are `Check`s and `Hypothesis`es over a column — pandera's home turf, and
with no per-object analogue in a per-record API.

### 1.4 Cost

10,000 tickets, ~400 tokens of state each, 6 derived fields: ~4M input tokens ≈
**$0.17** in one pass of ~10,000 requests.

*(All figures in §1.3–§1.4 are vendor-reported. §12 covers verification before
any of them appear in user-facing docs.)*

---

## 2. Design principles

1. **Two layers, and the lower one ships alone.** Derived columns are a pandera
   feature with its own users. If layer 2 is never built, layer 1 still pays for
   itself.
2. **One derivation mechanism.** A System One column is declared exactly like a
   `lambda`-derived column. There is no second path, no separate step to
   remember, and no "AI mode" for a schema.
3. **Plain Python carries the types.** `Enum`, `IntEnum`, `Literal`, and `bool`
   already say everything the question compiler needs. No pandera-specific
   annotation is introduced.
4. **Declarative per column, batched by the engine.** The user writes one
   declaration per column because that is where it is legible; pandera groups
   them into as few requests as possible. Writing per-column should never cost
   more than writing per-schema.
5. **The schema says what to ask; the runtime says who answers.** Providers are
   configured out of band (§4.6), so one schema runs against a recorded cassette
   in CI and a live provider in production without edits.
6. **Unsupported is a schema-build error.** A question that cannot be posed
   raises `SchemaInitError` when the schema is built, before any request.
7. **Deterministic tests.** No test in pandera's suite may require an API key or
   network access.

---

## 3. Layer 1: derived columns

### 3.1 What already works

Verified against `main` (`62f55e2d`), pandas backend. A dataframe-level parser
can already create columns, and the rest of the pipeline handles them correctly:

| Behavior | Result |
|---|---|
| df-level parser adds a declared column | ✅ works |
| ...with `strict=True` | ✅ works |
| ...with `coerce=True` on the created column | ✅ coerced after creation |
| ...with a `Check` on the created column | ✅ runs, reports failure cases |
| ...via `@pa.dataframe_parser` on a model | ✅ works |
| ...with `strict="filter"` dropping the source afterwards | ✅ works |
| chained parsers (`body → a → b`) | ✅ works, by list order |
| **column-level** parser on a not-yet-existing column | ❌ `SchemaError: column 'n_words' not in dataframe` |

The documented pipeline order —

> 1. dataframe-level parsing
> 2. column-level parsing
> 3. dataframe-level checks
> 4. column-level and index-level checks

— puts dataframe-level parsing before everything, and
`pandera/backends/pandas/container.py:806-823` already defers coercion for
columns that have parsers so that "parsers run before coercion".

So layer 1 is not about making derivation *possible*. It is about making it
**declared**.

### 3.2 What is missing

| Gap | Consequence |
|---|---|
| No declared `source` | Reading an absent column raises bare `KeyError: 'nope'` (§6.1) |
| No declared `target` | Pandera cannot tell a derived column from one the caller must supply |
| No dependency graph | Chaining depends on list order; cycles are undetectable |
| No provenance in serialization | A YAML schema omits how derived columns are produced |
| Column-level parsers cannot create their own column | Provenance cannot live next to the column it describes |
| No way for a parser to see its own column's type | Every parser must be told what it is producing |

The last one is what layer 2 turns out to need most.

### 3.3 `ParsedField` and `ParsedColumn`

The declarative spelling, and the primary API for both layers:

```python
class Tickets(pa.DataFrameModel):
    body: str
    n_words: int = pa.ParsedField(
        source="body",
        parser=lambda s: s.str.split().str.len(),
        ge=1,                          # everything Field accepts still works
    )
```

```python
schema = pa.DataFrameSchema({
    "body": pa.Column(str),
    "n_words": pa.ParsedColumn(
        int,
        source="body",
        parser=lambda s: s.str.split().str.len(),
        checks=pa.Check.ge(1),
        coerce=True,
    ),
})
```

```python
def ParsedField(
    *,
    parser: Callable | ColumnParser,
    source: str | list[str] | Callable[[Any], Any] | None = None,
    on_error: Literal["raise", "null", "drop"] = "raise",
    **field_kwargs,                    # description, checks, nullable, coerce, ...
) -> Any: ...
```

`ParsedColumn` is a `Column` subclass and `ParsedField` a `FieldInfo` subclass.
At schema-build time each desugars into (a) an ordinary `Column` and (b) a
`Parser` with `source`/`target` set, registered on the schema in dependency
order. `source` defaults to `Config.parser_source` when the field omits it,
which is what lets every field in the §0 example stay a one-liner.

Adding a *construct* whose entire purpose is derivation is the point here. An
earlier draft of this spec added `system_one_source=` as an *argument* to
`Column`, which put provider concerns in a namespace shared with every ordinary
column option; §12 records the reversal.

### 3.4 `Parser(source=..., target=...)`

The primitive `ParsedField` compiles to. Two new optional arguments on the
existing class:

```python
class Parser(BaseParser):
    def __init__(
        self,
        parser_fn: Callable,
        *,
        source: str | list[str] | None = None,
        target: str | list[str] | None = None,
        on_error: Literal["raise", "null", "drop"] = "raise",
        # ... existing: element_wise, ignore_na, name, title, description
    ): ...
```

- `source=None, target=None` — today's behavior exactly. Nothing changes for
  existing code.
- `source` declared — pandera checks those columns exist **before** invoking the
  function, raising `ParserSourceError` naming the parser and the missing column.
- `target` declared — those columns are known to be produced: exempt from
  "missing column" errors before parsing, permitted under `strict=True`, and the
  output is checked to actually contain them (`ParserTargetError` if not).
- Both declared — the parser joins the dependency graph (§3.6).

**Compatibility note.** `Parser` currently forwards unrecognized keyword
arguments to the parser function — verified: `Parser(fn, source="body")` today
calls `fn(series, source="body")`. Promoting `source`/`target` to real
parameters is a small breaking change for anyone whose parser function takes a
keyword by those names. Worth a deprecation cycle, or `**parser_kwargs` could be
narrowed to an explicit `parser_kwargs={...}` dict, which is the cleaner
long-term shape regardless.

### 3.5 The `ColumnParser` protocol

`parser=` accepts a plain callable, or an object that knows how to build one.
This is the extension point layer 2 plugs into, and the only part of layer 1
that exists because of it:

```python
class ColumnParser(Protocol):
    def bind(self, ctx: ParseContext) -> Callable: ...
    def batch_key(self, ctx: ParseContext) -> Hashable | None: ...

@dataclass(frozen=True)
class ParseContext:
    target: str                        # column being produced
    dtype: DataType                    # its declared dtype
    description: str | None            # the field's description
    nullable: bool
    checks: list[Check]
    source: tuple[str, ...] | Callable
    schema: DataFrameSchema            # for cross-field resolution
```

`bind` is called once at schema-build time and returns the function the parser
will run. Two consequences that matter:

- **A parser sees its own column's declared type.** `system_one.Choice()` can
  read `Department` off the context and derive its options from it. No parser
  needs to be told what it is producing.
- **A parser can refuse early.** `bind` raising `SchemaInitError` surfaces at
  schema-build time, before any data or any network call.

`batch_key` is how per-column declarations avoid per-column cost. Parsers
returning equal non-`None` keys are handed to a `batch()` classmethod that
compiles them into a single `Parser` with the union of their targets. Callables
return `None` and are never batched. Layer 2's key is
`(provider, source, on_error)` — §5.1.

### 3.6 Ordering, errors, and serialization

**Ordering.** With `source`/`target` declared, parsers are topologically sorted
by column dependencies rather than run in list order. Undeclared parsers keep
their list position and run first, preserving today's behavior. A cycle is a
`SchemaInitError` naming the columns in it.

**Errors.** `ParserError` gains `ParserSourceError` (a declared source is absent
or mistyped) and `ParserTargetError` (the function ran but did not produce its
declared targets). `on_error="null"` and `"drop"` degrade per row instead of
failing the batch, composing with the existing `drop_invalid_rows`.

**Serialization.** `source` and `target` serialize per parser in `pandera.io`.
A callable body cannot be serialized — a YAML schema records the dependency
edges plus the parser's `name`/`description`, and deserializing a schema whose
parsers are unresolved raises on `validate` rather than silently skipping.
`ColumnParser` objects declare a `to_dict`/`from_dict` pair, so layer 2 schemas
round-trip **completely** (§4.8). That asymmetry is worth stating plainly: a
lambda is opaque, a declared question is not.

### 3.7 Imperative escape hatch

`@pa.parser(*fields)` already exists and attaches a transform to those fields.
Generalize it: with `source`, the decorated method *derives* the named fields.

```python
class Tickets(pa.DataFrameModel):
    body: str
    n_words: int

    @pa.parser("n_words", source="body")
    def count_words(cls, s):
        """Number of whitespace-separated tokens."""
        return s.str.split().str.len()
```

No `source` → today's meaning, unchanged. One decorator for one concept, rather
than a second decorator alongside it.

### 3.8 Backend scope

Parsers are pandas-only today, and non-pandas backends **silently ignore them**
(§6.4) — much worse for a derived column than for a transform, since the column
simply will not be there. Closing that silent path is a prerequisite, and is
independently worth doing.

---

## 4. Layer 2: System One parsers

### 4.1 Three parser objects

```python
import pandera.system_one as system_one

system_one.Choice(instructions=None, criteria=None, *, provider=None, **opts)
system_one.Score(instructions=None, criteria=None, *, provider=None, **opts)
system_one.Noul(instructions=None, *, threshold=0.5, provider=None, **opts)
```

Each implements `ColumnParser`. Everything is explicit if you want it to be:

```python
department: Department = pa.ParsedField(
    parser=system_one.Choice(
        instructions="Which team should handle this ticket",
        criteria={
            "billing": "Payment, invoices or subscription issues",
            "technical": "Bugs, outages or integration problems",
            "sales": "Pricing, plans or account expansion",
        },
    ),
)
```

and inferred if you don't. Naming the question type explicitly rather than
deriving it from the dtype is deliberate: it is one word, it reads as
documentation, and it removes the only genuinely ambiguous inference in the
design (§4.3).

### 4.2 What gets inferred

| Question part | Explicit | Inferred from |
|---|---|---|
| instructions | `instructions=` | the field's `description` |
| `Choice` options | `criteria` keys | the annotated type's members (`Enum`, `StrEnum`, `Literal`, `Category` categories) |
| `Choice` criteria | `criteria` values | enum **member docstrings**; falls back to member names |
| `Score` levels | `criteria` list order | the annotated `IntEnum`'s members in value order |
| `Score` rubric | `criteria` values | enum member docstrings |
| "or none" branch | — | `nullable=True` on the field |
| option subset | — | `isin=[...]` on the field |
| `Noul` → `bool` | `threshold=` | `True` iff `p >= threshold` |
| `Noul` → `float` | — | raw probability when the field is `float` with `in_range=(0, 1)` |

So this:

```python
class Frustration(enum.IntEnum):
    calm = 0
    """Calm, simply stating facts."""
    annoyed = 1
    """Frustrated but civil."""
    angry = 2
    """Very angry, strong language."""

frustration: Frustration = pa.ParsedField(
    description="How frustrated the customer appears",
    parser=system_one.Score(),
)
```

compiles to exactly this:

```python
Score(
    instructions="How frustrated the customer appears",
    criteria=["Calm, simply stating facts",
              "Frustrated but civil",
              "Very angry, strong language"],
)
```

Anything given explicitly wins; anything omitted is inferred; anything that can
be neither is a `SchemaInitError` naming the column and saying which part is
missing. Python discards member docstrings at runtime, so the inference needs
source introspection (§6.7) — worth doing in core, since those descriptions
belong on `Column.description` and in generated docs regardless of any provider.

Criteria quality is the single biggest lever on answer quality, so the docs
should push hard toward docstrings. Keeping them on the enum keeps them in the
schema, versioned, rather than in a prompt string somewhere else.

### 4.3 Type compatibility

Each parser declares which dtypes it can fill, checked at schema build:

| Parser | Valid target dtype | Rejected |
|---|---|---|
| `Choice()` | `Enum`, `StrEnum`, `Literal[...]`, `Category` (≤ 255 options) | `bool`, numerics, `str`, `datetime` |
| `Score()` | ordered `IntEnum` or ordered `Category` (2–10 levels); `float` for an unrounded position | unordered categoricals, `bool` |
| `Noul()` | `bool`; `float` with `in_range=(0, 1)` | everything else |

`Score()` on an unordered `Enum` raises `SchemaInitError` telling you to use
`Choice()` or make the type ordered. That message is the whole argument for
naming the question type: an earlier draft inferred `Choice` vs `Score` from
whether the category was ordered, which is correct but silent — a user who
forgot `ordered=True` would get a `Choice` and never know they had asked a
different question than they meant.

This also **demotes the ordered-category gap** (§6.5) from a blocker to a
nice-to-have: an `IntEnum` under `Score()` gets its ordering from member values,
so ordering is no longer load-bearing for question selection.

### 4.4 Source

`source` is a layer-1 concept — every derived column has one — so it lives on
`ParsedField`, with a schema-wide default for the common case where every
derived column reads the same material:

```python
class Config:
    parser_source = "ticket_body"          # schema-wide default
```

```python
reply_is_on_policy: bool = pa.ParsedField(
    description="The reply follows the stated refund policy",
    parser=system_one.Noul(),
    source=["ticket_body", "agent_reply"],  # per-field override
)
```

```python
def _ticket_state(row):                     # callable, for constants/truncation
    return {"message": row["ticket_body"][:4000],
            "policy": "Refunds within 30 days."}
```

Jev's accuracy degrades with irrelevant context, so a source is always named,
never "the whole row". A source naming an undeclared column is a
`SchemaInitError` at schema build, not a `KeyError` at parse time.

### 4.5 Confidence is just another derived column

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

    @pa.dataframe_check
    def routing_distribution_is_stable(cls, df):
        observed = df["department"].value_counts(normalize=True)
        return observed.reindex(BASELINE.index).sub(BASELINE).abs().max() < 0.15
```

`Confidence("department")` is a `ColumnParser` that reads the decision already
produced for another field rather than issuing its own request — a dependency
edge in the layer-1 graph, which is what makes it free.

This resolves a naming problem earlier drafts had. Confidence columns were
auto-injected with a `__confidence` suffix, which collided with regex column
matching and meant pandera added columns the schema did not declare. Here the
column is declared like any other, named whatever you want, and validated with
an ordinary `Field(ge=...)`.

Three levels of strictness compose, all from existing features:

1. `abstain_below` — per row: low-confidence answers become `NA`.
2. `ge=` on the confidence column — per row: hard failure, normal failure-case
   reporting.
3. `@pa.dataframe_check` — per batch: mean confidence, abstention rate,
   distribution drift.

### 4.6 Providers are runtime, not schema

A schema says what to ask. Who answers is configured out of band:

```python
import pandera.system_one as system_one

system_one.set_provider("typesafe:jev-1.13.0")        # process-wide

with system_one.provider(ReplayProvider(cassette)):   # scoped, for tests
    Triage.validate(df)
```

plus `PANDERA_SYSTEM_ONE_PROVIDER` as the env-var form, and
`system_one.Choice(provider=...)` as a per-parser override.

**There is no default provider.** Validating a System One schema with none
configured raises `SystemOneConfigError` telling you how to set one. That is the
safety property that matters here: `Triage.validate(df)` does issue paid
requests — inherent to putting the question in the schema — so it must be
impossible to reach that state without having deliberately configured a
provider.

Supporting mitigations:

```python
Triage.questions()
#> {'department': Choice(...), 'frustration': Score(...), 'is_urgent': Noul(...)}

Triage.parse_plan(tickets_df)
#> SystemOnePlan(rows=10_000, batches=1, requests=10_000,
#>               est_input_tokens=4_812_000, est_cost_usd=0.2021)
```

`questions()` compiles and inspects without a provider at all, so a schema's
questions are reviewable in a test with no credentials.

### 4.7 Semantic checks

Judging an existing column rather than producing a new one is a `Check`, not a
parser:

```python
class Products(pa.DataFrameModel):
    name: str
    category: Category
    description: str = pa.Field(
        checks=system_one.Holds(
            "The description is a coherent description of a product "
            "belonging to the stated category",
            context=["name", "category"],
            min_probability=0.85,
        )
    )
```

`Holds` is an ordinary `Check` with a vectorized predicate, so failure cases,
`lazy=True`, `n_failure_cases`, and `raise_warning` work untouched. A
`PANDERA_SYSTEM_ONE_ENABLED` guard lets schemas carrying semantic checks run
offline, degrading to a skip with a warning.

### 4.8 Serialization

Unlike a lambda, a question set is declarative, so layer 2 schemas round-trip
completely:

```yaml
columns:
  department:
    dtype: category
    description: Which team should handle this ticket
    nullable: true
    parser:
      type: system_one.choice
      source: [ticket_body]
      criteria:
        billing: Payment, invoices or subscription issues
        technical: Bugs, outages or integration problems
      abstain_below: 0.55
  department_confidence:
    dtype: float64
    parser:
      type: system_one.confidence
      of: department
    checks:
      greater_than_or_equal_to: 0.70
```

No provider appears, because the provider is not part of the schema (§4.6). A
YAML file is a complete, reviewable specification of what will be asked — and
because question text is part of the cache key (§5.3), rewording a `description`
correctly invalidates its cached answers.

---

## 5. Execution

### 5.1 Batching: per-column declarations, per-request cost

Jev's request shape is one state plus many questions, and a tenth question costs
tokens but almost no time. Per-column declarations would be pathological if each
became its own request, so the compiler groups them.

`batch_key` for all three parsers is `(provider, source, on_error)`. Fields with
equal keys are compiled into a single `Parser` whose target is their union:

```python
class Conversation(pa.DataFrameModel):
    customer_msg: str
    agent_reply: str

    intent: Intent = pa.ParsedField(parser=system_one.Choice(), source="customer_msg")   # ┐
    is_urgent: bool = pa.ParsedField(parser=system_one.Noul(), source="customer_msg")    # ┘ batch A

    reply_is_on_policy: bool = pa.ParsedField(                                           # ┐ batch B
        parser=system_one.Noul(), source=["customer_msg", "agent_reply"],                # ┘
    )
```

Two batches → two requests per row. A 12-column schema over one source is one
request per row, not twelve. `parse_plan()` reports the batch count, so the
grouping is inspectable rather than a thing you hope happened.

TypeSafe reports this speculative fan-out at 12.2× cheaper and 10.0× faster than
one call per field.

### 5.2 Across rows

Bounded async fan-out through an `asyncio.Semaphore(max_concurrency)` behind a
token-bucket limiter sized to the provider's published limits, with exponential
backoff honoring `retry-after`:

```python
async def _parse(states, compiled, provider, limits):
    sem = asyncio.Semaphore(limits.max_concurrency)
    bucket = TokenBucket(rpm=limits.rpm, tps=limits.tps)

    async def one(i, state):
        async with sem:
            await bucket.acquire(estimate_tokens(state, compiled))
            return i, await provider.decide(state, compiled)

    return await gather_ordered(one(i, s) for i, s in enumerate(states))
```

Ordering is restored by index. A parser must return a frame whose index aligns
with its input — non-negotiable, since answers are joined onto an existing frame.
Concurrency and cache settings are provider-level, configured alongside it
(§4.6), since they are runtime concerns rather than schema ones.

State size is checked against the 64k combined / 32k individual token limits
before dispatch, raising with the offending row index rather than surfacing a
provider HTTP error.

### 5.3 Caching

Dataframe workloads re-run constantly — a new day appended to last week's, a
notebook cell run six times, a backfill overlapping a prior run. Key:
`sha256(provider_id, resolved_model_version, canonical(state), canonical(questions))`.

Built-ins: in-memory dict, SQLite, parquet directory. Stats land on the
validated frame:

```python
out.attrs["pandera.system_one"]
#> {'rows': 10_000, 'batches': 1, 'cached': 9_412, 'called': 588,
#>  'model_version': 'jev-1.13.0', 'input_tokens': 241_305,
#>  'est_cost_usd': 0.0101, 'wall_seconds': 4.3}
```

`jev-latest` can change answers between runs, so the resolved version is part of
the key and an unpinned provider warns when a cache is configured.

### 5.4 Partitions

Once parsers work on more than pandas (§6.4), the same schema runs
per-partition: pandas locally, polars per chunk, dask one event loop per
partition, pyspark via `mapInPandas`. Backend-specific work is confined to
pulling state out of rows and putting typed arrays back into columns.

---

## 6. Gaps to close first

Verified empirically against `main` (`62f55e2d`) with pandas and polars
installed. Every one is a pandera bug or hole that stands on its own.

### 6.1 Parsers have no declared source, so failures are opaque 🔴

```python
pa.DataFrameSchema(
    {"n": pa.Column(int)},
    parsers=pa.Parser(lambda d: d.assign(n=d["nope"].str.len())),
).validate(pd.DataFrame({"body": ["a"]}))
#> KeyError: 'nope'
```

A bare `KeyError`: not a `SchemaError`, no schema context, no parser name, no
indication of which of several parsers failed. §3.4 fixes this by construction.

### 6.2 `Literal` is not a supported dtype 🔴

```python
class M(pa.DataFrameModel):
    x: typing.Literal["billing", "technical"]
M.to_schema()
#> SchemaInitError: Invalid annotation 'x: typing.Literal['billing', 'technical']'
```

Wrapped in `Series[...]`, `pandera/typing/common.py:368-371` replaces `self.arg`
with `get_args(self.arg)[0]` — the *first literal value* — and `self.literal` is
never read again anywhere in the codebase. That value is then handed to the
dtype engine as if it were a dtype:

```python
x: Series[Literal["billing", "technical"]]
#> TypeError: data type 'billing' not understood      # opaque; no mention of x

x: Series[Literal["int64", "float64"]]
#> Column(name=x, type=DataType(int64))               # silently an int column
```

The last case is the dangerous one: whenever a literal value happens to be a
valid dtype string, the option set is silently discarded and the column typed as
that dtype, with no membership constraint and no warning.

**Fix:** map `Literal[...]` to a `Category` over its args (or `pl.Enum` for
polars) in both annotation forms; raise `SchemaInitError` naming the field for a
heterogeneous `Literal`; never fall through to the dtype engine with a literal
value.

### 6.3 `enum.Enum` maps to *members*, not values 🔴

`pandera/engines/pandas_engine.py:237-240` converts an `Enum` class to
`Category(categories=data_type)`, and iterating an `Enum` yields **members**, so
the dtype is `CategoricalDtype([<Dept.billing: 'billing'>, ...])`.

```python
class Dept(enum.Enum):
    billing = "billing"
    technical = "technical"

class M(pa.DataFrameModel):
    d: Dept

M.validate(pd.DataFrame({"d": pd.Categorical(["billing", "technical"])}))
# SchemaError: expected series 'd' to have type category with categories
# (<Dept.billing: 'billing'>, <Dept.technical: 'technical'>),
# got category with categories ('billing', 'technical')

pa.DataFrameSchema({"d": pa.Column(Dept, coerce=True)}).validate(
    pd.DataFrame({"d": ["billing", "technical"]})
)  # SchemaErrors — coercion fails too
```

`IntEnum` and `StrEnum` work by accident: their members subclass `int`/`str`
with matching `__eq__`/`__hash__`. Plain `Enum` — the most common spelling —
does not. A blocker, since Jev returns option *values*.

**Fix:** use `.value` for categories; let `coerce` accept members, values, or
names.

### 6.4 Parsers are pandas-only, and polars ignores them silently 🔴

```python
schema = pa.DataFrameSchema({"a": pa.Column(pl.Int64, parsers=pa.Parser(fn))})
schema.validate(pl.DataFrame({"a": [1, 2, 3]}))   # succeeds; fn never called
```

`Column(parsers=...)` is accepted by the polars API and the polars
`ColumnBackend` has no `run_parsers` at all. For a *transform* this silently
skips a step; for a **derived column** it silently produces a frame with a
missing column, or a confusing downstream error.

**Fix, in order:** (a) raise `SchemaInitError` when a non-pandas schema is given
parsers, closing the silent path immediately; (b) implement `run_parsers` for
polars; (c) extend to remaining backends. Step (a) is small and independently
shippable.

### 6.5 Ordered categories are unreachable from an annotation 🟢

`Engine.dtype(SomeEnum)` always constructs `Category(..., ordered=False)` —
verified. Now only a nice-to-have, since naming `Score()` explicitly (§4.3)
removes ordering from question selection. Still worth fixing: an ordered
category supports `ge`/`le` and monotonicity checks that an unordered one does
not.

**Fix:** `IntEnum` → `ordered=True`; `Field(dtype_kwargs={"ordered": True})` as
the general escape hatch (already works, undocumented for this purpose).

### 6.6 pandas and polars disagree about enums 🟡

```python
polars_engine.Engine.dtype(Dept)   #> Enum(categories=['billing','technical'])   ✅ by value
pandas_engine.Engine.dtype(Dept)   #> category with categories (<Dept.billing>,) ❌ by member
```

polars gets it right. The same model class produces different semantics on the
two backends, breaking the promise that a `DataFrameModel` is backend-portable.
This survived because nothing tests both engines against the same enum.

**Fix:** converge on polars' behavior (§6.3); add a cross-backend enum
conformance test.

### 6.7 Enum member docstrings are not captured 🔴

Now a blocker rather than a nicety: §4.2's inference is the design's main
ergonomic claim, and it rests entirely on member docstrings. Without them every
`Choice()` falls back to bare member names as criteria, which is exactly the
low-quality-criteria case the docs warn against.

**Fix:** a `pandera.dtypes.member_descriptions(EnumT)` helper using
`inspect.getsource` + `ast`, cached per class, with a graceful fallback when
source is unavailable (REPL, frozen app). Belongs in core: these descriptions
should reach `Column.description` and generated docs regardless of any provider.

### 6.8 `Category` has no per-category description slot 🟡

Criteria for a non-enum categorical have nowhere to live but a `metadata` blob.
This bites when building schemas programmatically from a runtime taxonomy, where
there is no enum class to hang docstrings on. A first-class
`Category(categories=..., descriptions={...})` would fix it and improve
generated docs for ordinary categorical columns.

### 6.9 Summary

| # | Gap | Severity | Blocks | Standalone value |
|---|---|---|---|---|
| 6.1 | Parsers have no declared source/target | 🔴 | layer 1 | yes — error quality |
| 6.2 | `Literal` unsupported; can silently mistype a column | 🔴 | `Choice()` | yes — silent data loss |
| 6.3 | `Enum` → members not values | 🔴 | `Choice()` | yes — pre-existing bug |
| 6.4 | Parsers pandas-only, silently ignored | 🔴 | non-pandas backends | yes — silent failure |
| 6.7 | No member docstring capture | 🔴 | criteria inference | yes — docs |
| 6.6 | pandas/polars enum divergence | 🟡 | portability | yes |
| 6.8 | No per-category descriptions | 🟡 | programmatic schemas | yes — docs |
| 6.5 | No ordered categories from annotation | 🟢 | — | yes |

---

## 7. Provider protocol

```python
class DecisionProvider(Protocol):
    id: str
    model_version: str          # resolved and pinned; part of the cache key

    def compile(self, questions: dict[str, Question]) -> Any:
        """Validate/translate the question set. Raises SchemaInitError."""

    async def decide(self, state: Any, compiled: Any) -> dict[str, Decision]:
        """One state -> typed answers."""

    @property
    def limits(self) -> ProviderLimits: ...
```

`Decision` is a dataclass: `value`, `confidence`, `probabilities`, `raw`.
Shipped: `TypeSafeProvider` (wraps `typesafe_sdk.AsyncTypeSafeClient`),
`RecordingProvider`/`ReplayProvider` (cassettes), `MockProvider` (seeded
deterministic answers, for docs and CI). Three methods is what keeps this from
becoming a one-vendor dead end.

---

## 8. Packaging

```toml
[project.optional-dependencies]
typesafe-ai = ["typesafe-sdk"]
```

- Layer 1 is **core pandera** — no extra, no new dependency.
- Layer 2 is `pandera.system_one`, installed with
  `pip install 'pandera[typesafe-ai]'`; Jev-specific code lives in
  `pandera.system_one.providers.typesafe`.
- Importing `pandera` without the extra is byte-for-byte unaffected.

---

## 9. Phasing

| Phase | Layer | Scope | Exit criteria |
|---|---|---|---|
| **0** | core | §6.2, §6.3, §6.4(a), §6.7; cross-backend enum conformance tests | `Enum` and `Literal` round-trip on pandas and polars; member docstrings readable; non-pandas parsers raise instead of no-op |
| **1** | 1 | `Parser(source=, target=)`, dependency sort, `ParserSourceError`/`ParserTargetError`, `ParsedColumn`/`ParsedField`, `ColumnParser` protocol + batching, `Config.parser_source`, `@pa.parser(source=)` | Derived columns work end-to-end on pandas with declared provenance. **Ships with no AI code at all.** |
| **2** | 2 | `Question`/`Decision`, `DecisionProvider`, `TypeSafeProvider`, `Choice`/`Score`/`Noul` + inference, provider configuration, async fan-out + limiter | The §0 example runs end-to-end; §4.2 inference and §4.3 compatibility fully covered by replay tests |
| **3** | 2 | `Confidence`, `abstain_below`, cache, stats, `parse_plan()` | Confidence floors and distribution checks work; cache-hit path makes zero network calls |
| **4** | both | §6.4(b), §6.8; polars parsers, `Holds`, YAML round-trip, CLI | Same model validates on pandas and polars with identical output |
| **5** | both | dask/modin/pyspark partitioning, multi-label, `ParsedIndex` | 1M-row parse on dask with bounded memory and correct rate limiting |

Phases 0 and 1 are worth doing whether or not layer 2 is ever built.

---

## 10. Testing

Hard constraint: **no test requires an API key or network access.**

- **Layer 1 tests need no provider at all** — the biggest testability win from
  the split. Dependency sorting, cycle detection, source/target validation,
  batching, error types, and serialization are exercised with `lambda s: s * 2`.
- **Question compilation is testable without a provider.** `Triage.questions()`
  compiles §4.2's inference with no credentials, so the inference rules —
  the design's main ergonomic claim — are unit tests, not integration tests.
- **Batching tests.** Fields sharing `(provider, source, on_error)` produce one
  request per row; differing keys produce one batch each; `parse_plan()` reports
  the true count.
- **Cassettes.** `RecordingProvider` captures real responses once behind
  `PANDERA_RECORD_CASSETTES=1`; `ReplayProvider` serves them in CI via the
  `system_one.provider(...)` context manager.
- **Compatibility tests.** §4.3 as a parametrized matrix: every (parser, dtype)
  pair either compiles or raises `SchemaInitError` naming the column.
- **Type-system regression tests** for §6.2–§6.8, written before the fixes.
- **Concurrency tests** with a latency-injecting fake provider: ordering
  preserved, concurrency capped, token bucket throttles, `retry-after` honored,
  one row's failure does not poison the batch.
- **Pipeline-order tests** asserting derivation runs before column parsing,
  before coercion, before checks.
- **Nightly key-gated smoke test** against `jev-latest`, non-blocking.

---

## 11. Documentation plan

- `docs/source/parsers.md` — extend with derived columns as a first-class
  concept; update the "pandas only" note as §6.4 lands.
- `docs/source/derived_columns.md` — layer 1 user guide.
- `docs/source/system_one.md` — layer 2 user guide: the §0 example, the
  inference rules, writing good criteria, confidence, providers, caching, cost.
- `docs/source/dtypes.md` — document enum/`Literal`/ordered-category behavior
  once §6 is fixed. Currently undocumented, which is why the gaps went unnoticed.
- `docs/source/integrations.md` — add a **TypeSafe AI (Jev)** row.
- `docs/source/reference/system_one.rst` — API reference.
- A notebook doing end-to-end ticket triage, runnable with `MockProvider`.

---

## 12. Risks and open questions

**`validate()` can cost money.** Putting the question in the schema means
`Triage.validate(df)` issues requests. This is inherent to the design and worth
stating plainly. Mitigations: no default provider, so an unconfigured schema
raises rather than calls (§4.6); `questions()` and `parse_plan()` inspect without
calling; stats always attached to the result. Worth considering an opt-in
confirmation above a configurable row threshold.

**Vendor concentration.** Jev is one vendor's proprietary model in early access
(opened 2026-09-15 — this is very new). Mitigation: the `DecisionProvider`
protocol, no Jev import in core, and a layer 1 that is independently useful.

**Vendor-reported numbers.** Every figure in §1.3–§1.4 is TypeSafe's. Before any
of it reaches pandera's docs we publish our own benchmark on a public dataset
and cite ours; until then the docs describe cost and latency qualitatively.

**Reproducibility.** Answers change across model versions. Mitigations: resolved
version in the cache key and in `attrs`, warning on unpinned `jev-latest` when
caching.

**Open questions:**

1. Does `Config.parser_source` earn its place (§4.4)? It keeps the §0 example a
   one-liner per field, but it is a schema-wide default for something that is
   conceptually per column. The alternative is repeating `source=` on every
   field, which is explicit but noisy.
2. Should `system_one.Ask()` exist — a parser that picks `Choice`/`Score`/`Noul`
   from the dtype? It would shorten the common case, at the cost of
   reintroducing the silent inference §4.3 deliberately removed.
3. Enum columns holding values vs. members (§6.3) — values recommended; either
   choice is a small breaking change.
4. Should `Confidence` be able to reference a field in a *different* batch, or
   only a sibling? Cross-batch would need the layer-1 graph to carry decision
   objects, not just columns.
5. Should layer 1 allow a parser to *remove* columns, or is `strict="filter"`
   sufficient? Removal would complete the derivation story but complicates the
   dependency graph.
6. `ParsedIndex` for symmetry, or is deriving an index out of scope?

**Resolved during review:**

- *How is a System One column declared?* Through **`ParsedField`/`ParsedColumn`
  with a parser object** (§0), not a separate step. Earlier drafts proposed a
  standalone `SystemOneParser(output=Model, source=...)` mirroring Pydantic AI's
  `Agent`. That was a faithful port of someone else's API rather than a pandera
  design: it split column declarations across two places, and it made the
  question set a property of a side object rather than of the column it fills.
  One derivation mechanism, with question types as parsers, is the pandera
  shape — and batching (§5.1) recovers the per-request economics that made the
  bundled form attractive.
- *Where do AI arguments live on `Field`/`Column`/`Config`?* **Nowhere.** An
  earlier draft added `system_one_source=` and `system_one_provider=` to the
  core constructs. `ParsedField(parser=...)` gives derivation its own construct,
  and providers moved to runtime configuration entirely.
- *`Choice` vs `Score`* — named explicitly rather than inferred from category
  ordering (§4.3), which also demoted §6.5 from blocker to nice-to-have.
- *Confidence column naming* — declared as an ordinary `ParsedField` (§4.5),
  replacing auto-injected `__confidence` suffixes.

---

## 13. Summary

The question "which team should handle this ticket" belongs next to the column
that holds the answer, in the same place a `lambda` would go if the answer were
computable. That is the whole design:

```python
department: Department = pa.ParsedField(
    description="Which team should handle this ticket",
    parser=system_one.Choice(),
)
```

Everything else follows. The type is plain Python and supplies the options; the
docstrings supply the criteria; the description supplies the question; the
engine batches the columns back into one request; and the answers are validated
by the same `Check`s as any other column.

Getting there needs one primitive pandera is missing — a parser that declares
the columns it reads and produces — plus a handful of type-system holes closed.
Both are worth having on their own, and a user who never installs the extra
still gets derived columns out of it.
