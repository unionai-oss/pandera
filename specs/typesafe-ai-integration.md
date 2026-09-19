# TypeSafe AI (Jev) Integration Spec — Typed Parsing for DataFrames

> **Status:** Draft / RFC
> **Author:** pandera maintainers
> **Install:** `pip install 'pandera[typesafe-ai]'`
> **Prior art:**
> [Pydantic AI — TypeSafe (Jev) integration](https://pydantic.dev/docs/ai/models/typesafe/)

---

## 0. TL;DR

Pydantic AI points an `Agent` at Jev and gets back one validated object whose
fields were filled by a typed classifier rather than a text generator. Questions
live in `Field(description=...)`; the *annotation* determines the question type.

This spec proposes the same programming model for dataframes, with the Jev call
implemented as a **parsing step** inside pandera's existing
parse-then-validate pipeline:

```python
import enum
import pandera.pandas as pa
from pandera.typing import Series

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

class TicketTriage(pa.DataFrameModel):
    ticket_body: Series[str]
    department: Series[Department] = pa.Field(
        description="Which team should handle this ticket",
        ai_source="ticket_body",
    )
    frustration: Series[Frustration] = pa.Field(
        description="How frustrated the customer appears",
        ai_source="ticket_body",
    )
    is_urgent: Series[bool] = pa.Field(
        description="The message conveys urgency or time-sensitivity",
        ai_source="ticket_body",
    )

    class Config:
        ai_provider = "typesafe:jev-1.13.0"

triaged = TicketTriage.validate(tickets_df)
```

Each field declares **what it reads** (`ai_source`) and **what it asks**
(`description`) next to **what it produces** (the annotation). Fields that share
a source share a request, so the three questions above are one call per row, not
three (§6.1).

`tickets_df` has one column. `triaged` has four, typed and checked. There is no
separate `extract()` call: the AI layer is a dataframe-level parser, so it runs
at step 1 of the pipeline pandera already documents, and everything downstream —
coercion, column checks, dataframe checks, `lazy=True`, `drop_invalid_rows` —
works unmodified.

The spec has two halves. §1–§7 describe the integration. **§8 describes the
gaps in pandera's own type system that have to close first** — plain
`enum.Enum` does not currently round-trip through the pandas engine, `Literal`
silently degrades to `object`, ordered categories are unreachable from an
annotation, and parsers are pandas-only and *silently ignored* by polars. Those
are pre-existing bugs this integration turns into blockers.

---

## 1. Motivation

### 1.1 Why Jev is a good fit for pandera specifically

Jev is a **System One model**: it does not generate text. It answers typed
questions and can only return values from the schema it was given. Three
question types:

| Question | Returns | Shape |
|---|---|---|
| `Noul(instructions=...)` | calibrated probability | `float` in `[0, 1]` |
| `Choice(instructions=..., criteria={opt: desc})` | one option | ≤ 255 options |
| `Score(instructions=..., criteria=[level, ...])` | position on a scale | 2–10 ordered levels |

Vendor-reported characteristics relevant to a dataframe workload: 70–500 ms
latency, 1,200 req/min, 250k tokens/sec, $0.042/M input tokens with output
free, a 0% structured-output error rate, and *"a tenth question costs tokens but
almost no time"* — which makes a whole schema per row the natural request unit.

The 0% structured-output rate is the load-bearing fact for pandera. It **moves
validation up a level**. There is no question of whether the model returned
`"Billing "`, `"BILLING"`, or a two-paragraph apology; the dtype and the domain
are guaranteed by construction. What remains uncertain is whether the
*distribution* of decisions is sane:

- Is 60% of today's batch routing to one department when the baseline is 4%?
- Did mean confidence on `frustration` drop after the last model version bump?
- Do 30% of rows now fall below the abstention threshold?

Those are `Check`s and `Hypothesis`es over a column — pandera's home turf, and
with no per-object analogue in Pydantic AI.

### 1.2 Why "parsing", not "extraction"

Pandera already draws this distinction, in `docs/source/parsers.md`:

> Validation is the act of verifying whether data follows some set of
> constraints, whereas parsing transforms raw data into some desired set of
> constraints.

Filling `department` from `ticket_body` is a transform from raw data into a
constrained form. It is parsing, in exactly pandera's sense of the word — the
same bucket as `coerce=True`, `strict="filter"`, and `add_missing_columns=True`.
Modeling it as anything else (a bespoke `extract()` entry point, a new schema
subclass) would duplicate a pipeline stage that already exists and already has
the right ordering semantics.

### 1.3 Cost

10,000 tickets, ~400 tokens of state each, 6 fields: ~4M input tokens ≈ **$0.17**
in one pass of ~10,000 requests. Row-level fan-out over a whole dataframe is a
reasonable default rather than an extravagance, which is what makes the
"just call `validate()`" ergonomics defensible.

*(All figures in §1.1 and §1.3 are vendor-reported. §13 covers verification
before any of them appear in user-facing docs.)*

---

## 2. Design principles

1. **`Field` is the only new surface.** No `Ask()`, no `Question` objects in
   user code, no parallel spec. `Field` gains exactly one argument,
   `ai_source`; question text reuses the existing `description`; the question
   *type* is derived from the annotation, exactly as Pydantic AI derives it
   from the output type's fields.
2. **Parsing is the integration point.** The AI layer is a `Parser`. It slots
   into the documented pipeline rather than bypassing it.
3. **Validation stays ordinary.** After the parser runs, the frame is validated
   by the same schema through the same code path as any other frame.
4. **Minimal core surface.** Core pandera gains two plumbing arguments
   (`Field(ai_source=...)`, `ai_provider` on schemas — §5.1) and the
   type-system fixes in §8, which are bugs and existing holes rather than new
   concepts. Everything Jev-specific lands behind the `typesafe-ai` extra.
5. **Provider-pluggable.** `ai_provider` names a provider; Jev is the first
   implementation of a protocol, not a hardcoded dependency.
6. **Unsupported is an init-time error.** A field that cannot be expressed as a
   question raises `SchemaInitError` when the schema is built — matching
   Pydantic AI's contract of raising `UserError` before any request is sent.
7. **Deterministic tests.** No test in pandera's suite may require an API key or
   network access.

---

## 3. The programming model

### 3.1 Question placement

Straight from the Pydantic AI docs, and adopted verbatim:

> Jev separates the material being judged from the questions about it. The
> prompt contains the content and supporting facts; questions belong on the
> output type's fields via `Field(description=...)`.

In pandera terms, both halves live on the field:
**`Field(ai_source=...)` names the columns that carry the material;
`Field(description=...)` carries the question.** A field with an `ai_source`
and an AI-answerable annotation, in a schema with an `ai_provider`, is an
AI-parsed column. Every other field is an ordinary column and is left alone.

Keeping the source on the field rather than on `Config` buys three things:

1. **A field is self-describing.** What it reads, what it asks, and what it
   produces are one declaration. Reviewing a diff to one field does not require
   scrolling to `Config` to learn what it was reading.
2. **Different fields can read different material.** A schema can triage
   `ticket_body` and separately score `agent_reply` in the same pass — which is
   the normal case for any frame with more than one text column, and is not
   expressible with a single schema-wide source.
3. **The marker is explicit.** `ai_source` is what makes a field AI-parsed, not
   the presence of a `description` — so `description` keeps its ordinary
   documentation meaning everywhere else. This resolves the largest design risk
   in earlier drafts of this spec (§13).

`description` is not a new argument: it already exists on `Field`, already flows
to `Column.description`, and already round-trips through `pandera.io` YAML. An
AI-parsed schema is reviewable in a PR and diffable when a question is reworded.

### 3.2 Type mapping

The annotation determines the question type. This table is the pandera
re-expression of Pydantic AI's supported-field-types table.

| Annotation | Question | Answer → column | Notes |
|---|---|---|---|
| `Series[bool]` | `Noul` | `True` iff `p >= ai_boolean_threshold` (default `0.5`) | |
| `Series[float]` + `Field(in_range=(0, 1))` | `Noul` | raw probability, unrounded | |
| `Series[StrEnum]` / `Series[Enum]` | `Choice` | chosen option | members are the options; ≤ 255 |
| `Series[Literal["a", "b"]]` | `Choice` | chosen option | **needs §8.2** |
| `Series[EnumT]` + `Field(nullable=True)` | `Choice` or none | option or `NA` | abstention |
| `Series[IntEnum]` *(ordered)* | `Score` | nearest level | member docstrings are the rubric; 2–10 levels; **needs §8.3** |
| `Series[float]` + ordered `IntEnum` in `Field(dtype_kwargs)` | `Score` | unrounded position (e.g. `1.035`) | |
| `Series[list[EnumT]]` | `Noul` per option | list of selected options | multi-label |
| Nested `DataFrameModel` | its fields, flattened | `outer_inner` columns | |
| `Series[str]`, `Series[datetime]`, unbounded numerics, `Series[dict]` | ✗ | — | `SchemaInitError` at schema build |

Disambiguating `Choice` from `Score` is the one place the annotation is not
quite enough on its own: both are "pick one of an ordered-or-unordered set". The
rule is **ordering**:

- an `Enum`/`StrEnum`, or an unordered category → `Choice`
- an `IntEnum`, or a category with `ordered=True` → `Score`

which is semantically right (a `Score` answer can land *between* levels at
`1.035`, which only means something for an ordered domain) and is why §8.3
matters.

### 3.3 Member docstrings are the criteria

`Choice` takes `criteria={option: description}` and `Score` takes an ordered
list of level descriptions. Those come from enum member docstrings, mirroring
Pydantic AI's `UseEnumMemberDocstrings`:

```python
class Frustration(enum.IntEnum):
    calm = 0
    """Calm, simply stating facts."""
    annoyed = 1
    """Frustrated but civil."""
    angry = 2
    """Very angry, strong language."""
```

compiles to:

```python
Score(
    instructions="How frustrated the customer appears",   # Field(description=...)
    criteria=["Calm, simply stating facts",
              "Frustrated but civil",
              "Very angry, strong language"],             # member docstrings
)
```

Python does not retain member docstrings at runtime, so this requires source
inspection (§8.5). Pydantic AI solves the same problem with a mixin; pandera
should solve it centrally so that `Column.description` and the generated docs
get the descriptions too, not just the AI layer.

Without docstrings, the member names are used as bare criteria. That works and
is a reasonable default, but the docs should push hard toward docstrings:
criteria quality is the single biggest lever on answer quality, and it is the
part that belongs in the schema rather than in a prompt somewhere else.

### 3.4 Field arguments that shape the question

Existing `Field` arguments are reused rather than shadowed:

| `Field` argument | Effect on the question |
|---|---|
| `ai_source` | the column(s) whose values become the state (**required** to mark a field as AI-parsed) |
| `description` | the question text (required alongside `ai_source`) |
| `nullable=True` | adds the "or none" branch; abstention → `NA` |
| `isin=[...]` | narrows a `Choice` to a subset of the annotation's options |
| `in_range=(0, 1)` | marks a `float` as a raw `Noul` probability |
| `title` | ignored by the provider; still used in error reports and docs |
| `metadata={"typesafe": {...}}` | escape hatch for provider-specific knobs |

Everything else about `Field` keeps working normally, and applies to the
*result*: `Field(description=..., isin=[...])` both narrows the question **and**
validates the answer, which is belt-and-braces but costs nothing.

---

## 4. Jev as a parsing step

### 4.1 Where it runs

`docs/source/parsers.md` documents the pipeline:

> 1. dataframe-level parsing
> 2. column-level parsing
> 3. dataframe-level checks
> 4. column-level and index-level checks

AI parsing is **step 1**. One dataframe-level parser is synthesized per schema,
holding the compiled question set for every AI-parsed column grouped by
`ai_source`. It receives the frame, builds one state per row per distinct
source group, dispatches the batch, and returns the frame with the answer
columns added.

Two properties of the existing implementation make this fit better than it has
any right to:

1. **Dataframe-level parsers run first and get the whole frame**, so they can
   add columns before column resolution happens.
2. **Coercion is already deferred when parsers are present.** In
   `pandera/backends/pandas/container.py:806-823`, `_coerce_dtype_helper` skips
   columns whose schema has parsers so that "parsers run before coercion". The
   answer columns therefore get coerced to their declared dtype *after* the
   parser materializes them, which is exactly the required ordering.

So the sequence for the §0 example is:

```
validate(tickets_df)
  ├─ 1. dataframe-level parsing
  │     └─ AIParser: 1 request/row, all 3 questions per request
  │        → adds department, frustration, is_urgent
  ├─ 2. column-level parsing        (user parsers, unchanged)
  ├─    coercion                    (answers → Category / IntEnum / bool)
  ├─ 3. dataframe-level checks      ("<10% route to sales")
  └─ 4. column-level checks         (Field(isin=...), confidence floors)
```

### 4.2 `AIParser`

```python
class AIParser(pandera.api.parsers.Parser):
    """Dataframe-level parser that fills columns from unstructured source data."""

    def __init__(
        self,
        # compiled from fields, grouped by resolved ai_source; not user-facing
        question_groups: dict[SourceKey, dict[str, Question]],
        provider: DecisionProvider,
        *,
        boolean_threshold: float = 0.5,
        abstain_below: float | None = None,
        max_concurrency: int = 16,
        cache: ExtractionCache | None = None,
        on_error: Literal["raise", "null", "drop"] = "raise",
    ): ...
```

It is a real `Parser` subclass — it registers a backend per dataframe library
through the existing `Parser.register_backend` mechanism, it appears in
`schema.parsers`, and `schema.validate` needs no special-casing to run it.
Users do not normally construct one; the schema builds it from the fields. But
it is constructible directly, which is the object-API escape hatch and the thing
that makes the whole design testable without a model class.

### 4.3 Column-level parsers: judging existing columns

The mirror case — a column that already exists, and a question about it — is a
**column-level parser**, step 2, no new machinery at all:

```python
class Reviews(pa.DataFrameModel):
    review_text: Series[str]
    sentiment: Series[float] = pa.Field(
        description="The review expresses a positive opinion of the product",
        ai_source="review_text",
        in_range=(0, 1),
        ge=0.0,
    )

    class Config:
        ai_provider = "typesafe:jev-latest"
```

Here `ai_source` points at a column the schema already declares, which is the
case field-level sourcing makes natural and a schema-wide source makes
ambiguous.

and for pure validation (a question whose answer is a pass/fail rather than a
column), the same compiler emits a `Check`:

```python
description: Series[str] = pa.Field(
    checks=pa.Check.ai(
        "The description is a coherent description of a product "
        "belonging to the stated category",
        context=["name", "category"],
        min_probability=0.85,
    )
)
```

`Check.ai` is an ordinary `Check` with a vectorized predicate, so failure cases,
`lazy=True`, `n_failure_cases`, and `raise_warning` all work untouched.

### 4.4 Errors

A failed request is a parsing failure, and pandera already has a category for
that: `ParserError`. `AIParser` raises `pandera.errors.ParserError` subclasses:

- `AIProviderError` — transport, auth, rate-limit exhaustion.
- `AIQuestionError` — a question the provider rejected (should be impossible;
  §2 principle 6 catches these at schema-build time).
- `AIStateError` — state exceeded the 64k combined / 32k individual token
  limits, raised with the offending row index *before* dispatch rather than
  surfacing as a provider HTTP error.

`on_error="null"` and `on_error="drop"` degrade per-row instead of failing the
batch, which composes with `drop_invalid_rows` for a manual-review queue.

### 4.5 What this design deliberately does not do

**No implicit network calls on an ordinary schema.** Two independent things must
both be true for a field to issue a request: the field sets `ai_source`, and the
schema sets `ai_provider`. Neither alone does anything — a `description` is
still just documentation, and an `ai_provider` on a schema whose fields declare
no source is inert. This is the single most important safety property in the
design, since `validate()` looks free and is not, and it is the main reason
`ai_source` belongs on the field: the opt-in is visible on the line that
triggers it, rather than inferred from a documentation string.

**No hidden cost.** `schema.ai_questions()` returns the compiled question set
without making a call; `schema.validate(df, ai_dry_run=True)` reports projected
request count, token count, and cost, then stops.

---

## 5. Configuration: `ai_source` and `ai_provider`

### 5.1 Where each setting lives

One new argument on the field, one on the schema, plus an existing field
argument that takes on a second job:

| Setting | Lives on | Why |
|---|---|---|
| `ai_source` | **`Field` / `Column`** | per-field: what this column reads (§3.1) |
| `description` | `Field` / `Column` (existing) | per-field: what this column asks |
| `ai_provider` | **`Config` / `DataFrameSchema`** | one provider per schema; batching depends on it |

Settings that rarely vary between fields — thresholds, concurrency, caching —
stay schema-wide. `Config.ai_source` also exists as an **optional default** for
the common case where every AI-parsed field reads the same column, overridden by
any field that sets its own. That default-plus-override shape is the same one
`on_missing_columns` / `Field(on_missing=...)` already uses, down to the
docstring wording ("This overrides the schema-wide `on_missing_columns` config
option").

**Implementation note.** `ai_source` is a genuine core change, small but not
free: `Field(**kwargs)` raises `SchemaInitError` for any key it does not
recognize (`model_components.py:250-256`), and `BaseFieldInfo` uses `__slots__`,
so the attribute must be declared. `on_missing` is the template for this — it
threads `Field` → `FieldInfo.__slots__` → `column_properties`, where it is
forwarded through an `extra` dict *only when explicitly set* "so that backends
whose `Column` does not (yet) support it are unaffected". `ai_source` should
follow that pattern exactly, which keeps non-pandas backends untouched until
§8.6 lands.

### 5.2 Settings reference

**Model API:**

```python
class TicketTriage(pa.DataFrameModel):
    ...
    class Config:
        ai_provider = "typesafe:jev-1.13.0"   # or a DecisionProvider instance
        ai_source = "ticket_body"             # optional per-schema default
        ai_boolean_threshold = 0.5
        ai_abstain_below = None
        ai_max_concurrency = 16
        ai_cache = None
        ai_confidence_suffix = "__confidence"
        ai_on_error = "raise"
```

**Schema API:**

```python
schema = pa.DataFrameSchema(
    columns={
        "ticket_body": pa.Column(str),
        "department": pa.Column(
            Department,
            description="Which team should handle this ticket",
            ai_source="ticket_body",
        ),
        "is_urgent": pa.Column(
            bool,
            description="The message conveys urgency or time-sensitivity",
            ai_source="ticket_body",
        ),
    },
    ai_provider="typesafe:jev-1.13.0",
)

triaged = schema.validate(tickets_df)
```

`ai_provider` accepts:

| Value | Meaning |
|---|---|
| `"typesafe:jev-1.13.0"` | pinned version, recommended |
| `"typesafe:jev-latest"` | unpinned; warns when `ai_cache` is set |
| `TypeSafeProvider(...)` | an instance, for custom clients/retry policy |
| `None` (default) | no AI parsing; schema behaves exactly as today |

A global default is settable through the existing `PanderaConfig` /
`PANDERA_AI_PROVIDER` env var, so the same schema definition runs against a
recorded cassette in CI and the real provider in production without edits.

---

## 6. Execution

### 6.1 Two axes of parallelism

**Within a row — one request per source group.** Jev's request shape is one
state plus many questions, so every field reading the *same* state can share a
call. The parser groups fields by resolved `ai_source` and issues one
`system_one` call per group per row. TypeSafe reports this speculative fan-out
at 12.2× cheaper and 10.0× faster than one call per field with identical
answers: a 12-field schema over one source column is one request, not twelve.

Per-field sourcing therefore costs **one request per distinct source per row**,
not one per field — the schema-wide-source economics are preserved exactly
whenever fields agree on a source, which is the common case:

```python
class Conversation(pa.DataFrameModel):
    customer_msg: Series[str]
    agent_reply: Series[str]

    intent: Series[Intent] = pa.Field(..., ai_source="customer_msg")   # ┐ group A
    is_urgent: Series[bool] = pa.Field(..., ai_source="customer_msg")  # ┘ 1 request

    reply_is_on_policy: Series[bool] = pa.Field(                       # ┐ group B
        ..., ai_source=["customer_msg", "agent_reply"],                # ┘ 1 request
    )
```

Two groups → two requests per row. The grouping key is the canonicalized source
spec: an ordered tuple of column names for declarative sources, or the callable's
identity for callable sources (so sharing one function across fields groups
them, while two equivalent lambdas do not). `schema.ai_questions()` returns the
groups, making the request count visible before anything is dispatched.

**Across rows and groups — bounded async fan-out.** Every (row, source group)
pair is a unit of work, dispatched concurrently through an
`asyncio.Semaphore(ai_max_concurrency)` behind a token-bucket limiter sized to
the provider's published limits, with exponential backoff honoring `retry-after`.
Groups are flattened into the same queue rather than run in sequence, so a
schema with two source groups saturates the same concurrency budget as one with
a single group.

```python
async def _parse(frame, groups, provider, limits):
    sem = asyncio.Semaphore(limits.max_concurrency)
    bucket = TokenBucket(rpm=limits.rpm, tps=limits.tps)

    async def one(idx, group, state):
        async with sem:
            await bucket.acquire(estimate_tokens(state, group.compiled))
            return idx, group, await provider.decide(state, group.compiled)

    work = ((idx, g, g.build_state(row))
            for g in groups
            for idx, row in frame.iterrows())
    return await gather_ordered(one(*w) for w in work)
```

Results are scattered back by `(index, group)`. The parser must return a frame
whose index aligns with its input — non-negotiable, since the answers are joined
onto an existing frame.

### 6.2 Partitions

Once `Parser` works on more than pandas (§8.6), the same parser runs
per-partition:

```python
TicketTriage.validate(pandas_df)     # local asyncio fan-out
TicketTriage.validate(polars_df)     # per-chunk
TicketTriage.validate(dask_df)       # one event loop per partition
TicketTriage.validate(spark_df)      # mapInPandas
```

Backend-specific work is confined to pulling state out of rows, putting typed
arrays back into columns, and choosing a partition strategy. Planner, provider,
cache, and limiter are shared.

### 6.3 Caching

Dataframe workloads re-run constantly — a new day appended to last week's, a
notebook cell run six times, a backfill overlapping a prior run. Without a
cache every re-run pays full freight; with one, incremental parsing is the
normal case.

Key: `sha256(provider_id, resolved_model_version, canonical(state), canonical(questions))`.

Because the question text is part of the key, **rewording a `description`
correctly invalidates its cached answers** — which is the payoff for putting
questions in the schema rather than in a prompt string somewhere.

```python
class Config:
    ai_cache = "sqlite:///.pandera_ai_cache.db"
```

Built-ins: in-memory dict, SQLite, parquet directory. `ExtractionCache` is a
`get`/`set`/`stats` protocol, so Redis or a warehouse table is user-supplied.
Stats land on the validated frame:

```python
out.attrs["pandera.ai"]
#> {'rows': 10_000, 'cached': 9_412, 'called': 588, 'model_version': 'jev-1.13.0',
#>  'input_tokens': 241_305, 'est_cost_usd': 0.0101, 'wall_seconds': 4.3}
```

### 6.4 State construction

Jev's accuracy degrades with irrelevant context, so state is explicit, never
"the whole row". `ai_source` accepts three spellings:

```python
# one column: the value is the state
department: Series[Department] = pa.Field(
    description="Which team should handle this ticket",
    ai_source="ticket_body",
)

# several columns: a dict of {column: value}
reply_is_on_policy: Series[bool] = pa.Field(
    description="The reply follows the stated refund policy",
    ai_source=["customer_msg", "agent_reply", "policy_text"],
)

# a callable: full control, including constants and truncation
def _ticket_state(row):
    return {"message": row["ticket_body"][:4000],
            "policy": "Refunds within 30 days."}

is_refundable: Series[bool] = pa.Field(
    description="This ticket describes a refundable purchase",
    ai_source=_ticket_state,
)
```

Referencing a column the schema does not declare is a `SchemaInitError` at
schema build, not a `KeyError` at parse time. Callable sources are validated
lazily against the first row and their result must be JSON-serializable.

---

## 7. Confidence

Jev returns calibrated confidence alongside every answer. In the per-object case
that is a number you might log. In the dataframe case it is a **column**, and
therefore something pandera can validate.

```python
class TicketTriage(pa.DataFrameModel):
    ticket_body: Series[str]
    department: Series[Department] = pa.Field(
        description="Which team should handle this ticket",
        ai_source="ticket_body",
        nullable=True,
        metadata={"typesafe": {"confidence": True}},
    )
    department__confidence: Series[float] = pa.Field(ge=0.70)

    class Config:
        ai_provider = "typesafe:jev-1.13.0"
        ai_abstain_below = 0.55

    @pa.dataframe_check
    def routing_distribution_is_stable(cls, df):
        observed = df["department"].value_counts(normalize=True)
        return observed.reindex(BASELINE.index).sub(BASELINE).abs().max() < 0.15
```

Three levels of strictness compose, all from existing features:

1. `ai_abstain_below` — per row: low-confidence answers become `NA`.
2. `Field(ge=...)` on the confidence column — per row: hard failure, normal
   failure-case reporting.
3. `@pa.dataframe_check` — per batch: "mean confidence > 0.8", "≤5% abstentions",
   "routing distribution hasn't drifted".

Full `probabilities` and `scores` distributions are attached to the frame's
`attrs` rather than materialized as columns (they can be 255-wide), retrievable
via `pandera.ai.probabilities(df, "department")`.

Declaring the confidence column explicitly — rather than auto-injecting it — is
deliberate: a schema should describe every column in the frame it validates, and
`strict=True` would otherwise reject a column pandera itself added.

---

## 8. Gaps in pandera's type system

This section is the actual prerequisite work. The type mapping in §3.2 leans on
`Enum`, `IntEnum`, `StrEnum`, `Literal`, and ordered categories — and several of
those do not currently behave the way the mapping needs. Findings below were
verified empirically against `main` (`62f55e2d`) with pandas and polars
installed.

### 8.1 `enum.Enum` maps to *members*, not values 🔴

`pandera/engines/pandas_engine.py:237-240` converts an `Enum` class to
`Category(categories=data_type)`. `dtypes.Category.__init__` does
`tuple(categories)`, and iterating an `Enum` class yields **members**, so the
resulting dtype is `CategoricalDtype([<Dept.billing: 'billing'>, ...])`.

```python
class Dept(enum.Enum):
    billing = "billing"
    technical = "technical"

class M(pa.DataFrameModel):
    d: Series[Dept]

M.validate(pd.DataFrame({"d": pd.Categorical(["billing", "technical"])}))
# SchemaError: expected series 'd' to have type category with categories
# (<Dept.billing: 'billing'>, <Dept.technical: 'technical'>),
# got category with categories ('billing', 'technical')

pa.DataFrameSchema({"d": pa.Column(Dept, coerce=True)}).validate(
    pd.DataFrame({"d": ["billing", "technical"]})
)  # SchemaErrors — coercion fails too
```

`IntEnum` and `StrEnum` work by accident: their members subclass `int`/`str`
with matching `__eq__`/`__hash__`, so member-keyed categories compare equal to
value-keyed ones. Plain `Enum` — the most common spelling, and the one
Pydantic AI users will reach for — does not.

This is a blocker: Jev returns option *values*. It is also a pre-existing bug
worth fixing independently of this integration.

**Fix:** use `.value` for categories, keep `coerce` able to accept members,
values, or names. Decide explicitly whether the validated column holds values
(recommended — it is what serializes, what every backend can store, and what
the provider returns) or members.

### 8.2 `Literal` silently degrades to `object` 🔴

`pandera/typing/common.py:368-371` sets `self.literal` when the annotation arg
is a `Literal`, then takes `get_args(self.arg)[0]` — the *first* literal value —
and `self.literal` is never read anywhere else in the codebase.

```python
class M(pa.DataFrameModel):
    x: Series[typing.Literal["a", "b"]]

M.to_schema()
#> Column(name=x, type=DataType(object))     # option set silently discarded
```

No error, no membership constraint. `Literal` is one of Pydantic AI's two
primary `Choice` spellings, so this has to work.

**Fix:** map `Literal[...]` to a `Category` over its args (or `pl.Enum` for
polars), and make a heterogeneous `Literal` a `SchemaInitError` rather than a
silent `object`.

### 8.3 Ordered categories are unreachable from an annotation 🟡

`dtypes.Category` has an `ordered` flag, but `Engine.dtype(SomeEnum)` always
constructs `Category(categories=..., ordered=False)` — verified. There is no
annotation-level spelling for an ordered category.

Ordering is what distinguishes `Score` from `Choice` (§3.2), and it is
independently meaningful: an ordered category supports `ge`/`le` checks and
monotonicity assertions that an unordered one does not.

**Fix:** `IntEnum` → `ordered=True` by default; a general escape hatch via
`Field(dtype_kwargs={"ordered": True})`, which already works but is
undocumented for this purpose.

### 8.4 pandas and polars disagree about enums 🟡

```python
polars_engine.Engine.dtype(Dept)      #> Enum(categories=['billing','technical'])   ✅ by value
pandas_engine.Engine.dtype(Dept)      #> category with categories (<Dept.billing>,) ❌ by member
```

polars gets it right — it maps a Python `Enum` class to `pl.Enum` using values.
The same model class therefore produces different semantics on the two backends,
which breaks the promise that a `DataFrameModel` is backend-portable.

**Fix:** converge on the polars behavior (§8.1). Add a cross-backend conformance
test for enum handling — this asymmetry survived because nothing tests both
engines against the same enum.

### 8.5 Enum member docstrings are not captured 🟡

`Choice` criteria and `Score` rubrics come from member docstrings (§3.3), but
Python discards them at runtime. Pydantic AI solves this with a
`UseEnumMemberDocstrings` mixin that inspects source.

**Fix:** a `pandera.dtypes.member_descriptions(EnumT)` helper using
`inspect.getsource` + `ast`, with results cached per class and a graceful
fallback when source is unavailable (REPL, frozen app). Worth doing in core
rather than in the extra: these descriptions belong on `Column.description` and
in generated schema documentation regardless of whether an AI provider is
configured.

### 8.6 Parsers are pandas-only — and polars ignores them silently 🔴

`docs/source/parsers.md` states parsers are pandas-only. The failure mode is
worse than documented:

```python
schema = pa.DataFrameSchema({"a": pa.Column(pl.Int64, parsers=pa.Parser(fn))})
schema.validate(pl.DataFrame({"a": [1, 2, 3]}))   # succeeds; fn never called
```

`Column(parsers=...)` is accepted by the polars API and the polars
`ColumnBackend` has no `run_parsers` at all — the parser is silently dropped. A
user who writes an AI-parsed polars schema would get a passing validation on a
frame with no answer columns, or a confusing missing-column error.

**Fix, in order:** (a) immediately raise `SchemaInitError` when a non-pandas
schema is given parsers, so the silent path closes regardless of this
integration's timeline; (b) implement `run_parsers` for polars; (c) extend to
the remaining backends. Step (a) is a small, independently shippable bug fix.

### 8.7 `Category` has no per-category description slot 🟢

Choice criteria are per-option descriptions. Today they can only live in enum
member docstrings (§8.5) or a `metadata` blob. A first-class
`Category(categories=..., descriptions={...})` would let non-enum categorical
columns carry criteria and would improve generated docs generally. Nice to have,
not a blocker — `metadata` is sufficient for v1.

### 8.8 Summary

| # | Gap | Severity | Blocks | Independently valuable |
|---|---|---|---|---|
| 8.1 | `Enum` → members not values | 🔴 blocker | `Choice` | yes — pre-existing bug |
| 8.2 | `Literal` → silent `object` | 🔴 blocker | `Choice` | yes — silent data-quality hole |
| 8.3 | No ordered categories from annotation | 🟡 | `Score` | yes |
| 8.4 | pandas/polars enum divergence | 🟡 | portability | yes |
| 8.5 | No member docstring capture | 🟡 | criteria quality | yes — docs |
| 8.6 | Parsers pandas-only, silently ignored | 🔴 blocker | non-pandas backends | yes — silent failure |
| 8.7 | No per-category descriptions | 🟢 | — | yes |

Every one of these is a pandera-core improvement that stands on its own. That is
a feature of this design, not a coincidence: routing the integration through
`Field`, `Parser`, and the dtype engine means it exercises existing abstractions
hard enough to find where they are thin, instead of routing around them.

---

## 9. Provider protocol

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
Shipped implementations:

- `TypeSafeProvider` — wraps `typesafe_sdk.AsyncTypeSafeClient`.
- `RecordingProvider` / `ReplayProvider` — cassettes for tests.
- `MockProvider` — seeded deterministic answers, for docs and CI.

Keeping this protocol to three methods is what prevents a one-vendor dead end.

---

## 10. Packaging

```toml
[project.optional-dependencies]
typesafe-ai = [
    "typesafe-sdk",
]
```

```bash
pip install 'pandera[typesafe-ai]'
```

- Module namespace: `pandera.ai` (provider-neutral), with
  `pandera.ai.providers.typesafe` holding everything Jev-specific.
- Importing `pandera` without the extra must be byte-for-byte unaffected. The
  `ai_*` config keys exist but reject a non-`None` `ai_provider` with an
  actionable `ImportError` naming the extra.
- `typesafe-ai` is added to the `all` extra and to a dedicated nox session.

---

## 11. Phasing

| Phase | Scope | Exit criteria |
|---|---|---|
| **0 — Type system** | §8.1, §8.2, §8.6(a); cross-backend enum conformance tests | Plain `Enum` and `Literal` round-trip on pandas and polars; non-pandas parsers raise instead of no-op. **Ships independently of any AI work.** |
| **1 — Core** | `Field(ai_source=...)` / `Column(ai_source=...)` plumbing (§5.1) and `ai_provider` on both APIs; `Question`/`Decision`, `DecisionProvider`, `TypeSafeProvider`, question compiler + `SchemaInitError` coverage, source grouping, `AIParser`, pandas backend, async fan-out + limiter | The §0 example runs end-to-end on pandas; full §3.2 table covered by replay tests |
| **2 — Ergonomics** | §8.3, §8.5; confidence columns, `ai_abstain_below`, cache, stats, `ai_dry_run` | Confidence floors and distribution checks work; cache-hit path makes zero network calls |
| **3 — Breadth** | §8.6(b); polars backend, `Check.ai`, YAML round-trip, CLI | Identical model class validates on pandas and polars with identical output |
| **4 — Scale** | dask/modin/pyspark partitioning, multi-label, nested models | 1M-row parse on dask with bounded memory and correct rate limiting |

Phase 0 is worth doing whether or not the rest of this spec is accepted.

---

## 12. Testing

Hard constraint: **no test requires an API key or network access.**

- **Cassettes.** `RecordingProvider` captures real responses once behind
  `PANDERA_RECORD_CASSETTES=1`; `ReplayProvider` serves them in CI, keyed the
  same way as the cache so re-recording is incremental.
- **Compiler contract tests.** §3.2 becomes a parametrized test: every supported
  annotation compiles to the expected question type and criteria; every
  unsupported one raises `SchemaInitError` naming the field.
- **Source-grouping tests.** Fields sharing an `ai_source` produce exactly one
  request per row; distinct sources produce one each; `Config.ai_source` is
  used only where a field does not override it; an `ai_source` naming an
  undeclared column raises `SchemaInitError`; a field with `ai_source` but no
  `ai_provider` (and vice versa) makes no request at all.
- **Type-system regression tests** for every item in §8, written *before* the
  fixes, including the cross-backend enum conformance matrix from §8.4.
- **Concurrency tests** with a fake latency-injecting provider: ordering
  preserved, concurrency capped, token bucket throttles, `retry-after` honored,
  one row's failure does not poison the batch.
- **Pipeline-order tests** asserting AI parsing runs before column parsing,
  before coercion, before checks — the property §4.1 depends on.
- **Nightly key-gated smoke test** against `jev-latest`, non-blocking, to detect
  upstream drift.

---

## 13. Risks and open questions

**Vendor concentration.** Jev is one vendor's proprietary model in early access
(opened 2026-09-15 — this is very new). Mitigation: the `DecisionProvider`
protocol, and no Jev import anywhere in pandera core.

**Vendor-reported numbers.** Every figure in §1 is TypeSafe's. Before any of it
reaches pandera's docs we publish our own benchmark on a public dataset and cite
ours; until then the docs describe cost and latency qualitatively.

**`validate()` now costs money.** This is the sharpest edge in the design. A
method that has always been free can now issue thousands of paid requests.
Mitigations in §4.5: AI parsing is inert unless `ai_provider` is set,
`ai_questions()` inspects without calling, `ai_dry_run=True` reports projected
cost, and stats are always attached to the result. Worth considering an
additional opt-in confirmation above a configurable row threshold.

**Reproducibility.** Answers change across model versions. Mitigations:
resolved version in the cache key and in `attrs`, warning on unpinned
`jev-latest` when caching.

**Open questions:**

1. Should the validated enum column hold values or members (§8.1)? Values are
   recommended; members are what the current pandas behavior implies, so this is
   a (small) breaking change either way.
2. `Choice` vs `Score` disambiguation by ordering (§3.2) — correct but implicit.
   Is `Field(metadata={"typesafe": {"type": "score"}})` needed as an override?
3. Confidence column naming — `__confidence` suffix can collide with regex
   column matching. Alternative: an accessor returning a parallel frame.
4. Is `Config.ai_source` worth keeping at all as a schema-wide default (§5.1)?
   It saves repetition in the single-source case, but two ways to say the same
   thing is the kind of thing pandera has regretted before. Dropping it costs
   one line per field and makes the opt-in unmissable.

**Resolved during review:**

- *Where does `ai_source` live?* **On the field** (§3.1). An earlier draft put it
  on `Config` and used `description` as the AI-parsed marker; that overloaded a
  documentation field, made the opt-in invisible at the point of use, and could
  not express a schema whose fields read different columns. Grouping fields by
  source (§6.1) preserves the one-request-per-row economics that motivated the
  `Config` version.

---

## 14. Documentation plan

- `docs/source/ai_parsing.md` — user guide: the §0 example, type mapping,
  writing good criteria, confidence, caching, cost.
- `docs/source/parsers.md` — extend with AI parsing as a parser kind, and update
  the "pandas only" note as §8.6 lands.
- `docs/source/dtypes.md` — document enum/`Literal`/ordered-category behavior
  once §8 is fixed. This is currently undocumented and is why the gaps went
  unnoticed.
- `docs/source/integrations.md` — add a **TypeSafe AI (Jev)** row alongside
  FastAPI / Pydantic / Mypy.
- `docs/source/reference/ai.rst` — API reference.
- A notebook doing end-to-end ticket triage, runnable with `MockProvider` for
  readers without a key.

---

## 15. Summary

Pydantic AI's Jev integration answers *"fill this object from this text."* The
pandera analogue answers *"fill this **column** from this **corpus**, and tell me
when the answers stop looking right."*

The design adds two arguments to user-facing pandera — `Field(ai_source=...)`
for what a column reads and `ai_provider` for who answers — and otherwise reuses
`Field(description=...)` for questions, the annotation for question types,
`Parser` for the call, and `Check` for everything downstream. Both arguments are
required for anything to happen, which keeps the opt-in visible on the line that
pays for it. The work that makes it real is mostly in §8, and all of it is work
pandera's type system wants anyway.
