# Derived Columns and System One Parsing — Integration Spec

> **Status:** Draft / RFC
> **Author:** pandera maintainers
> **Install:** `pip install 'pandera[typesafe-ai]'`
> **Prior art:**
> [Pydantic AI — TypeSafe (Jev) integration](https://pydantic.dev/docs/ai/models/typesafe/)

---

## 0. TL;DR

This spec proposes **two layers**, one of which is useful on its own.

**Layer 1 — derived columns (pandera core).** Today a `Parser` transforms data
that already exists. Generalize it so a parser can *produce* a column from a
source column, with the relationship declared rather than implied:

```python
class Tickets(pa.DataFrameModel):
    body: str
    n_words: int = pa.ParsedField(source="body", parser=lambda s: s.str.split().str.len())
```

**Layer 2 — System One parsing (`pandera[typesafe-ai]`).** A parser that fills
its target columns by asking a decision model. The schema being filled is a
**plain pandera model** — no new field arguments, no new annotations, nothing
that only makes sense when an AI provider is attached:

```python
import pandera.pandas as pa
from pandera.system_one import SystemOneParser

class Triage(pa.DataFrameModel):                      # an ordinary model
    department: Department = pa.Field(description="Which team should handle this ticket")
    frustration: Frustration = pa.Field(description="How frustrated the customer appears")
    is_urgent: bool = pa.Field(description="The message conveys time-sensitivity")

triage = SystemOneParser(
    "typesafe:jev-1.13.0",
    output=Triage,                                    # the "output type"
    source="ticket_body",
)

triaged = triage(tickets_df)                          # or: schema(parsers=[triage])
```

This mirrors how Pydantic AI factors the same problem. There, `output_type` is
an untouched `BaseModel` and the provider binding lives in the `Agent`. Here,
`output` is an untouched `DataFrameModel` and the binding lives in the
`Parser` — so `Triage` is still a normal schema you can validate with,
serialize, and reuse with no provider in sight.

A previous draft of this spec put `system_one_source` and `system_one_provider`
on `Field`/`Column`/`Config`. §12 records why that was abandoned.

---

## 1. Motivation

### 1.1 Layer 1 is missing from pandera independently

Deriving a column from another column is one of the most common things anyone
does to a dataframe, and pandera has no way to *say* it. You can compute it
before validating, or hide it in a parser closure, but the schema — the thing
that is supposed to describe the data — cannot express that `n_words` comes from
`body`. That costs:

- **Provenance.** A schema does not record which columns are derived, or from
  what. The YAML form of a schema is an incomplete description of the pipeline.
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
exists, layer 2 is a `Parser` subclass and little else — which is the test of
whether the abstraction is right.

Pandera already draws this line, in `docs/source/parsers.md`:

> Validation is the act of verifying whether data follows some set of
> constraints, whereas parsing transforms raw data into some desired set of
> constraints.

### 1.3 Why Jev is a good fit for the layer-2 slot

Jev is a **System One model**: it does not generate text. It answers typed
questions and can only return values from the schema it was given. Three
question types:

| Question | Returns | Shape |
|---|---|---|
| `Noul(instructions=...)` | calibrated probability | `float` in `[0, 1]` |
| `Choice(instructions=..., criteria={opt: desc})` | one option | ≤ 255 options |
| `Score(instructions=..., criteria=[level, ...])` | position on a scale | 2–10 ordered levels |

Vendor-reported: 70–500 ms latency, 1,200 req/min, 250k tokens/sec, $0.042/M
input tokens with output free, a 0% structured-output error rate, and *"a tenth
question costs tokens but almost no time"* — which makes a whole schema per row
the natural request unit.

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

### 1.4 Cost

10,000 tickets, ~400 tokens of state each, 6 fields: ~4M input tokens ≈ **$0.17**
in one pass of ~10,000 requests.

*(All figures in §1.3 and §1.4 are vendor-reported. §12 covers verification
before any of them appear in user-facing docs.)*

---

## 2. Design principles

1. **Two layers, and the lower one ships alone.** Derived columns are a pandera
   feature with its own users. If layer 2 is never built, layer 1 still pays for
   itself.
2. **The schema stays plain.** No argument on `Field`, `Column`, or `Config`
   exists only to serve an AI provider. A model used as a System One `output` is
   indistinguishable from any other model — which is the property that makes
   Pydantic AI's `output_type` pleasant to work with.
3. **The binding lives in the parser.** Provider, source, concurrency, caching,
   and thresholds are constructor arguments on a `SystemOneParser` object —
   the analogue of Pydantic AI's `Agent`.
4. **Plain Python carries the type information.** `Enum`, `IntEnum`, `Literal`,
   and `bool` already say everything the question compiler needs. No pandera-
   specific annotation is introduced.
5. **Unsupported is a construction-time error.** A column that cannot be
   expressed as a question raises `SchemaInitError` when the parser is built,
   before any request — matching Pydantic AI's `UserError` contract.
6. **Deterministic tests.** No test in pandera's suite may require an API key or
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

This is a better starting point than expected. The pipeline ordering pandera
documents —

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
| No error attribution | A bad derived column is not traceable to the parser that made it |
| Column-level parsers cannot create their own column | Provenance cannot live next to the column it describes |

### 3.3 `Parser(source=..., target=...)`

The primitive. Two new optional arguments on the existing class:

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

Semantics:

- `source=None, target=None` — today's behavior exactly. Nothing changes for
  existing code.
- `source` declared — pandera checks those columns exist **before** invoking the
  function and raises `ParserError` naming the parser and the missing column.
- `target` declared — those columns are known to be produced. They are exempt
  from "missing column" errors before parsing, they are permitted under
  `strict=True`, and the parser's output is checked to actually contain them
  (`ParserError` if not).
- Both declared — the parser participates in a dependency graph (§3.6).

```python
word_count = pa.Parser(
    lambda d: d.assign(n_words=d["body"].str.split().str.len()),
    source="body",
    target="n_words",
)

schema = pa.DataFrameSchema(
    {"body": pa.Column(str), "n_words": pa.Column(int)},
    parsers=[word_count],
)
```

When `source` and `target` are both single columns, the function may be written
`Series -> Series` instead of `DataFrame -> DataFrame`; the backend adapts,
matching how `element_wise` already switches the calling convention.

**Compatibility note.** `Parser` currently forwards unrecognized keyword
arguments to the parser function — verified: `Parser(fn, source="body")` today
calls `fn(series, source="body")`. Promoting `source`/`target` to real
parameters is therefore a small breaking change for anyone whose parser function
takes a keyword by those names. Worth a deprecation cycle, or `**parser_kwargs`
could be narrowed to an explicit `parser_kwargs={...}` dict, which is the
cleaner long-term shape regardless.

### 3.4 Declarative: `ParsedColumn` and `ParsedField`

Sugar that puts the provenance next to the column, fixing the locality problem
and the "column-level parsers cannot create their own column" gap:

```python
schema = pa.DataFrameSchema({
    "body": pa.Column(str),
    "n_words": pa.ParsedColumn(
        int,
        source="body",
        parser=lambda s: s.str.split().str.len(),
        checks=pa.Check.ge(1),          # everything Column accepts still works
        coerce=True,
    ),
})
```

```python
class Tickets(pa.DataFrameModel):
    body: str
    n_words: int = pa.ParsedField(
        source="body",
        parser=lambda s: s.str.split().str.len(),
        ge=1,
    )
```

`ParsedColumn` is a `Column` subclass. At schema-build time it desugars into
(a) an ordinary `Column` and (b) a `Parser` with `source`/`target` set, appended
to the schema's `parsers` in dependency order. `ParsedField` is the `FieldInfo`
equivalent. Both are pure sugar over §3.3 — the same relationship `Field` has to
`Column`.

This is where the earlier draft's mistake is corrected. `ParsedColumn` adds a
*construct* whose entire purpose is derivation; `Column(system_one_source=...)`
added an *argument* to a construct that is mostly not about derivation, which is
what made it feel bolted on.

### 3.5 Imperative: `@pa.parser`

`@pa.parser(*fields)` already exists and attaches a transform to those fields.
Generalize it: with `source`, the decorated method *derives* the named fields
instead of transforming them.

```python
class Tickets(pa.DataFrameModel):
    body: str
    n_words: int

    @pa.parser("n_words", source="body")
    def count_words(cls, s):
        """Number of whitespace-separated tokens."""
        return s.str.split().str.len()
```

No `source` → today's meaning, unchanged. This keeps one decorator for one
concept rather than introducing `@pa.derives` alongside it.

### 3.6 Ordering, errors, and serialization

**Ordering.** With `source`/`target` declared, parsers are topologically sorted
by their column dependencies rather than run in list order. Undeclared parsers
keep their list position and run first, preserving today's behavior. A cycle is
a `SchemaInitError` naming the columns in it.

**Errors.** Derivation failures are parsing failures, and pandera has a category
for that. `ParserError` gains subclasses:

- `ParserSourceError` — a declared source column is absent, or has the wrong
  dtype for the parser's declared input.
- `ParserTargetError` — the function ran but did not produce its declared
  targets.

`on_error="null"` and `on_error="drop"` degrade per row instead of failing the
batch, composing with the existing `drop_invalid_rows`.

**Serialization.** `source` and `target` serialize per parser in
`pandera.io`. A callable body cannot be serialized — a YAML schema records the
dependency edges and the parser's `name`/`description`, and deserializing a
schema whose parsers are unresolved raises on `validate` rather than silently
skipping. Layer 2 parsers serialize *completely*, because their "function" is a
declarative question set (§4.8).

### 3.7 Backend scope

Parsers are pandas-only today, and non-pandas backends **silently ignore them**
(§6.4) — which is a much worse failure for derived columns than for transforms,
since the column simply will not be there. Closing that silent path is a
prerequisite, and is independently worth doing (§6.4).

---

## 4. Layer 2: System One parsing

### 4.1 The output model is a plain pandera model

```python
import enum
import pandera.pandas as pa

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
    department: Department = pa.Field(description="Which team should handle this ticket")
    frustration: Frustration = pa.Field(description="How frustrated the customer appears")
    is_urgent: bool = pa.Field(description="The message conveys time-sensitivity")
```

Everything here is either plain Python (`StrEnum`, `IntEnum`, `bool`, member
docstrings) or plain pandera (`Field(description=...)`). `Triage` validates a
dataframe, serializes to YAML, and generates docs with no provider configured.
Nothing about it is AI-specific — which is the whole point, and the property
Pydantic AI's `output_type` has that the earlier draft of this spec gave away.

### 4.2 `SystemOneParser`

```python
class SystemOneParser(pandera.api.parsers.Parser):
    def __init__(
        self,
        provider: str | DecisionProvider,       # "typesafe:jev-1.13.0"
        *,
        output: type[DataFrameModel] | DataFrameSchema | None = None,
        source: str | list[str] | Callable[[Any], dict],
        target: str | list[str] | None = None,
        boolean_threshold: float = 0.5,
        abstain_below: float | None = None,
        confidence: bool | list[str] = False,
        max_concurrency: int = 16,
        cache: ExtractionCache | None = None,
        on_error: Literal["raise", "null", "drop"] = "raise",
    ): ...
```

It is an ordinary layer-1 parser: `source` is what it reads, its `target` is the
columns of `output`. At construction it compiles `output`'s columns into a
question set and raises `SchemaInitError` for any column it cannot express —
before any request is sent.

Note what the argument names are *not*: there is no `system_one_` prefix,
because these arguments live on a dedicated object rather than sharing a
namespace with the rest of `Column`. Removing the need for that prefix is a
direct consequence of the modular factoring.

### 4.3 Three ways to use it

**Standalone**, the closest analogue to `agent.run_sync`:

```python
triage = SystemOneParser("typesafe:jev-1.13.0", output=Triage, source="ticket_body")

triaged = triage(tickets_df)        # -> tickets_df + Triage's columns, validated
```

**Composed into a schema**, where it is just a parser:

```python
schema = pa.DataFrameSchema(
    {"ticket_body": pa.Column(str), **Triage.to_schema().columns},
    parsers=[triage],
    checks=[pa.Check(lambda df: df["department"].isna().mean() < 0.05,
                     name="abstention_rate")],
)

triaged = schema.validate(tickets_df)
```

**Declared on a model.** `Config` has no `parsers` attribute today; adding one
is a small generic core change that mirrors `DataFrameSchema(parsers=...)` and
is useful for every parser, not just this one:

```python
class TicketTriage(Triage):                 # inherits the three answer columns
    ticket_body: str

    class Config:
        parsers = [SystemOneParser("typesafe:jev-1.13.0", source="ticket_body")]
```

With no `output`, the parser resolves its targets from the schema it is attached
to: every column that has a `description`, has an answerable dtype, and is not a
source. Convenient, and the one implicit thing in the design — §12 asks whether
it should exist at all.

### 4.4 Type mapping

The column's dtype determines the question type. This is the pandera
re-expression of Pydantic AI's supported-field-types table.

| Column dtype | Question | Answer → column | Notes |
|---|---|---|---|
| `bool` | `Noul` | `True` iff `p >= boolean_threshold` | |
| `float` + `Field(in_range=(0, 1))` | `Noul` | raw probability, unrounded | |
| `StrEnum` / `Enum` | `Choice` | chosen option | members are the options; ≤ 255 |
| `Literal["a", "b"]` | `Choice` | chosen option | **needs §6.2** |
| any of the above + `nullable=True` | `Choice` or none | option or `NA` | abstention |
| `IntEnum` *(ordered)* | `Score` | nearest level | member docstrings are the rubric; 2–10 levels; **needs §6.3** |
| `float` + ordered `IntEnum` | `Score` | unrounded position (e.g. `1.035`) | |
| `list[EnumT]` | `Noul` per option | list of selected options | multi-label |
| `str`, `datetime`, unbounded numerics, `dict` | ✗ | — | `SchemaInitError` at parser construction |

`Choice` versus `Score` is decided by **ordering**: an unordered category is a
`Choice`, an ordered one is a `Score`. That is semantically right — a `Score`
answer can land *between* levels at `1.035`, which only means something for an
ordered domain — and it is why §6.3 matters.

`Field` arguments shape the question and then validate the answer:

| `Field` argument | Effect |
|---|---|
| `description` | the question text |
| `nullable=True` | adds the "or none" branch; abstention → `NA` |
| `isin=[...]` | narrows a `Choice` to a subset of the dtype's options |
| `in_range=(0, 1)` | marks a `float` as a raw `Noul` probability |

### 4.5 Criteria come from member docstrings

`Choice` takes `criteria={option: description}` and `Score` an ordered list of
level descriptions. Those come from enum member docstrings, mirroring Pydantic
AI's `UseEnumMemberDocstrings`. `Frustration` above compiles to:

```python
Score(
    instructions="How frustrated the customer appears",   # Field(description=...)
    criteria=["Calm, simply stating facts",
              "Frustrated but civil",
              "Very angry, strong language"],             # member docstrings
)
```

Python discards member docstrings at runtime, so this needs source inspection
(§6.5). Worth doing in core rather than the extra: these descriptions belong on
`Column.description` and in generated docs regardless of any provider.

Without docstrings, member names are used as bare criteria. That works, but the
docs should push hard toward docstrings — criteria quality is the biggest single
lever on answer quality, and putting it in the enum keeps it in the schema
rather than in a prompt somewhere else.

### 4.6 Confidence

Jev returns calibrated confidence with every answer. Per-object that is a number
you might log; per-dataframe it is a **column**, and therefore something pandera
can validate.

```python
class Triage(pa.DataFrameModel):
    department: Department = pa.Field(description="...", nullable=True)
    department__confidence: float = pa.Field(ge=0.70)
    ...

triage = SystemOneParser(
    "typesafe:jev-1.13.0",
    output=Triage,
    source="ticket_body",
    confidence=["department"],       # fills department__confidence
    abstain_below=0.55,
)
```

Three levels of strictness compose, all from existing features:

1. `abstain_below` — per row: low-confidence answers become `NA`.
2. `Field(ge=...)` on the confidence column — per row: hard failure with normal
   failure-case reporting.
3. `@pa.dataframe_check` — per batch: "mean confidence > 0.8", "≤5% abstentions",
   "routing distribution hasn't drifted".

The confidence column is declared like any other, not auto-injected: a schema
should describe every column in the frame it validates, and `strict=True` would
otherwise reject a column pandera itself added.

### 4.7 Semantic checks

Judging an existing column, rather than producing a new one, is a `Check` — not
a parser:

```python
class Products(pa.DataFrameModel):
    name: str
    category: Category
    description: str = pa.Field(
        checks=SystemOneCheck(
            "typesafe:jev-1.13.0",
            "The description is a coherent description of a product "
            "belonging to the stated category",
            context=["name", "category"],
            min_probability=0.85,
        )
    )
```

`SystemOneCheck` is an ordinary `Check` with a vectorized predicate, so failure
cases, `lazy=True`, `n_failure_cases`, and `raise_warning` work untouched. It is
guarded by a `PANDERA_SYSTEM_ONE_ENABLED` env var so schemas carrying semantic
checks still run offline, degrading to a skip with a warning.

### 4.8 Serialization

Unlike a general derived column, a System One parser has no opaque callable —
its "function" *is* its declarative question set, so it round-trips completely:

```yaml
columns:
  department:
    dtype: category
    description: Which team should handle this ticket
    nullable: true
parsers:
  - type: system_one
    provider: typesafe:jev-1.13.0
    source: [ticket_body]
    target: [department, frustration, is_urgent]
    criteria:
      department:
        billing: Payment, invoices or subscription issues
        technical: Bugs, outages or integration problems
```

A YAML file is therefore a complete, reviewable specification of a System One
parsing step — questions, sources, types, and checks — with no Python at all.
Because the question text is part of the cache key (§5.3), rewording a
`description` correctly invalidates its cached answers.

---

## 5. Execution

### 5.1 Two axes of parallelism

**Within a row — one request per parser.** Jev's request shape is one state plus
many questions, so every column a parser fills shares a call. TypeSafe reports
this speculative fan-out at 12.2× cheaper and 10.0× faster than one call per
field. A 12-column `output` is one request per row, not twelve.

Two parsers reading different sources are two requests per row — and the parser
object is the unit that makes that legible, since each one names its own source.

**Across rows — bounded async fan-out**, through an
`asyncio.Semaphore(max_concurrency)` behind a token-bucket limiter sized to the
provider's published limits, with exponential backoff honoring `retry-after`.

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

### 5.2 Partitions

Once parsers work on more than pandas (§6.4), the same parser runs
per-partition: `pandas` locally, `polars` per chunk, `dask` one event loop per
partition, `pyspark` via `mapInPandas`. Backend-specific work is confined to
pulling state out of rows and putting typed arrays back into columns.

### 5.3 Caching

Dataframe workloads re-run constantly — a new day appended to last week's, a
notebook cell run six times, a backfill overlapping a prior run. Key:
`sha256(provider_id, resolved_model_version, canonical(state), canonical(questions))`.

Built-ins: in-memory dict, SQLite, parquet directory. `ExtractionCache` is a
`get`/`set`/`stats` protocol. Stats land on the validated frame:

```python
out.attrs["pandera.system_one"]
#> {'rows': 10_000, 'cached': 9_412, 'called': 588, 'model_version': 'jev-1.13.0',
#>  'input_tokens': 241_305, 'est_cost_usd': 0.0101, 'wall_seconds': 4.3}
```

Model version pinning matters: `jev-latest` can change answers between runs, so
the resolved version is part of the key and an unpinned provider warns when a
cache is configured.

### 5.4 State construction

Jev's accuracy degrades with irrelevant context, so `source` is explicit, never
"the whole row":

```python
SystemOneParser(..., source="ticket_body")                          # one column
SystemOneParser(..., source=["ticket_body", "customer_tier"])       # a dict
SystemOneParser(..., source=lambda row: {                           # full control
    "message": row["ticket_body"][:4000],
    "policy": "Refunds within 30 days.",
})
```

Sources are checked against the schema at construction (`SchemaInitError`, not a
runtime `KeyError`), and against the 64k combined / 32k individual token limits
before dispatch, raising with the offending row index rather than surfacing a
provider HTTP error.

### 5.5 No hidden cost

A parser is an object you constructed with a provider argument, so there is no
way to accidentally make a paid call by validating an ordinary schema — the
strongest safety property of the layered design, and one the earlier draft had
to work for.

```python
triage.questions()
#> {'department': Choice(...), 'frustration': Score(...), 'is_urgent': Noul(...)}

triage.plan(tickets_df)
#> SystemOnePlan(rows=10_000, requests=10_000,
#>               est_input_tokens=4_812_000, est_cost_usd=0.2021)
```

---

## 6. Gaps to close first

Verified empirically against `main` (`62f55e2d`) with pandas and polars
installed. Every one of these is a pandera bug or hole that stands on its own.

### 6.1 Parsers have no declared source, so failures are opaque 🔴

```python
pa.DataFrameSchema(
    {"n": pa.Column(int)},
    parsers=pa.Parser(lambda d: d.assign(n=d["nope"].str.len())),
).validate(pd.DataFrame({"body": ["a"]}))
#> KeyError: 'nope'
```

A bare `KeyError`: not a `SchemaError`, no schema context, no parser name, no
indication of which of several parsers failed. §3.3 fixes this by construction.

### 6.2 `Literal` is not a supported dtype 🔴

`Literal` is one of Pydantic AI's two primary `Choice` spellings. How it fails
depends on how it is spelled.

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
valid dtype string, the option set is silently discarded and the column is typed
as that dtype, with no membership constraint and no warning.

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
with matching `__eq__`/`__hash__`. Plain `Enum` — the most common spelling — does
not. A blocker, since Jev returns option *values*.

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

### 6.5 Ordered categories are unreachable from an annotation 🟡

`Engine.dtype(SomeEnum)` always constructs `Category(..., ordered=False)` —
verified. Ordering distinguishes `Score` from `Choice` (§4.4) and independently
enables `ge`/`le` and monotonicity checks on categorical columns.

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

### 6.7 Enum member docstrings are not captured 🟡

Needed for `Choice` criteria and `Score` rubrics (§4.5). **Fix:** a
`pandera.dtypes.member_descriptions(EnumT)` helper using `inspect.getsource` +
`ast`, cached per class, with a graceful fallback when source is unavailable.

### 6.8 `Category` has no per-category description slot 🟡

Criteria for a non-enum categorical have nowhere to live but a `metadata` blob.
This bites when building schemas programmatically from a runtime taxonomy, where
there is no enum class to hang docstrings on. A first-class
`Category(categories=..., descriptions={...})` would fix it and improve generated
docs for ordinary categorical columns.

### 6.9 Summary

| # | Gap | Severity | Blocks | Standalone value |
|---|---|---|---|---|
| 6.1 | Parsers have no declared source/target | 🔴 | layer 1 | yes — error quality |
| 6.2 | `Literal` unsupported; can silently mistype a column | 🔴 | `Choice` | yes — silent data loss |
| 6.3 | `Enum` → members not values | 🔴 | `Choice` | yes — pre-existing bug |
| 6.4 | Parsers pandas-only, silently ignored | 🔴 | non-pandas backends | yes — silent failure |
| 6.5 | No ordered categories from annotation | 🟡 | `Score` | yes |
| 6.6 | pandas/polars enum divergence | 🟡 | portability | yes |
| 6.7 | No member docstring capture | 🟡 | criteria quality | yes — docs |
| 6.8 | No per-category descriptions | 🟡 | programmatic schemas | yes — docs |

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
- Layer 2 is `pandera.system_one`, installed with `pip install 'pandera[typesafe-ai]'`;
  Jev-specific code lives in `pandera.system_one.providers.typesafe`.
- Importing `pandera` without the extra is byte-for-byte unaffected.

---

## 9. Phasing

| Phase | Layer | Scope | Exit criteria |
|---|---|---|---|
| **0** | core | §6.2, §6.3, §6.4(a); cross-backend enum conformance tests | `Enum` and `Literal` round-trip on pandas and polars; non-pandas parsers raise instead of no-op |
| **1** | 1 | `Parser(source=, target=)`, dependency sort, `ParserSourceError`/`ParserTargetError`, `ParsedColumn`/`ParsedField`, `@pa.parser(source=)`, `Config.parsers` | Derived columns work end-to-end on pandas with declared provenance. **Ships with no AI code at all.** |
| **2** | 2 | `Question`/`Decision`, `DecisionProvider`, `TypeSafeProvider`, question compiler + `SchemaInitError` coverage, `SystemOneParser`, async fan-out + limiter | The §0 example runs end-to-end; full §4.4 table covered by replay tests |
| **3** | 2 | §6.5, §6.7; confidence columns, `abstain_below`, cache, stats, `plan()` | Confidence floors and distribution checks work; cache-hit path makes zero network calls |
| **4** | both | §6.4(b), §6.8; polars parsers, `SystemOneCheck`, YAML round-trip, CLI | Same model validates on pandas and polars with identical output |
| **5** | both | dask/modin/pyspark partitioning, multi-label, nested models | 1M-row parse on dask with bounded memory and correct rate limiting |

Phases 0 and 1 are worth doing whether or not layer 2 is ever built.

---

## 10. Testing

Hard constraint: **no test requires an API key or network access.**

- **Layer 1 tests need no provider at all** — the biggest testability win from
  the split. Dependency sorting, cycle detection, source/target validation,
  error types, and serialization are all exercised with `lambda s: s * 2`.
- **Cassettes.** `RecordingProvider` captures real responses once behind
  `PANDERA_RECORD_CASSETTES=1`; `ReplayProvider` serves them in CI.
- **Compiler contract tests.** §4.4 as a parametrized test: every supported
  dtype compiles to the expected question type and criteria; every unsupported
  one raises `SchemaInitError` naming the column.
- **Type-system regression tests** for §6.2–§6.8, written before the fixes,
  including the cross-backend enum matrix.
- **Concurrency tests** with a latency-injecting fake provider: ordering
  preserved, concurrency capped, token bucket throttles, `retry-after` honored,
  one row's failure does not poison the batch.
- **Pipeline-order tests** asserting derivation runs before column parsing,
  before coercion, before checks.
- **Nightly key-gated smoke test** against `jev-latest`, non-blocking.

---

## 11. Documentation plan

- `docs/source/parsers.md` — extend with derived columns as a first-class
  concept, and update the "pandas only" note as §6.4 lands.
- `docs/source/derived_columns.md` — layer 1 user guide.
- `docs/source/system_one.md` — layer 2 user guide: the §0 example, type
  mapping, writing good criteria, confidence, caching, cost.
- `docs/source/dtypes.md` — document enum/`Literal`/ordered-category behavior
  once §6 is fixed. Currently undocumented, which is why the gaps went unnoticed.
- `docs/source/integrations.md` — add a **TypeSafe AI (Jev)** row.
- `docs/source/reference/system_one.rst` — API reference.
- A notebook doing end-to-end ticket triage, runnable with `MockProvider`.

---

## 12. Risks and open questions

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

1. Should `SystemOneParser` be allowed to infer its targets from the attached
   schema (§4.3), or always require an explicit `output`? Inference is the nicer
   one-liner and the only implicit thing left in the design.
2. Does `ParsedColumn` need `ParsedIndex` for symmetry, or is deriving an index
   out of scope?
3. Enum columns holding values vs. members (§6.3) — values recommended; either
   choice is a small breaking change.
4. `Choice`/`Score` disambiguation by ordering (§4.4) is correct but implicit.
   Does it need an explicit override?
5. Confidence column naming — the `__confidence` suffix can collide with regex
   column matching. Alternative: an accessor returning a parallel frame.
6. Should layer 1 allow a parser to *remove* columns, or is
   `strict="filter"` sufficient? Removal would complete the derivation story but
   complicates the dependency graph.

**Resolved during review:**

- *Where does the AI binding live?* **In the parser object**, not on
  `Field`/`Column`/`Config` (§4.2). Earlier drafts added `system_one_source` and
  `system_one_provider` as schema arguments. That made every schema partly an AI
  artifact, put provider concerns in a namespace shared with ordinary column
  options, and — worst — meant `validate()` on a plain-looking schema could
  issue paid requests. Factoring the binding into a `Parser` keeps the schema a
  plain schema, matches how Pydantic AI separates `output_type` from `Agent`,
  and makes the whole feature testable without a provider.
- *Where does the source column live?* On the **parser**, which is the object
  that reads it. An earlier draft put it on `Config`, then on each `Field`.

---

## 13. Summary

Pydantic AI's Jev integration answers *"fill this object from this text"*, and
it does it without changing what a `BaseModel` is. The pandera analogue should
answer *"fill this **column** from this **corpus**, and tell me when the answers
stop looking right"* — without changing what a schema is.

That requires one primitive pandera is missing: a parser that declares the
columns it reads and the columns it produces. Build that, and the System One
integration is a `Parser` subclass whose function happens to be a question set.
Build it well, and derived columns become a pandera feature that people who have
never heard of Jev will use.
