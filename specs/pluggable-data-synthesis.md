# Pluggable Data Synthesis Backends (Realistic Data Generation) Spec

> **Status:** Draft
> **Scope:** `pandera/api/synthesis/` (new), `pandera/strategies/base_strategies.py`
> (additive), `pandera/strategies/_values.py` (new), `pandera/strategies/faker_strategies.py`
> (new), `pandera/strategies/{polars,pyspark,ibis,datafusion}_strategies.py` (new),
> `pandera/strategies/pandas_strategies.py` (additive), `pandera/api/{pandas,polars,
> pyspark,ibis,datafusion}/` (additive), `pandera/config.py` (additive),
> `pyproject.toml` (entry points + optional deps), `docs/source/` (additive)
> **Author:** pandera maintainers

---

## 1. Motivation

`pandera`'s data synthesis layer — `Schema.strategy()`, `Schema.example()`,
and `pa.Field(...)` data generation — is built exclusively on
[`hypothesis`](https://hypothesis.readthedocs.io/). It generates values that are
*valid under the schema's type and `Check` constraints*, but they rarely *look
like real data*. A `str_matches` email column yields `"faH3k0x@qMzgLv.xi"`, a
`str_matches` name column yields `"QzRtaWd IyopLb"`, and an unconstrained string
column yields near-gibberish.

The [Data Synthesis Strategies guide](../docs/source/data_synthesis_strategies.md)
already acknowledges this:

> The generated data uses `hypothesis` to generate random data that satisfies
> the schema... in many cases will not qualitatively look like the original data.
>
> For libraries that generate realistic data, refer to:
> - [Faker](https://github.com/joke2k/faker)
> - [Mimesis](https://github.com/lk-geimfari/mimesis)
> - [SDV](https://github.com/sdv-dev/sdv)

Hypothesis will always be the right backbone for pandera: it is the only engine
that can *guarantee* a value satisfies arbitrary check predicates, and it powers
property-based generation with shrinking and example reduction. But "realistic"
and "constraint-satisfying" are complementary needs, and today pandera has no way
to blend the two — the synthesis path is hard-wired to one engine.

This spec makes synthesis **pluggable and opt-in**. It adds a thin backend
abstraction in front of the value-generating path, ships one production-quality
backend built on the most battle-tested *realistic* generation library, and lets
users and third-party libraries register their own backends. The default remains
today's pure-hypothesis behavior, byte-for-byte compatible.

---

## 2. Research: top 3 battle-tested candidate libraries

The constraints on a good candidate are:

1. **Battle-tested**: mature, widely adopted, actively maintained, permissive
   license, low maintenance risk.
2. **Light weight**: pandera's core is deliberately dependency-light; the
   backend must not drag in a large ML stack by default (it must be *opt-in*).
3. **Constraint-aware interop**: must be able to produce values that still pass
   a pandera `Check`, i.e. either field-level providers that can be filtered /
   constrained, or a documented way to bias output toward a schema.
4. **Deterministic-seedable**: so `Schema.example()` remains reproducible.

The following were evaluated against these criteria (metrics collected
2026-10 from PyPI + GitHub):

| Library | PyPI ver | Python | Core deps | GitHub ⭐ | License | Maintenance |
|---|---|---|---|---|---|---|
| **Faker** (`joke2k/faker`) | 40.41.0 | >=3.10 | `tzdata` only | ~19.4k | MIT | Very active; long track record |
| **Mimesis** (`lk-geimfari/mimesis`) | 22.2.0 | >=3.10 | none (pure py) | ~4.8k | MIT | Active |
| **SDV** (`sdv-dev/SDV`) | 1.38.5 | <3.15,>=3.9 | pandas, numpy, copulas, ctgan (torch) | ~3.6k | Business Source | Active |

Also considered and rejected:

- **synthcity** (~689⭐, Apache-2.0): strong ML-based tabular synthesis, but a
  very heavy dependency graph (torch, opacus, optuna, xgboost, nflows, ...)
  with a restrictive `torch<2.3` pin — a poor fit for pandera's lean core and
  Python 3.14 support.
- **ydata-synthetic** (tensorflow==2.15.* pinned, `<3.12`): even heavier, pins
  TensorFlow, and drops Python 3.12+ — rejected.
- **DataSynthesizer** (old, low activity): rejected on maintenance grounds.
- **Gretel** (managed-SaaS agent): closed/remote, requires a service — rejected.

### 2.1 Selection rationale

The three selected candidates represent two distinct *modes* of synthesis:

**Mode A — field-level realistic generation (constraint-first domains):**
Faker and Mimesis generate *realistic atomic values* (names, emails, phone
numbers, addresses, dates, enums) from a seeded RNG, with no training data and
no heavy dependencies. This composes naturally with pandera's existing
"build a value from a dtype + constraints" path: when a string/category field
carries an identifiable *semantic* (e.g. a name, an email, a phone), the backend
draws a realistic value instead of a random token, then falls back to
hypothesis whenever the value must additionally satisfy a constraint the
provider can't honor.

**Mode B — whole-table statistical synthesis (data-driven):** SDV and synthcity
*learn* joint distributions and correlations from a real reference dataset and
emit entire tables (GaussianCopula, CTGAN). This is a genuinely different use
case — it requires training data and, in SDV's case, a heavier stack. It is a
strong *future* Mode B candidate but out of scope for the first implementation.

**Top pick to implement now: Faker.** It is the largest, longest-lived, most
widely adopted realistic-generation library in Python (MIT, ~19.4k⭐, essential
zero-dependency), it is deterministic and seedable, and its `Faker.provider`
architecture maps 1:1 onto pandera's notion of a *field generator*. It is the
Mode A winner by every criterion and keeps the added dependency surface to
essentially nothing. Mimesis is a close second (faster, more locales) and is
kept as the #2 candidate behind the same interface; SDV is the #3 candidate for
the table-level tier.

---

## 3. Goals & Non-goals

### 3.1 Goals

1. Introduce a **pluggable, opt-in synthesis backend interface** so that the
   value-generating path is no longer hard-wired to hypothesis.
2. Implement the **top candidate (Faker)** as the first production backend,
   producing realistic field values that still satisfy the schema.
3. Keep the default (no backend selected) **byte-for-byte identical** to
   today's hypothesis behavior.
4. Provide a **third-party contribution path** via packaging entry points in
   addition to a programmatic registration API.
5. Preserve backward compatibility of all existing APIs — `strategy()`,
   `example()`, `pa.Check(strategy=...)`, `STRATEGY_DISPATCHER`,
   `CONSTRAINT_DISPATCHER`, extensions — with zero changes to existing tests.
6. Integrate cleanly with the `FieldConstraints` / `compile_field_strategy`
   architecture described in `specs/optimized-strategies.md` — the realistic
   backend enriches the *base value* before constraints are applied, never by
   post-hoc mutation of the result.
7. Keep determinism: `Schema.example(..., seed=n)` is reproducible per backend.
8. **Bring data synthesis to every currently-unsupported backend** — implement
   `strategy()` / `example()` for polars, pyspark, ibis, and datafusion (which
   today either raise `NotImplementedError` or lack the methods entirely), while
   keeping the realistic Mode A backend library-agnostic so it is reused across
   all of them. See §6.

### 3.2 Non-goals

- Replacing hypothesis as the default or as the constraint-satisfaction engine.
- Auto-deriving "semantic" meaning from column names (a fragile heuristic; see
  §5.4 for the explicit opt-in mechanism and why we avoid guessing).
- Implementing the Mode B (SDV / whole-table statistical) backend in this
  milestone. The interface is designed to accommodate it (§5.7) but the
  implementation ships later.
- Realistic (Mode A) enrichment for *distributed* backends (pyspark) in the
  first milestone — the engine-agnostic value core makes this a natural later
  phase (§6.5). Basic, non-realistic synthesis still ships for these backends.
- Changing validation semantics of any `Check`.

---

## 4. Background — current architecture

### 4.1 Where synthesis happens today

```
pandera/strategies/
├── __init__.py              # module docstring; lazy tensordict re-export
├── base_strategies.py       # SearchStrategy stub, STRATEGY_DISPATCHER,
│                            # CONSTRAINT_DISPATCHER, strategy_import_error
├── constraints.py           # FieldConstraints (specs/optimized-strategies.md)
├── pandas_strategies.py     # dtype_strategy, compile_field_strategy,
│                            # field_element_strategy, series/column/index/
│                            # dataframe_strategy
└── xarray_strategies.py     # data_array_strategy, dataset_strategy
```

`Schema.strategy()` and `Schema.example()` (in `pandera/api/pandas/container.py`)

```290:336:pandera/api/pandas/container.py
@strategy_import_error
def strategy(self, *, size: int | None = None, n_regex_columns: int = 1):
    ...
    return st.dataframe_strategy(self.dtype, columns=self.columns, ...)

def example(self, size=None, n_regex_columns=1):
    ...
    return self.strategy(size=size, n_regex_columns=n_regex_columns).example()
```

delegate to `pandas_strategies.py`. The value-producing leaf is
`compile_field_strategy(pandera_dtype, constraints)`, which turns a merged
`FieldConstraints` into a single hypothesis `SearchStrategy`, delegating
dtype-specific bridging to `pandas_dtype_strategy(...)`.

### 4.2 The seam we extend

Two natural insertion points exist, and both live inside `pandas_strategies.py`:

1. **The base-value draw**, produced by `pandas_dtype_strategy` /
   `compile_field_strategy`: today a string draws a random token; an int draws
   from a bounded integer range. This is where "realistic" must attach.
2. **The constraint application**, via `compile_field_strategy` (§4.3 of the
   optimized-strategies spec) which folds `FieldConstraints` (bounds,
   membership, regex, residual filters) on top of the base value.

The design principle: **realistic backends only influence point (1)** — they
supply a *better-informed base value*. Constraints are still compiled by the
existing hypothesis machinery (point 2), so correctness and guarantee are
unchanged. If the realistic base cannot satisfy a given constraint set, the
backend simply declines (§5.5) and the default path draws the value.

---

## 5. Proposed architecture

### 5.1 Core abstraction: `SynthesisBackend` protocol

A new module introduces two small protocols. They are duck-typed (structural),
so a backend need only implement the methods relevant to the synthesis mode it
supports.

```
pandera/api/synthesis/
├── __init__.py              # public re-exports: register_synthesis_backend,
│                            # get_synthesis_backend, SynthesisBackend,
│                            # FieldGenerator, SemanticType
├── protocol.py              # SynthesisBackend, FieldGenerator protocols
├── registry.py              # backend registry + entry-point loader
└── semantic.py              # SemanticType enum (+ helper mapping functions)
```

> **Naming note.** This lives in `pandera/api/synthesis/` to sit beside the
> other pluggable API surfaces (`pandera/api/extensions.py`), but depends only
> on `pandera/strategies/` primitives (`FieldConstraints`, `SearchStrategy`),
> so the direction of dependency is API → strategies, matching the existing
> layout.

```python
# pandera/api/synthesis/protocol.py

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Protocol, runtime_checkable

from pandera.strategies.constraints import FieldConstraints
from pandera.strategies.semantic import SemanticType


@runtime_checkable
class FieldGenerator(Protocol):
    """Generates realistic *base values* for a single field.

    A ``FieldGenerator`` is the Mode A building block: given a pandera dtype,
    the schema ``FieldConstraints`` (so it can decide feasibility without a
    trial-and-error loop) and an optional explicit ``semantic`` tag, it produces
    either a hypothesis ``SearchStrategy`` yielding realistic values, a
    concrete value or iterable of values, or ``None`` to signal "no opinion —
    use the default hypothesis path".
    """

    def generate(
        self,
        pandera_dtype: DataType,          # pandera.engines dtype object
        constraints: FieldConstraints,
        *,
        semantic: SemanticType | None = None,
        seed: int | None = None,          # per-backend reproducibility
    ) -> SearchStrategy | object | Iterator[object] | None:
        """Return a realistic generator, a single value, an iterable of
        candidate values, or ``None`` (default path)."""
        ...
```

```python
@runtime_checkable
class SynthesisBackend(Protocol):
    """A named, registered data-synthesis backend (Mode A, Mode B, or both).

    Backends are registered under a string name (see §5.2) and selected via
    ``schema.example(..., synthesis_backend=...)`` or the
    ``PANDERA_SYNTHESIS_BACKEND`` config. A backend advertises its capabilities
    by implementing the corresponding protocol method; unimplemented
    capabilities are simply not used.
    """

    name: str

    # Mode A (field-level realistic base values). If the backend does not
    # implement this it simply contributes no field enrichment.
    def field_generator(self) -> FieldGenerator: ...
    # convenience: also may expose a plain ``field_strategy(...)`` callable
    # that is duck-typed to the (dtype, constraints, semantic, seed) shape.

    # Mode B (whole-table statistical synthesis, future). Optional.
    def synthesize(
        self,
        schema,                        # schema/model metadata
        *,
        reference_data=None,           # Mode B learns from this; Mode A ignores
        size: int | None = None,
        seed: int | None = None,
    ) -> Any: ...
```

The `SynthesisBackend` is deliberately a *structural* protocol rather than an
ABC with forced methods: a minimal Faker backend implements only `name` +
`field_generator()`, while a future SDV backend implements `name` +
`synthesize()`. `@runtime_checkable` lets pandera detect capabilities with
`isinstance(backend, SynthesisBackend)` and `hasattr`-style probing without
forcing inheritance.

### 5.2 Registry and third-party contribution (entry points)

```python
# pandera/api/synthesis/registry.py

from __future__ import annotations

import warnings
from collections.abc import Callable

_SYNTHESIS_BACKENDS: dict[str, SynthesisBackend] = {}


def register_synthesis_backend(name: str, backend: SynthesisBackend) -> None:
    """Register a synthesis backend under a string name (additive, opt-in).

    Raises ``ValueError`` if ``name`` is already registered by a *different*
    object. Re-registering the identical object is a no-op (idempotent), so
    importing a schema module twice does not raise.
    """
    existing = _SYNTHESIS_BACKENDS.get(name)
    if existing is not None and existing is not backend:
        raise ValueError(
            f"Synthesis backend {name!r} is already registered as {existing!r}."
        )
    _SYNTHESIS_BACKENDS[name] = backend


def get_synthesis_backend(name: str | None) -> SynthesisBackend | None:
    """Resolve a backend by name.

    ``None`` selects the default (pure-hypothesis) path. Unknown names are
    resolved against installed third-party entry points once, then raise
    ``SynthesisBackendNotFoundError`` if still unknown.
    """
    if name is None:
        return None
    backend = _SYNTHESIS_BACKENDS.get(name)
    if backend is not None:
        return backend
    _load_entry_point_backends()          # lazy importlib.metadata scan
    backend = _SYNTHESIS_BACKENDS.get(name)
    if backend is None:
        from pandera.errors import SynthesisBackendNotFoundError
        raise SynthesisBackendNotFoundError(
            f"No data synthesis backend named {name!r} is registered. "
            "Install the corresponding package (e.g. pandera[faker]) or "
            "register it with pandera.api.synthesis.register_synthesis_backend."
        )
    return backend
```

Entry points let third-party libraries contribute backends **without
modifying pandera**. A third-party package `my_pandera_synth` declares in its
`pyproject.toml`:

```toml
[project.entry-points."pandera.synthesis_backends"]
my_synth = "my_pandera_synth:backend"      # resolves to a SynthesisBackend
```

and pandera discovers it lazily (only when that name is requested):

```python
def _load_entry_point_backends() -> None:   # registry.py (internal)
    from importlib.metadata import entry_points
    eps = entry_points(group="pandera.synthesis_backends")
    for ep in eps:
        if ep.name in _SYNTHESIS_BACKENDS:
            continue
        try:
            obj = ep.load()                  # lazy; never at import time
        except Exception:                    # broken third-party backend
            warnings.warn(f"Failed to load synthesis backend {ep.name}: ...")
            continue
        _SYNTHESIS_BACKENDS[ep.name] = obj
```

Installed pandera backends are registered the same way, so the built-in Faker
backend is shipped via pandera's own `[project.entry-points]` (plus a programmatic
registration on first use for the local/editable case).

### 5.3 Opt-in selection

Selection is additive and defaults to the status quo:

- **Per call** — `Schema.example(..., synthesis_backend="faker")`,
  `Schema.strategy(..., synthesis_backend="faker")`, and the model-level
  equivalents. This is the primary mechanism.
- **Global** — a new `PANDERA_SYNTHESIS_BACKEND` env var / `PanderaConfig`
  field `synthesis_backend: str | None = None`. When set, any `example()` /
  `strategy()` call without an explicit backend uses it.

```python
# pandera/api/pandas/container.py (additive)

@strategy_import_error
def strategy(
    self,
    *,
    size: int | None = None,
    n_regex_columns: int = 1,
    synthesis_backend: str | None = None,   # NEW
):
    import pandera.strategies.pandas_strategies as st
    if synthesis_backend is None:
        synthesis_backend = get_config().synthesis_backend   # None by default
    return st.dataframe_strategy(
        self.dtype, columns=self.columns, checks=self.checks,
        unique=self.unique, index=self.index, size=size,
        n_regex_columns=n_regex_columns,
        synthesis_backend=synthesis_backend,   # forwarded; None == default
    )
```

`pandas_strategies.dataframe_strategy` / `column_strategy` / `field_element_strategy`
thread this value down to the leaf; when it is `None` the code path is
**textually the same as today** (a single early branch), so there is no
behavioral change.

### 5.4 Semantic typing (the opt-in interface that unlocks realism)

A generator cannot know a column is meant to be a *name* rather than a *zip
code* from the dtype alone — both are `str`. Realism therefore needs a *semantic
signal*. We provide an explicit, opt-in `semantic` tag rather than guessing
from column names (name-based inference is fragile, localised, and
unpredictable, and would break determinism guarantees):

```python
# pandera/api/synthesis/semantic.py

from enum import Enum

class SemanticType(str, Enum):
    NAME          = "name"
    FIRST_NAME    = "first_name"
    LAST_NAME     = "last_name"
    EMAIL         = "email"
    PHONE         = "phone"
    ADDRESS       = "address"
    CITY          = "city"
    STATE         = "state"
    POSTAL_CODE   = "postal_code"
    COUNTRY       = "country"
    COMPANY       = "company"
    JOB           = "job"
    DATE          = "date"
    DATETIME      = "datetime"
    BOOLEAN       = "boolean"
    CURRENCY      = "currency"
    UUID          = "uuid"
    ...
```

Attached via a new optional `semantic=` keyword on `Column` / `Index` / `Field`:

```python
import pandera.pandas as pa

schema = pa.DataFrameSchema({
    "customer":  pa.Column(str, checks=[pa.Check.str_length(1, 80)],
                           semantic="name"),
    "email":     pa.Column(str, checks=[pa.Check.str_matches(EMAIL_RE)],
                           semantic="email"),
    "country":   pa.Column(str, pa.Check.isin(COUNTRIES), semantic="country"),
})
```

```python
# pandera/api/pandas/components.py (additive)

class Column(...):
    def __init__(self, dtype, checks=None, *, semantic: SemanticType | str | None = None, ...):
        self.semantic = SemanticType(semantic) if semantic is not None else None
```

The `semantic` tag is pure metadata for synthesis — it has **no validation
semantics** and does not affect `validate()`. If a `semantic` is untranslatable
to a realistic provider, synthesis silently falls back to hypothesis (§5.5).

**Secondary channel for existing schemas (no new tags required):** when no
`semantic` is set, the Faker backend also consults the field's `FieldConstraints`
to detect a few *unambiguous, high-value* cases from the schema itself — e.g. a
`str_matches` with the canonical email regex, or a `str_matches` that exactly
equals a Faker provider's natural pattern — and pairs the provider with the
constraint. This is opt-in per backend and fully documented; it never overrides
an explicit `semantic`.

### 5.5 Integration with the hypothesis engine (Mode A enrichment)

Inside `compile_field_strategy` (the leaf that turns `FieldConstraints` into a
base strategy), a single new hook at the top selects the base value:

```
compile_field_strategy(pandera_dtype, constraints, *, semantic=None, synthesis_backend=None)
    |
    | if synthesis_backend provides a field generator AND it returns non-None:
    |   base = backend_field.generate(pandera_dtype, constraints,
    |                                semantic=semantic, seed=seed)
    |   if base is not None:
    |       strategy = coerce_to_search_strategy(base)   # builds/st.just/sampled_from
    |       ^--- hypothesis SearchStrategy accepted as-is; built object wrapped in st.just
    |   else:
    |       strategy = default pure-hypothesis base (unchanged)
    |
    v
    apply constraints on top of `strategy` (existing merge/compile logic, unchanged)
```

```python
# pandera/strategies/pandas_strategies.py (additive)

def compile_field_strategy(
    pandera_dtype,
    constraints: FieldConstraints,
    *,
    semantic: SemanticType | None = None,
    synthesis_backend: str | None = None,
    seed: int | None = None,
) -> SearchStrategy:
    backend = get_synthesis_backend(synthesis_backend)   # None → today's path
    strategy = None
    if backend is not None and HAS_HYPOTHESIS:
        try:
            gen = backend.field_generator()
            base = gen.generate(
                pandera_dtype, constraints, semantic=semantic, seed=seed,
            )
        except Exception:
            base = None
        if base is not None:
            strategy = _coerce_base(base)   # st.builds / st.just / sampled_from
    if strategy is None:
        strategy = _default_base_strategy(pandera_dtype, constraints)  # unchanged

    # ... existing constraint application on top of `strategy` (unchanged) ...
```

Crucially, the realistic base is still **filtered / constrained by the normal
hypothesis machinery**: a `FieldGenerator` that produces emails is passed
through the existing `str_matches` / length / membership compilation at §4.3 of
the optimized-strategies spec. If a realistic provider cannot honor a
constraint (e.g. a `name` field with `isin={"A", "B"}` that no name provider
matches), the backend is responsible for returning `None` for that
dtype/constraint combination, and the default path draws a compatible value.
A `FieldGenerator` must therefore be written to *pre-check feasibility against
`constraints`* rather than yield values and hope — this mirrors exactly the
`CONSTRAINT_DISPATCHER` philosophy in `specs/optimized-strategies.md` (merge
first, generate once).

`_coerce_base` handles the three legal returns:

```python
def _coerce_base(base) -> SearchStrategy | None:
    import hypothesis.strategies as st
    if isinstance(base, st.SearchStrategy):
        return base
    if isinstance(base, Iterator):
        return st.sampled_from(list(base))
    return st.just(base)          # a single concrete realistic value
```

### 5.6 Built-in Faker backend (the implemented top candidate)

A new module `pandera/strategies/faker_strategies.py` provides a `FakerBackend`
that registers itself under the name `"faker"` (via pandera's own entry point
and programmatic registration) and implements `field_generator()`.

Dependency policy: **Faker becomes a new optional extra, never a core
dependency.**

```toml
# pyproject.toml (additive)
strategy-faker = [
    "faker >= 25",
]
```

```python
# pandera/strategies/faker_strategies.py

from pandera.api.synthesis import register_synthesis_backend
from pandera.api.synthesis.protocol import FieldGenerator
from pandera.api.synthesis.semantic import SemanticType

_SEMANTIC_TO_FAKER = {
    SemanticType.NAME:        lambda f: f.name(),
    SemanticType.FIRST_NAME:  lambda f: f.first_name(),
    SemanticType.LAST_NAME:   lambda f: f.last_name(),
    SemanticType.EMAIL:       lambda f: f.email(),
    SemanticType.PHONE:       lambda f: f.phone_number(),
    SemanticType.ADDRESS:     lambda f: f.address(),
    SemanticType.CITY:        lambda f: f.city(),
    SemanticType.STATE:       lambda f: f.state(),
    SemanticType.POSTAL_CODE: lambda f: f.postcode(),
    SemanticType.COUNTRY:     lambda f: f.country(),
    SemanticType.COMPANY:     lambda f: f.company(),
    SemanticType.JOB:         lambda f: f.job(),
    SemanticType.BOOLEAN:     lambda f: f.boolean(),
    SemanticType.UUID:        lambda f: f.uuid4(),
    SemanticType.CURRENCY:    lambda f: f.pricetag(),
    ...
}

class _FakerFieldGenerator(FieldGenerator):
    def __init__(self, locale: str | list[str] | None = None):
        from faker import Faker
        self._factory = Faker(locale=locale)   # single shared, seedable factory

    def generate(self, pandera_dtype, constraints, *, semantic=None, seed=None):
        if semantic is None:
            semantic = self._infer_semantic(pandera_dtype, constraints)  # §5.4
        if semantic is None or semantic not in _SEMANTIC_TO_FAKER:
            return None                       # decline → default hypothesis path
        if not self._feasible(constraints):   # pre-check vs constraints (§5.5)
            return None
        if seed is not None:
            self._factory.seed_instance(seed + hash(semantic) & 0xFFFFFFFF)
        provider = _SEMANTIC_TO_FAKER[semantic]
        return _builds(self._factory, semantic, provider, constraints)
```

Where `_builds` wraps the provider in a hypothesis `st.builds(...)`/`st.just(...)`
and, where the constraint set narrows it cheaply, a single `st.sampled_from(...)`
of pre-computed realistic candidates (e.g. `isin` sets) so that no
trial-and-error filtering occurs — again mirroring the merged-constraints
philosophy.

The `_feasible` pre-check is the key correctness guard: it rejects any
`FieldConstraints` combination the provider manifestly cannot satisfy (e.g.
`eq` to a value no provider would emit, an `isin` disjoint from the provider's
domain, or a `regex_fullmatch` pattern the provider's output cannot match).
Only *feasible* combinations enrich; everything else returns `None` and stays on
the default path. This keeps the guarantee that **generated data always
satisfies the schema**, because the realistic base still flows through the
constraint compiler.

### 5.7 Mode B placeholder (future SDV / table-level tier)

The interface reserves a `synthesize(...)` method for data-driven backends, and
this milestone only adds the plumbing — no Mode B implementation ships.

A future SDV backend would:

1. Implement `SynthesisBackend.synthesize(schema, reference_data, size, seed)`.
2. Learn a model from `reference_data` (e.g. `SDV` + `GaussianCopulaSynthesizer`),
   constrained/post-processed so the output passes the schema's own checks
   (SDV can be re-validated with `schema.validate(...)` in a loop, or its
   constraints can be biased from `FieldConstraints`).
3. Be selected via the same `synthesis_backend="sdv"` mechanism and entry-point
   machinery, shipped by a *separate* extra `pandera[strategy-sdv]`.

Backwards compatibility for the entry-point story is identical, which is why
§5.2/§5.3 generalize over both tiers now rather than being Faker-specific.

### 5.8 Determinism and reproducibility

- `Schema.example(..., seed=n)` seeds the hypothesis global and forwards `seed`
  to the backend (`field_generator.generate(..., seed=n)`), so a given
  `(schema, seed, backend)` always reproduces identical output.
- Faker instances are seeded via `Faker.seed_instance(...)` scoped per semantic
  to keep cross-column independence.
- The default (no backend) path remains seeded exactly as today, so existing
  `example(seed=n)` outputs are unchanged.

---

## 6. Backend coverage: bringing data synthesis to every backend

### 6.1 Current support matrix

Synthesis support today is uneven across backends:

| Backend | `strategy()` / `example()` today | Implemented in |
|---|---|---|
| pandas | ✓ full support | `pandera/strategies/pandas_strategies.py` |
| geopandas | ✓ works (pandas-backed, maps to `GeoDataFrame`) | `pandera/api/geopandas/container.py` |
| xarray | ✓ `data_array_strategy` / `dataset_strategy` | `pandera/strategies/xarray_strategies.py` |
| tensordict | ✓ lazily exposed | `pandera/strategies/tensordict_strategies.py` |
| **polars** | ✗ `NotImplementedError` | `pandera/api/polars/container.py` |
| **pyspark** | ✗ methods absent | `pandera/api/pyspark/` |
| **ibis** | ✗ `NotImplementedError` | `pandera/api/ibis/container.py` |
| **datafusion** | ✗ `NotImplementedError` | `pandera/api/datafusion/container.py` |

(dask/modin are pandas-backed and inherit the pandas machinery where it applies,
but are not first-class synthesis backends today.) This section makes
**polars, pyspark, ibis, and datafusion** first-class synthesis backends and
keeps the realistic Mode A enrichment (Faker) usable by all of them.

### 6.2 Two separable problems

"Adding data synthesis to a backend" decomposes into two layers:

1. **Field value compilation** — turning a dtype + merged `FieldConstraints` into
   a stream of *valid values*. Today this is hypothesis-specific
   (`SearchStrategy`) and lives in `pandas_strategies.py` / `xarray_strategies.py`.
2. **Container assembly** — building the backend-native table from those values:
   `pl.DataFrame`, a distributed `pyspark.sql.DataFrame`, an ibis `Table` /
   expression, or a DataFusion `Table`.

The key realization: **problem (1) is engine-agnostic**. A `FieldConstraints`
merge and a feasible realistic base value do not care what dataframe library
will hold the result. Only problem (2) is backend-specific. This is exactly what
makes the pluggable architecture of §5 reusable across every backend.

### 6.3 An engine-agnostic value core

The current leaf (`compile_field_strategy`) returns a hypothesis `SearchStrategy`.
Polars/pyspark/ibis/datafusion have no `SearchStrategy` analog (and
`hypothesis.extra` covers numpy/pandas, not these engines). We therefore extract
a **deterministic scalar sampler** that encodes the *same* merged-constraint
logic but emits native scalar values lazily instead of a `SearchStrategy`:

```
pandera/strategies/
├── base_strategies.py        # unchanged: dispatchers, decorators
├── constraints.py            # unchanged: FieldConstraints
├── _values.py                # NEW: engine-agnostic scalar value core
│                             #   compile_constraints_to_sampler(dtype, constraints,
│                             #     *, semantic=None, synthesis_backend=None, seed)
│                             #     -> iterator of native scalar values
├── pandas_strategies.py      # keeps hypothesis path; may reuse _values for realism
├── faker_strategies.py       # NEW: realistic FieldGenerator (engine-agnostic)
├── polars_strategies.py      # NEW: thin bridge (container assembly)
├── pyspark_strategies.py     # NEW: thin bridge (container assembly)
├── ibis_strategies.py        # NEW: thin bridge (container assembly)
└── datafusion_strategies.py  # NEW: thin bridge (container assembly)
```

```python
# pandera/strategies/_values.py (schematic)
from collections.abc import Iterator
from pandera.strategies.constraints import FieldConstraints
from pandera.api.synthesis.semantic import SemanticType
from pandera.api.synthesis.protocol import get_synthesis_backend

def compile_constraints_to_sampler(
    pandera_dtype,
    constraints: FieldConstraints,
    *,
    semantic: SemanticType | None = None,
    synthesis_backend: str | None = None,
    seed: int | None = None,
) -> Iterator:
    rng = random.Random(seed)          # deterministic, per-call RNG
    backend = get_synthesis_backend(synthesis_backend)
    # Resolve the *base* (realistic provider or plain dtype draw), exactly as in
    # §5.5 but yielding scalars instead of a SearchStrategy:
    base = _base_value_iter(pandera_dtype, constraints,
                            semantic=semantic, backend=backend, rng=rng)
    # Apply the merged constraints (bounds / membership / regex / residual)
    # to each drawn scalar — the same fold logic as compile_field_strategy.
    checker = _constraint_checker(constraints)
    for value in base:
        if checker(value):
            yield value
```

A per-backend `*_strategy.py` then wraps the shared core in one small function
that (a) bridges the backend dtype to a canonical form / pandas-convertible
representation, (b) consumes the sampler, and (c) builds the native container,
all shipped behind the standard `strategy()` / `example()` methods. Each new
module is ~100 lines of assembly glue; **none of the realistic/Faker logic is
duplicated** because it lives only in the engine-agnostic core and the
`FieldGenerator` protocol.

### 6.4 Determinism in distributed backends

pyspark is distributed, so a single global seeded `random.Random` cannot be
threaded across executors. Two supported strategies, both reproducible from
`seed=n`:

1. **Driver materialization (default, deterministic):** draw the required rows
   on the driver via the seeded sampler and materialize with
   `spark.createDataFrame(pd_frame)`. Simple, byte-reproducible, and identical
   semantics to every other backend.
2. **Per-partition seeding (scalable):** derive per-partition seeds from the
   overall seed (`seed + partition_id`) and let each partition draw its own
   slice. Reproducible given a fixed number of partitions; best for large
   sizes where driver materialization is memory-bound.

Documented reproducibility contract for every backend: `example(seed=n)` (and
`strategy(..., seed=n).example()`) reproduces identical output; the pandas
default path (hypothesis, no backend) is unchanged.

### 6.5 Phasing

| Phase | Scope |
|---|---|
| **1 — pandas realistic (this milestone)** | the pluggable interface (§5), engine-agnostic value core (§6.3), Faker `FieldGenerator`, `semantic=` metadata. |
| **2 — polars** | native `strategy()` / `example()` via the shared core + Faker (highest demand after pandas). |
| **3 — ibis, datafusion** | thin bridges; same signatures, native return types. |
| **4 — pyspark** | thin bridge + distributed seeding (§6.4). |
| **all — Mode B (SDV)** | whole-table statistical synthesis is backend-agnostic (emits a native table) and cross-cuts every backend via the reserved `synthesize()` capability. |

Each phase keeps the identical `strategy(size=, n_regex_columns=, synthesis_backend=, seed=)` / `example(...)` signatures (polars/datafusion already stub the first two kwargs; pyspark/ibis gain the methods new) so the opt-in, pluggable story is uniform across every backend. The shared value core means later phases are mostly additive glue and reuse the entire realistic/enrichment and constraint-merging machinery built in Phase 1.

---

## 7. API surface (full summary)

| API | Kind | Notes |
|---|---|---|
| `Schema.example(..., synthesis_backend="faker", seed=None)` | param (additive) | opt-in |
| `Schema.strategy(..., synthesis_backend="faker", seed=None)` | param (additive) | opt-in |
| `DataFrameModel.example(..., synthesis_backend=...)` | param (additive) | mirrors `strategy` |
| `pa.Column(..., semantic=...)` / `pa.Index(..., semantic=...)` / `Field(...)` | param (additive) | metadata only |
| `pandera.api.synthesis.register_synthesis_backend(name, backend)` | function (new) | programmatic |
| `pandera.api.synthesis.get_synthesis_backend(name)` | function (new) | resolver |
| `entry_points group="pandera.synthesis_backends"` | packaging (new) | third-party |
| `PANDERA_SYNTHESIS_BACKEND` env var / `config.synthesis_backend` | config (additive) | global opt-in |
| `pandera[strategy-faker]` extra | packaging (new) | Faker dep |
| `pandera.errors.SynthesisBackendNotFoundError` | exception (new) | unknown backend |
| `polars/pyspark/ibis/datafusion` `strategy()` / `example()` | methods (new) | first-class synthesis (§6); same signatures, native return types |
| `PANDERA_SYNTHESIS_BACKEND` for non-pandas backends | config | uniform opt-in across all backends |

No existing signature, enum, or dispatch table is modified. Every new
parameter defaults to the current behavior, and the backend methods newly
added to polars/pyspark/ibis/datafusion follow the same contract.

---

## 8. Faker constraint mapping (reference table)

The built-in backend chooses a provider from `semantic`, then checks it against
`FieldConstraints`. Representative mapping and guard behavior:

| `semantic` | Faker provider | Guarded against |
|---|---|---|
| `name` | `f.name()` | `str_length`, `regex_fullmatch` (falls back if unmatcheable) |
| `email` | `f.email()` | `regex_fullmatch` (email regex is satisfiable) |
| `phone` | `f.phone_number()` | `str_length` |
| `country` / `state` / `city` | `f.country()/state()/city()` | `isin` (pre-compute sampled domain) |
| `boolean` | `f.boolean()` | `isin={True,False}`, `eq` |
| `date` / `datetime` | `f.date_object()`/`f.date_time()` | `min_value`/`max_value`, `isin` |
| `uuid` | `f.uuid4()` | `str_length`, `regex_fullmatch` |
| `currency` | `f.pricetag()` | numeric bounds |

Rules:
- If `semantic` is `None` (and no safe inference applies) → decline (`None`).
- If any constraint is provably unmatcheable by the chosen provider → decline.
- If `isin` is present, the backend *pre-computes* a candidate set from the
  provider (bounded) and returns `st.sampled_from(...)` so no filter chain
  forms.
- If the provider returns a value type incompatible with `pandera_dtype` after
  conversion → decline and let hypothesis draw.
- All declines fall through to the default pure-hypothesis path; correctness is
  preserved in every case.

---

## 9. Worked example

```python
import pandera.pandas as pa
from pandera.api.synthesis import SemanticType

EMAIL_RE = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

class CustomerSchema(pa.DataFrameModel):
    name:   str = pa.Field(semantic="name",    str_length={"min_value": 2, "max_value": 80})
    email:  str = pa.Field(semantic="email",   str_matches=EMAIL_RE)
    phone:  str = pa.Field(semantic="phone",   str_length={"min_value": 7, "max_value": 20})
    country: str = pa.Field(semantic="country", isin=["US", "CA", "GB", "MX"])
    age:    int = pa.Field(ge=18, le=99)

# Opt-in; default remains pure-hypothesis.
df = CustomerSchema.example(size=100, synthesis_backend="faker", seed=42)
```

`name`/`email`/`phone`/`country` now hold realistic values that still pass their
constraints; `age` is generated by the unchanged hypothesis path (Faker has no
semantic for it and declines). Dropping `synthesis_backend="faker"` reproduces
today's behavior exactly.

---

## 10. Backwards compatibility matrix

| Behavior | Today | With backend=None (default) | With backend="faker" |
|---|---|---|---|
| `example()`/`strategy()` signatures | fixed | identical (new kwargs default to None) | + opt-in kwargs |
| Output values (no backend) | hypothesis | **unchanged, byte identical** | n/a (different path) |
| `schema.example(seed=n)` | reproducible | unchanged | reproducible per backend |
| `pa.Check(strategy=...)` | works | works | works |
| `STRATEGY_DISPATCHER` / `CONSTRAINT_DISPATCHER` | as-is | read-only, unchanged | unchanged |
| extensions `register_check_method` | works | works | works |
| polars/pyspark/ibis/datafusion `strategy()` | *absent / `NotImplementedError`* | **now fully implemented** (new methods; no prior behavior to preserve) | realistic enrichment available (Phase 2–4, §6.5) |
| `semantic=` on Column/Field | n/a | no-op metadata (accepted, ignored) | drives realistic base |

Because `semantic=` is pure metadata with no validation behavior, adding it is
safe even for callers who never use synthesis. Because the new
`strategy()`/`example()` kwargs default to their existing values, every
existing call site and test is unaffected.

---

## 11. Error handling

| Condition | Behavior |
|---|---|
| Unknown `synthesis_backend="x"` | `SynthesisBackendNotFoundError` at `strategy()`/`example()` time, with a hint to register/install. |
| Backend raises inside `field_generator().generate(...)` | caught, logged via warning; falls back to default path (never fails synthesis). |
| Backend produces a value that later fails a constraint | impossible by construction: feasible base still compiled through the constraint compiler; infeasible bases decline. |
| `semantic` invalid string | `ValueError` from `SemanticType(semantic)` at schema definition time (fail fast). |
| Faker not installed but `"faker"` requested | `ImportError` equivalent to today's `strategy_import_error` message, pointing to `pip install 'pandera[strategy-faker]'`. |
| Broken third-party entry point | warning at load time; skipped; other backends unaffected. |

Existing error paths (missing hypothesis, `Unsatisfiable`/`ConstraintConflictError`
→ schema-definition error) are unchanged in the default path.

---

## 12. Implementation plan

1. **Primitives** — add `pandera/api/synthesis/{protocol,registry,semantic}.py`
   and `pandera/strategies/semantic.py` (or fold into `api/synthesis/semantic.py`).
   Registry + entry-point loader + `SynthesisBackendNotFoundError`.
2. **Plumbing** — thread `synthesis_backend`/`seed` through
   `dataframe_strategy` → `column_strategy` → `field_element_strategy` →
   `compile_field_strategy`; add the §5.5 hook; keep the `None` path identical.
   Wire new kwargs on `Schema.strategy()`/`example()` and model equivalents.
   Add `config.synthesis_backend` + `PANDERA_SYNTHESIS_BACKEND` env var.
3. **`semantic=` metadata** — add to `Column`/`Index`/`Field`; plumb through
   `field_element_strategy`; no validation behavior.
4. **Engine-agnostic value core** — extract the merged-constraint fold into a
   deterministic scalar sampler in `pandera/strategies/_values.py` (§6.3), so
   pandas and the new backends share one value-production path; realistic
   `FieldGenerator`s drop in unchanged.
5. **Faker backend** — `pandera/strategies/faker_strategies.py` implementing
   `FakerBackend` + semantic/provider table + `_feasible` pre-check +
   `_coerce_base` support; register `"faker"` via entry point and
   programmatically.
6. **Packaging** — add `strategy-faker` extra and pandera's own entry point.
7. **Backend coverage (phased, §6.5)** — polars → ibis/datafusion → pyspark
   `*_strategies.py` bridges reusing the value core; add `strategy()`/`example()`
   (and distributed seeding for pyspark) to each backend's schema/container.
8. **Docs** — new section in `data_synthesis_strategies.md`: opt-in usage,
   `semantic` reference, provider mapping table, per-backend notes, third-party
   contribution guide.
9. **Benchmarks/QA** — sanity that a Faker-column schema produces non-empty,
   schema-valid, seed-reproducible output across all supported backends; no
   regression in default path.

---

## 13. Testing strategy

- **Unit (default path unchanged)**: run the entire existing
  `tests/strategies/` suite; assert zero modified tests.
- **Unit (registry)**: register/unregister; entry-point discovery via a real
  `importlib.metadata` fixture; unknown-name error; re-registration idempotency.
- **Unit (Faker backend)**:
  - each `semantic` maps to a schema-valid realistic value for a representative
    dtype + constraint (property: `schema.validate(generated).passes`), using a
    fixed seed;
  - infeasible `FieldConstraints` → base declines → output still validates;
  - `isin` yields only members; `str_matches`, length, `ge/le` all hold;
  - reproducibility: `example(seed=s)` twice → identical frames.
- **Backward-compat regression**: `example(seed=s)` and `strategy(size=n)`
  with `synthesis_backend=None` produce the same bytes as before the change on
  a fixture schema (golden test).
- **Opt-in gating**: `example(synthesis_backend="faker")` without Faker
  installed raises a clear `ImportError`; no Faker import at pandera import time.

---

## 14. Third-party contribution guide

A third party adds a realistic or statistical backend without touching pandera:

```python
# my_pandera_synth.py
from pandera.api.synthesis import register_synthesis_backend
from pandera.api.synthesis.protocol import SynthesisBackend, FieldGenerator

class MyGenerator(FieldGenerator):
    def generate(self, pandera_dtype, constraints, *, semantic=None, seed=None):
        ...   # return SearchStrategy / value / iterable / None

backend = MyBackend(...)      # a SynthesisBackend with .name and .field_generator()

# Programmatic (import this module before synthesis):
register_synthesis_backend("my_synth", backend)
```

```toml
# their pyproject.toml — declarative alternative, no import required by users
[project.entry-points."pandera.synthesis_backends"]
my_synth = "my_pandera_synth:backend"
```

Users select it with `schema.example(..., synthesis_backend="my_synth")` or
`PANDERA_SYNTHESIS_BACKEND=my_synth`. Guidelines for contributors are spelled
out in the docs: always pre-check feasibility against `FieldConstraints`, return
`None` rather than producing invalid values, and honor `seed` for
reproducibility.

---

## 15. Resolved design decisions

1. **Structural protocol over ABC** — `SynthesisBackend`/`FieldGenerator` are
   `@runtime_checkable` Protocols so Mode A and Mode B backends implement only
   what they need and third parties are not forced into inheritance.
2. **Realistic only touches the base value, not the constraint compiler** —
   guarantees are preserved because the realistic base still flows through the
   existing merged-`FieldConstraints` compilation. This is the same
   "merge first, generate once" principle as `specs/optimized-strategies.md`.
3. **Explicit `semantic` opt-in over column-name inference** — inference is a
   fragile, non-deterministic heuristic; an explicit tag is predictable,
   documented, and validation-neutral. A *narrow* inference channel (unambiguous
   email regex → email provider) is provided as a documented, per-backend
   convenience.
4. **Entry points + programmatic registration both supported** — declarative for
   installed packages, imperative for notebooks/editable installs.
5. **Faker as the first backend; Mimesis as #2; SDV as the Mode B path** — Faker
   wins on adoption, maintenance, and zero-dependency footprint; Mimesis
   reuses the same `FieldGenerator` interface unchanged; SDV slots into the
   reserved `synthesize()` tier later.
6. **Default is byte-for-byte today's behavior** — new params/env/config default
   to "no backend", and the `None` branch is a single early path, minimizing
   regression surface.
7. **Engine-agnostic value core before per-backend bridges** — value
   compilation (dtype + merged `FieldConstraints` → scalars) does not depend on
   the target dataframe library, so it is extracted once (§6.3) and reused by
   pandas, polars, pyspark, ibis, and datafusion. Each new backend is then just
   thin container-assembly glue plus a dtype bridge, and realistic
   `FieldGenerator`s are reused unchanged across all of them.

---

## 16. Summary

Pandera's synthesis is valuable because it *guarantees* schema validity; its
weakness is that hypothesis alone produces unrealistic data. This spec makes the
value-generating layer **pluggable and opt-in**: a thin, structural
`synthesis_backend` interface sits in front of the leaf where base values are
drawn, and the top candidate — **Faker** — ships as the first production
backend behind a clean `semantic`-driven contract. The default is unchanged.
It also **brings data synthesis to every backend that lacks it today** —
polars, pyspark, ibis, and datafusion gain first-class `strategy()`/`example()`
methods by extracting an engine-agnostic value core that all backends share,
so the realistic-enrichment and constraint-merging machinery is built once and
reused everywhere. Third-party and future statistical (SDV) backends can be
contributed through the same registry and packaging entry points, keeping
pandera lightweight while letting its synthetic data finally *look real* —
on every supported dataframe backend.
