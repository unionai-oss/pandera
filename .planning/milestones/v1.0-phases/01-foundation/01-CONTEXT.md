# Phase 1: Foundation - Context

**Gathered:** 2026-03-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Create the scaffolding that all subsequent phases build on: the `NarwhalsData` named tuple, the `_to_native()` utility helper, and the narwhals dtype engine with coercion support. No end-to-end validation runs in this phase — it delivers the foundation types and engine only.

Covered requirements: INFRA-01, INFRA-02, INFRA-03, ENGINE-01, ENGINE-02, ENGINE-03.

</domain>

<decisions>
## Implementation Decisions

### NarwhalsData type
- `NarwhalsData` is a `NamedTuple` with `frame: nw.LazyFrame` and `key: str = "*"` — mirrors `PolarsData` exactly
- Always-lazy: incoming frames are converted to `nw.LazyFrame` on entry; uniform handling across backends
- `"*"` sentinel means "whole frame"; named column strings mean column-level checks — same semantics as `PolarsData.key`
- `NarwhalsCheckResult` named tuple is also defined in `types.py` with `nw.LazyFrame` fields, parallel to `PolarsCheckResult`
- `NarwhalsData` is the dispatch key — `Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend)` will route on this type in Phase 2

### Parameterized dtypes (Datetime, Duration)
- Register base classes only: `nw.Datetime` (unparameterized), `nw.Duration` (unparameterized)
- No pre-registered variants (e.g., `nw.Datetime("us", "UTC")`) — combinatorial explosion for time_zone makes this impractical
- `coerce()` receives the full pandera `DataType` instance (which carries user-specified params) and passes it to `nw.col(name).cast(nw_dtype)` — narwhals handles parameterized casts natively
- Same approach for both `nw.Datetime` and `nw.Duration`

### Coercion error mapping
- `COERCION_ERRORS = (TypeError, nw.exceptions.InvalidOperationError, nw.exceptions.ComputeError)` — import from `narwhals.stable.v1.exceptions`
- Narwhals wraps backend-native exceptions into its own types; catching at the narwhals level keeps the engine backend-agnostic
- `try_coerce()` must call `.collect()` to trigger failures from lazy cast operations (narwhals has no `strict=False` on `cast()`)
- Per-row failure case identification: Claude's discretion based on implementation complexity

### List and Struct dtypes
- Register `nw.List` and `nw.Struct` as unparameterized base classes — type-checking matches any List or any Struct
- Inner type validation (e.g., `List(Int64)` vs `List(String)`) is not performed in Phase 1 — deferred
- `coerce()` attempts cast with inner types when provided (narwhals supports `nw.col(name).cast(nw.List(nw.Int64))`); same code path as scalar dtypes — no special handling needed

### Claude's Discretion
- Per-row failure case identification in `try_coerce()` (whether to do a second pass to find which rows failed, as polars engine does)
- Exact structure of `narwhals_coerce_failure_cases()` helper if implemented
- Whether additional type aliases beyond `NarwhalsData` and `NarwhalsCheckResult` are needed in `types.py`

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `pandera/api/polars/types.py` — direct template for `pandera/api/narwhals/types.py`; copy structure, substitute `pl.LazyFrame` → `nw.LazyFrame`
- `pandera/api/polars/utils.py` — template for `pandera/api/narwhals/utils.py`; `_to_native()` wraps `nw.to_native()`
- `pandera/engines/polars_engine.py` — template for `narwhals_engine.py`; `COERCION_ERRORS`, `coerce()`, `try_coerce()` patterns all reusable
- `pandera/backends/polars/register.py` — shows `lru_cache` + import-inside-function pattern; Phase 1 does not implement registration (Phase 4) but should follow this structure when needed

### Established Patterns
- Engine metaclass: `pandera.engines.engine.Engine` is the base; `narwhals_engine.py` uses `@Engine.register_dtype` decorators per dtype — exactly as `polars_engine.py` does
- Dispatcher routing: `Check.register_backend(type, Backend)` dispatches on the type of the first argument to a check function — `NarwhalsData` must be a distinct named type for this to work
- `lru_cache` on register functions: prevents duplicate registrations across imports
- `narwhals.stable.v1` import: all narwhals code imports from `narwhals.stable.v1` (aliased as `nw`) — not bare `narwhals`

### Integration Points
- `pandera/api/narwhals/types.py` — new file; consumed by check backend (Phase 2) and column backend (Phase 3)
- `pandera/api/narwhals/utils.py` — new file; `_to_native()` used at every `SchemaError` construction site starting Phase 3/4
- `pandera/engines/narwhals_engine.py` — new file; consumed by column backend `check_dtype()` (Phase 3)
- `pyproject.toml` — `narwhals>=2.15.0` added as optional extra: `pandera[narwhals]`; narwhals 2.15.0 already installed in dev env

</code_context>

<specifics>
## Specific Ideas

- `NarwhalsData` field should be named `frame` (not `lazyframe`) — makes it clearer it's a narwhals frame, not a polars-specific one
- All narwhals imports use `import narwhals.stable.v1 as nw` consistently throughout the new modules
- narwhals 2.15.0 already installed in dev environment; no install step needed during development

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-foundation*
*Context gathered: 2026-03-09*
