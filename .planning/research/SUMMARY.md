# Project Research Summary

**Project:** Narwhals-backed dataframe validation backend for pandera
**Domain:** Validation library infrastructure — new backend using Narwhals as a cross-library abstraction layer
**Researched:** 2026-03-09
**Confidence:** HIGH

## Executive Summary

This project adds a new validation backend to pandera that uses Narwhals 2.15.0 as a unified dataframe API layer, enabling a single implementation path to validate Polars, Ibis, pandas, PyArrow, DuckDB, and PySpark frames. Rather than introducing a new user-facing `pandera.narwhals` API, the backend registers transparently against existing per-library schema classes (`pandera.polars.DataFrameSchema`, etc.) and is activated via an explicit opt-in (`pandera.use_backend("narwhals")`). Users continue writing schemas exactly as before; the narwhals backend replaces the execution engine underneath them.

The recommended approach is to build the backend incrementally in 9 ordered steps: utility types and narwhals engine first (both are prerequisites for everything else), then check backend and builtin checks, then column and container backends, and finally registration and integration hooks. The Polars backend serves as the primary reference implementation; every Polars API call has a direct Narwhals equivalent, making the port straightforward for the main path. The highest-complexity piece is the dtype engine: a `narwhals_engine.py` must map Narwhals dtype objects (`nw.Int64`, `nw.String`, etc.) to pandera `DataType` subclasses using the existing Engine metaclass pattern, and this must be resolved before any coercion or dtype-checking code can be written.

The three critical risks are: (1) attempting to use `map_batches` / `map_elements` for element-wise checks on lazy SQL backends (Ibis, PySpark, DuckDB) — not implemented in narwhals for those backends and must be handled with explicit library branching; (2) backend registration conflicts if narwhals registration is not kept strictly opt-in, which would silently override native backends and break existing users; and (3) Narwhals wrapper objects leaking into `SchemaError` failure cases, producing opaque error messages. All three have clear mitigations that must be designed in the foundation phase before implementation begins.

## Key Findings

### Recommended Stack

The only new dependency is `narwhals>=2.15.0`. It has zero required dependencies of its own, making it safe to add as a pandera optional extra (`pip install pandera[narwhals]`). All other libraries pandera already requires (polars >=0.20, pandas >=2.1.1, ibis >=9.0.0, pyarrow >=13, PySpark >=3.5.0) are compatible with narwhals 2.15.0 with no version conflicts.

Pandera must import from `narwhals.stable.v1` (not `narwhals` directly) to insulate itself from breaking API changes between narwhals releases. The `stable.v1` namespace provides semantic-versioning-equivalent stability guarantees and re-exports the full public API surface.

**Core technologies:**
- `narwhals.stable.v1` (2.15.0): Unified dataframe API layer — abstracts ~80% of common validation operations across all target backends via a Polars-like expression API
- `narwhals_engine.py` (new): Dtype bridge mapping `nw.DType` objects to pandera `DataType` subclasses — must follow the existing Engine metaclass pattern from `polars_engine.py`
- `pyarrow` (already in pandera): Default materialization target for Ibis/DuckDB/PySpark lazy frames via `LazyFrame.collect()`

**Critical API limitations confirmed from narwhals 2.15.0 source:**
- `map_batches` is `not_implemented()` on `LazyExpr` base — unavailable for Ibis, PySpark, DuckDB
- Ibis columns are `IbisInterchangeSeries` — raises `NotImplementedError` on all methods except `.dtype`; all Ibis operations must use Expr chains on the LazyFrame
- `over()` window expressions are `not_implemented()` for Ibis; group-by checks must use `group_by(...).agg()` instead
- `LazyFrame.collect()` defaults to PyArrow for Ibis and PySpark; for Polars lazy it collects to Polars

### Expected Features

**Must have (table stakes):**
- Column presence check — `collect_schema().names()` membership test
- Column dtype check — requires narwhals_engine.py dtype resolution (highest complexity gate)
- Nullable check — `nw.col(name).is_null()` plus `is_nan()` for float columns
- Schema-level and column-level unique checks — requires `.collect()` then `is_duplicated()`; cannot be lazy
- Strict/filter column mode — `frame.drop(cols)` with column list from `collect_schema()`
- All 14 builtin checks — all map directly to narwhals Expr equivalents (equal_to, not_equal_to, comparisons, is_in, notin, str_*, unique_values_eq); uniformly low-to-medium complexity
- Lazy validation mode (`lazy=True` collects all errors) — reuse existing `ErrorHandler`; no new infrastructure
- Coerce dtype — `nw.col(name).cast(nw_dtype)` inside `with_columns`; needs narwhals_engine.py
- Backend registration — follows `pandera/backends/polars/register.py` pattern with direct `BACKEND_REGISTRY` writes (not `register_backend()`) to override existing entries

**Should have (differentiators):**
- Lazy-mode pandas validation — pandas currently validates eagerly only; narwhals wraps it in a lazy graph; unblocks full error collection on pandas
- DuckDB and PyArrow table validation — inherited for free once core is working; just add registration entries
- Closes Ibis backend xfail gaps — current Ibis backend is missing `coerce_dtype`, column `unique`, and `set_default`; narwhals backend implements these generically

**Defer to v2+:**
- `add_missing_columns` and `set_default` — depend on narwhals_engine.py being complete; mark `NotImplementedError` in v1
- `drop_invalid_rows` — requires positional row alignment, which has no narwhals abstraction; implement with library-specific strategies after core is stable
- Hypothesis strategies and schema IO (YAML/JSON) — explicitly deferred in PROJECT.md
- Series/Index/MultiIndex validation — narwhals has no index concept; scope to DataFrameSchema + Column only
- Groupby-based checks — not implemented in Polars check backend either; keep `raise NotImplementedError`
- pandas registration — defer until Polars and Ibis are stable; pandas backend has 55+ special-case branches that narwhals cannot abstract

### Architecture Approach

The narwhals backend adds `pandera/backends/narwhals/` and `pandera/engines/narwhals_engine.py`. There is no `pandera/api/narwhals/` API layer — only two utility files (`types.py`, `utils.py`) under `pandera/api/narwhals/` for the `NarwhalsData` named tuple and helpers. The backend is activated by writing directly into `BACKEND_REGISTRY` (bypassing the first-writer-wins `register_backend()` helper) and is strictly opt-in. The original native object must be retained alongside the narwhals-wrapped frame throughout each validation call, because `NarwhalsCheckBackend` must pass native containers to user-defined checks while passing `NarwhalsData` to built-in checks.

**Major components:**
1. `pandera/api/narwhals/types.py` — `NarwhalsData` named tuple (analogous to `PolarsData`); enables Dispatcher routing to narwhals builtin checks
2. `pandera/engines/narwhals_engine.py` — Engine metaclass + `@Engine.register_dtype` entries for nw.Int64, nw.Float64, nw.String, etc.; implements `coerce()`, `check()`, `try_coerce()`
3. `pandera/backends/narwhals/base.py` — `NarwhalsSchemaBackend` with shared helpers: `subsample()`, `run_check()`, `failure_cases_metadata()`, `drop_invalid_rows()`
4. `pandera/backends/narwhals/checks.py` — `NarwhalsCheckBackend` routing built-ins to `NarwhalsData` and user checks to native containers
5. `pandera/backends/narwhals/builtin_checks.py` — 14 check functions typed on `NarwhalsData`; registered via `@register_builtin_check`
6. `pandera/backends/narwhals/components.py` — `ColumnBackend` for per-column nullable/unique/dtype/run_checks
7. `pandera/backends/narwhals/container.py` — `DataFrameSchemaBackend` wrapping validation pipeline; wraps native frame with `nw.from_native()` as first step, unwraps with `nw.to_native()` as last step
8. `pandera/backends/narwhals/register.py` — `register_narwhals_backends()` with `lru_cache` and per-library `try/except ImportError` guards

### Critical Pitfalls

1. **Treating narwhals as a complete abstraction** — narwhals covers ~80% of needed operations; the remaining 20% (error formatting, element-wise checks, `drop_invalid_rows` positional alignment) must use explicit `nw.to_native()` escape hatches at precisely defined boundaries. Map every library-specific operation in existing backends against the narwhals API before writing code.

2. **Dtype system mismatch** — `nw.col.dtype` returns a narwhals dtype object, not a native one; passing it to existing engines (`polars_engine.Engine.dtype()`) causes dispatch misses. Resolve by writing `narwhals_engine.py` that accepts narwhals dtype objects directly. This is the foundation-phase prerequisite — retrofitting later is expensive.

3. **Backend registration conflicts** — `register_backend()` is first-writer-wins; narwhals activation must write directly into `BACKEND_REGISTRY` and must be strictly opt-in. Importing narwhals backend code must never side-effect native backend registrations.

4. **Element-wise checks not abstractable** — `map_batches` / `map_elements` is unavailable for Ibis/PySpark/DuckDB in narwhals. For these backends, raise `NotImplementedError` with a clear explanation. For Polars-backed frames, unwrap to native and use `map_elements` directly.

5. **Narwhals wrappers leaking into error messages** — `SchemaError.failure_cases` must always be a native frame. Establish an `_to_native()` helper and use it at every `SchemaError` construction site. Write tests asserting `type(failure_cases)` is a native frame type.

## Implications for Roadmap

Based on combined research findings, the build order is constrained by these hard dependencies: utility types must precede engine; engine must precede any check/parser; check backend and builtin checks precede column backend; column backend precedes container backend; all backends must exist before registration. This suggests 5 natural phases:

### Phase 1: Foundation — Types, Engine, and Registration Design
**Rationale:** The dtype engine is the single hardest architectural decision (marked "Pending" in PROJECT.md) and blocks every coercion and dtype-check implementation. Resolving it first removes the most expensive retrofit risk. Registration design must also happen here before any backend code is written, to avoid silent conflicts with native backends.
**Delivers:** `pandera/api/narwhals/types.py` (NarwhalsData), `pandera/engines/narwhals_engine.py` (Engine metaclass + core dtypes), registration activation mechanism design, Narwhals API gap inventory
**Addresses:** Column dtype check (prerequisite), coerce=True (prerequisite), backend registration safety
**Avoids:** Dtype mismatch pitfall, backend registration conflict pitfall

### Phase 2: Check Backend and Builtin Checks
**Rationale:** All 14 builtin checks map directly to narwhals Expr equivalents — low risk, high coverage. Building the check dispatch chain first gives an end-to-end test harness for column validation. The `NarwhalsCheckBackend` routing logic (built-in vs. user-defined) is also critical to get right before column and container backends depend on it.
**Delivers:** `NarwhalsCheckBackend` with built-in/user-check routing, all 14 builtin checks in `builtin_checks.py`, lazy expression return contract enforced
**Uses:** `NarwhalsData` from Phase 1, `narwhals.stable.v1` Expr API
**Implements:** Check dispatch architecture, `Dispatcher` routing pattern
**Avoids:** Element-wise check pitfall (explicit `NotImplementedError` for SQL-lazy backends), eager materialization pitfall (check functions return lazy expressions only)

### Phase 3: Column Backend
**Rationale:** Column validation (nullable, unique, dtype, run_checks) is self-contained and can be tested against a single column before the full container pipeline exists. Building and verifying this layer before the container ensures dtype checking and nullable logic are correct before they're buried in a larger call chain.
**Delivers:** `ColumnBackend` with `check_nullable` (with float NaN handling), `check_unique` (with collect-first pattern), `check_dtype` (via narwhals_engine), `run_checks` (via NarwhalsCheckBackend)
**Uses:** narwhals_engine from Phase 1, NarwhalsCheckBackend from Phase 2
**Avoids:** Silent nullable gaps (float NaN detection), dtype dispatch failures

### Phase 4: Container Backend and Core Validation Pipeline
**Rationale:** The full `DataFrameSchemaBackend` validation pipeline depends on all prior components. This phase produces the first end-to-end `schema.validate(df)` call. Registration for Polars runs here as the first concrete target, enabling the test suite to be run.
**Delivers:** `DataFrameSchemaBackend` with full pipeline (collect_column_info, strict/filter columns, coerce, presence check, unique check, run_schema_component_checks, run_checks), `register.py` for Polars, end-to-end Polars validation green
**Implements:** DataFrameSchemaBackend, native-frame wrap/unwrap boundaries, `_to_native()` at all error construction sites
**Avoids:** Narwhals wrappers leaking into error messages, eager materialization in lazy path

### Phase 5: Ibis Registration and Lazy Integration
**Rationale:** Ibis is the second target and exercises the SQL-lazy code path that Polars lazy does not. Ibis registration closes the known xfail gaps in the existing Ibis backend. This phase surfaces any remaining narwhals/Ibis-specific issues (no Series API, no window expressions, collect defaults to PyArrow) against the complete backend.
**Delivers:** Ibis registration in `register.py`, Ibis-specific edge cases handled (element-wise NotImplementedError, group-by via agg not window expressions), `drop_invalid_rows` with library-specific strategy, full xfail gap closure
**Uses:** Complete backend from Phase 4
**Avoids:** `drop_invalid_rows` positional join pitfall, Ibis collect semantics pitfall

### Phase Ordering Rationale

- Phases 1-2 are strictly ordered by dependency: types before engine, engine before checks
- Phase 3 (Column) before Phase 4 (Container) because container calls column backends; building bottom-up reduces debugging complexity
- Phase 5 (Ibis) after Phase 4 (Polars) because Ibis exercises more constrained paths; Polars-green gives a stable baseline to diff against
- pandas registration is deliberately not a phase — it is deferred until Phase 5 is stable; the pandas backend's special-case complexity warrants a separate milestone

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1 (narwhals_engine.py):** Dtype equivalence mapping from narwhals dtypes to pandera DataType subclasses needs comprehensive coverage; the polars_engine.py pattern is well-understood but narwhals dtype names differ from polars dtype names in edge cases (e.g., Datetime time_unit/time_zone parameters)
- **Phase 5 (drop_invalid_rows for Ibis):** Positional join strategy vs. synthetic row-number column involves Ibis-specific SQL backend capability detection; the existing `POSITIONAL_JOIN_BACKENDS` logic needs to be ported or replaced with a narwhals-compatible equivalent

Phases with standard patterns (skip research-phase):
- **Phase 2 (builtin checks):** All 14 checks have direct narwhals Expr equivalents; the PolarsCheckBackend pattern is a direct template
- **Phase 3 (Column backend):** Mirrors `pandera/backends/polars/components.py` with straightforward narwhals substitutions
- **Phase 4 (Container backend, Polars registration):** Direct port of `pandera/backends/polars/container.py`; Polars is the best-supported narwhals backend

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | narwhals 2.15.0 inspected directly from installed source; all API findings are from code, not docs |
| Features | HIGH | Based on direct analysis of existing polars and ibis backend source + narwhals Expr API coverage |
| Architecture | HIGH | Grounded in existing pandera architecture patterns (BACKEND_REGISTRY, Engine metaclass, Dispatcher) confirmed from source |
| Pitfalls | HIGH | Based on direct codebase analysis of both existing backends + confirmed narwhals limitations from source |

**Overall confidence:** HIGH

### Gaps to Address

- **Narwhals datetime dtype parameters:** `nw.Datetime` takes `time_unit` and `time_zone` parameters; the narwhals_engine.py dtype registration needs to handle parameterized dtypes correctly, mirroring the polars_engine.py `Datetime` handling. Verify during Phase 1.
- **`unique_values_eq` on lazy frames:** This check forces materialization. The narwhals backend must collect before calling `Series.unique()`. Confirm this does not interact badly with the lazy validation mode (`lazy=True`) error handler. Verify during Phase 2.
- **Narwhals stable.v2 readiness:** `stable.v2` exists in 2.15.0 but stability guarantees are not yet announced. Monitor narwhals release notes; migrate from `stable.v1` to `stable.v2` only when explicitly stabilized.
- **PySpark floor version:** narwhals 2.15.0 requires PySpark >=3.5.0; pandera currently supports PySpark >=3.2.0. The narwhals backend raises the floor for the narwhals code path only — document this clearly in the optional extra notes.

## Sources

### Primary (HIGH confidence)
- `narwhals` 2.15.0 source code (`.pixi/envs/default/lib/python3.12/site-packages/narwhals/`) — full API surface, dtype hierarchy, backend limitations
- `pandera/backends/polars/` source — reference implementation for all backend patterns
- `pandera/backends/ibis/` source — secondary reference, xfail gaps, Ibis-specific strategies
- `pandera/engines/polars_engine.py` — Engine metaclass pattern; direct template for narwhals_engine.py
- `pandera/api/base/schema.py` — BACKEND_REGISTRY semantics, first-writer-wins behavior confirmed
- `pandera/api/function_dispatch.py` — Dispatcher dispatch on `type(args[0])` confirmed

### Secondary (MEDIUM confidence)
- `.planning/codebase/CONCERNS.md` — known ibis backend NotImplementedErrors (coerce, unique, set_default)
- `.planning/codebase/ARCHITECTURE.md` — backend/engine separation, BACKEND_REGISTRY pattern
- `.planning/PROJECT.md` — explicit "Key Decisions" section; dtype engine decision marked as pending

---
*Research completed: 2026-03-09*
*Ready for roadmap: yes*
