# Phase 6: Eliminate Unnecessary Materialization - Context

**Gathered:** 2026-03-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Enforce a single principle throughout the narwhals backend: `.collect()` / `.execute()` is
called *only* to evaluate the scalar boolean "did the check pass?" — and everything else
(`failure_cases`, `check_output`) stays in the user's original type. This phase covers:

- Collapsing the dead `_is_ibis_result` bifurcation in `run_check()` (dead after Phase 5)
- Removing `fc.collect()` and `_materialize(check_output)` from `run_check()`
- Redesigning `failure_cases_metadata()` to be backend-agnostic (no forced polars conversion)
- Fixing `subsample()` materializing before `.head()` / `.tail()`
- Fixing `check_nullable()` materializing the whole frame to evaluate a scalar `.any()`

No new user-facing capabilities — purely architectural enforcement of the lazy-first principle.

</domain>

<decisions>
## Implementation Decisions

### CoreCheckResult.failure_cases type (internal)
- `CoreCheckResult.failure_cases` is **not user-facing** — it is internal backend state
- Must hold narwhals wrappers (`nw.LazyFrame` for polars lazy, `nw.DataFrame` for ibis)
- `run_check()` removes `fc.collect()` — failure_cases stays lazy in CoreCheckResult
- `run_check()` removes `_materialize(check_output)` and `_to_native()` — check_output stays lazy
- `run_check()` only materializes once: `select(nw.col(CHECK_OUTPUT_KEY).all())` to evaluate the scalar passed bool
- `_is_ibis_result` guard is fully removed — dead code after Phase 5's uniform expression protocol

### SchemaError.failure_cases (single-error, non-lazy path)
- Type: **native** (unwrap from narwhals before storing in SchemaError)
- Content: **raw failing rows only** — no enrichment columns (consistent with polars backend)
- Users call `.collect()` / `.execute()` themselves if they want materialized data
- Consistent with polars backend behavior; replaces the ibis backend's forced `.to_pandas()` conversion

### SchemaErrors.failure_cases (lazy mode, aggregated) — failure_cases_metadata() redesign
- **In scope for Phase 6** — current polars hardcoding violates lazy-first principle for ibis inputs
- Type: **native** (pl.LazyFrame for polars lazy, ibis.Table for ibis, pl.DataFrame for polars eager)
- Content: **enriched with metadata columns** — `failure_case`, `schema_context`, `column`, `check`, `check_number`, `index` — consistent with existing polars and pandas backends
- `failure_cases_metadata()` redesigned to build enriched frame using narwhals / native backend ops instead of always converting to `pl.DataFrame` via polars
- **Row index**: included only for eager inputs (`pl.DataFrame`, `pd.DataFrame`) where row order is well-defined; `None` for lazy/SQL backends (`pl.LazyFrame`, `ibis.Table`) — forcing materialization just for index is inconsistent with the lazy-first principle, and SQL backends don't guarantee row ordering anyway

### check_nullable() evaluation
- Fix: only materialize the scalar boolean — `_materialize(combined_lf.select(nw.col(CHECK_OUTPUT_KEY).any()))` — one row, not the full frame
- `failure_cases` stays lazy (`combined_lf.filter(nw.col(CHECK_OUTPUT_KEY)).select(col)`) — narwhals wrapper in CoreCheckResult
- `check_output` stays lazy (the wide `combined_lf` with CHECK_OUTPUT_KEY column)
- Matches the run_check lazy-first pattern

### subsample() lazy strategy
- `head=`: use `nw.LazyFrame.head(n)` directly — fully lazy for all backends, no materialization
- `tail=` on polars: use `nw.LazyFrame.tail(n)` directly — lazy
- `tail=` on SQL-lazy backends (ibis/DuckDB): raise `NotImplementedError` — SQL has no native TAIL without forced ordering; consistent with element_wise checks raising NotImplementedError for ibis
- Both head + tail: concatenate lazily via `nw.concat([check_obj.head(head), check_obj.tail(tail)]).unique()`
- No full-frame materialization in `subsample()`

### No Native-Type Branching (Hard Constraint)
- The implementation must NOT introduce new `isinstance(x, ibis.Table)` / `isinstance(x, pl.LazyFrame)` / `isinstance(x, pl.DataFrame)` branches scattered through the backend logic
- The narwhals abstraction (`nw.LazyFrame`, `nw.DataFrame`, `nw.to_native()`, `_materialize()`) should handle the unified path throughout
- Native-type handling is acceptable **only at the boundaries** — e.g., when constructing the final `SchemaError.failure_cases` / `SchemaErrors.failure_cases` to hand back to the user, or when a backend-specific API is truly unavoidable (and must be isolated in a helper)
- The goal of this phase is to *reduce* native-type branching (remove `_is_ibis_result`), not add new forms of it

### Claude's Discretion
- Whether `_materialize` helper can be simplified or removed after these changes
- Exact narwhals operations used to build the enriched metadata frame in `failure_cases_metadata()` for ibis (e.g., `mutate()` for literal columns)
- Whether `check_unique()` and `check_dtype()` need similar lazy fixes (investigate during planning)
- How to detect whether the input is eager vs lazy for the row-index decision in `failure_cases_metadata()`

</decisions>

<specifics>
## Specific Ideas

- `run_check()` final shape: one `_materialize(check_result.check_passed)` call for the scalar bool, everything else returned as narwhals wrappers, `_is_ibis_result` block deleted entirely
- `failure_cases_metadata()` for ibis: attach literal metadata columns via `table.mutate(schema_context=ibis.literal(...), ...)` — avoids `.execute()` and keeps the result as ibis.Table
- `subsample()` for SQL-lazy tail: error message should match the style of the element_wise NotImplementedError already in checks.py

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_materialize()` in `pandera/api/narwhals/utils.py`: handles both `nw.LazyFrame` (collect) and SQL-lazy `nw.DataFrame` (execute) — still needed for the scalar bool evaluation in run_check and check_nullable
- `_to_native()` in `pandera/api/narwhals/utils.py`: used to unwrap narwhals → native; will be used at SchemaError creation point instead of inside run_check
- `nw.LazyFrame.head(n)` / `nw.LazyFrame.tail(n)`: both work lazily in narwhals for polars; tail not supported for ibis

### Established Patterns
- Phase 4 locked: `failure_cases_metadata()` is the designated materialization point for SchemaErrors — this phase enforces that contract
- Phase 5 locked: `apply()` returns uniform narwhals frames — `_is_ibis_result` guard checking for `ir.BooleanScalar` is dead code
- `element_wise` checks already raise `NotImplementedError` for SQL-lazy backends — same pattern for `tail=` in `subsample()`

### Integration Points
- `pandera/backends/narwhals/base.py`: `run_check()`, `failure_cases_metadata()`, `subsample()`
- `pandera/backends/narwhals/components.py`: `check_nullable()`
- `pandera/backends/narwhals/container.py`: where `SchemaError` is raised directly — unwrap narwhals → native here
- `pandera/backends/narwhals/components.py` line ~86: `SchemaError` raised from `error_handler.schema_errors[0]` in non-lazy path — unwrap failure_cases to native here

</code_context>

<deferred>
## Deferred Ideas

- Making `postprocess_bool_output` produce the user's backend frame instead of falling back to polars for ibis — currently out of scope
- Redesigning `check_unique()` for full laziness (currently calls `_materialize(dup_values)`) — investigate during planning; may be in scope
- Whether `failure_cases_metadata()` enrichment should also apply to `SchemaError.failure_cases` (single-error path) — decided against for now (raw rows only), but worth revisiting if users find it confusing

</deferred>

---

*Phase: 06-eliminate-unnecessary-materialization-lazy-first-failure-cases-and-check-output*
*Context gathered: 2026-03-23*
