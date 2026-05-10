# Phase 3: Column Backend - Context

**Gathered:** 2026-03-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement `ColumnBackend` in `pandera/backends/narwhals/components.py` with the 4 core per-column validation methods: `check_nullable`, `check_unique`, `check_dtype`, and `run_checks`. Also includes `run_checks_and_handle_errors` (the orchestrating loop) and a `NarwhalsSchemaBackend` base class in `base.py`. No `coerce_dtype`, `set_default`, or full `validate()` pipeline in this phase — those belong in Phase 4 (container backend).

Covered requirements: COLUMN-01, COLUMN-02.

</domain>

<decisions>
## Implementation Decisions

### Scope of components.py
- Implement 4 core check methods only: `check_nullable`, `check_unique`, `check_dtype`, `run_checks`
- Also include `run_checks_and_handle_errors` — the orchestrating loop that calls all 4 checks; needed to make the column validation unit testable in isolation
- Do NOT include `coerce_dtype`, `set_default`, or `validate()` — these belong in Phase 4
- Each of `check_nullable`, `check_unique`, `check_dtype` decorated with `@validate_scope` matching polars backend: DATA/DATA/SCHEMA respectively
- Tests in new file: `tests/backends/narwhals/test_components.py`

### Base class structure
- Create `pandera/backends/narwhals/base.py` with `NarwhalsSchemaBackend` now (Phase 4 will expand it)
- `ColumnBackend` inherits from `NarwhalsSchemaBackend`
- Phase 3 adds to `NarwhalsSchemaBackend`: `subsample()`, `run_check()`, `is_float_dtype()` helpers
- Phase 4 will add: `failure_cases_metadata()`, `drop_invalid_rows()`
- `subsample()`: raises `NotImplementedError` immediately for `sample=` param (narwhals backend always uses LazyFrame; `sample=` is never supported — use `head=` or `tail=` instead)
- `_materialize()` stays in `checks.py`; `ColumnBackend` imports it from there — no code movement needed

### Float NaN detection in check_nullable
- Use narwhals dtype `.is_float()` method: `check_obj.collect_schema()[col_name].is_float()`
- Add `is_float_dtype(check_obj, col_name)` helper to `NarwhalsSchemaBackend` using this approach
- For non-float columns: apply `is_null()` only — NaN is a float concept
- For float columns: apply `is_null() | is_nan()` — matches polars backend behavior and COLUMN-01 requirement

### check_dtype failure cases format
- Use `str(collect_schema()[col])` — narwhals-style dtype string (e.g. `"Int64"`)
- Matches Polars users exactly; minor capitalization difference for Ibis/pandas users (`"int64"` vs `"Int64"`) — **known inconsistency, intentional**: getting truly native dtype strings would require backend-specific materialization that defeats the narwhals unification purpose
- Short-circuit when `schema.dtype is None`: return `CoreCheckResult(passed=True)` immediately — matches all existing backends

### check_unique collect strategy
- Collect the column of interest before calling `is_duplicated()` (narwhals `is_duplicated()` requires eager DataFrame)
- Pattern: `check_obj.select(schema.selector).collect().select(nw.col("*").is_duplicated())`
- Document the collect-first pattern in a comment per COLUMN-02 requirement

### Claude's Discretion
- Exact failure case construction for check_nullable (how to combine original frame + check output to identify failing rows)
- Exact failure case construction for check_unique
- Internal helper naming and structure within NarwhalsSchemaBackend
- Whether to define a module-level `is_float_dtype()` function or a method on `NarwhalsSchemaBackend`

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `pandera/backends/polars/components.py` — direct template for `components.py`; same 4 core checks, same `run_checks_and_handle_errors` pattern
- `pandera/backends/polars/base.py` — template for `NarwhalsSchemaBackend`; `subsample()`, `run_check()`, `is_float_dtype()` all have direct narwhals equivalents
- `pandera/backends/narwhals/checks.py` — `_materialize()` helper already defined; `NarwhalsCheckBackend` pattern shows how narwhals frames are handled
- `pandera/api/narwhals/types.py` — `NarwhalsData(frame, key)` already defined; used as the dispatch container

### Established Patterns
- `collect_schema()` is the narwhals equivalent of `get_lazyframe_schema()` — lazy-safe, no materialization
- `nw.col().is_nan()` only valid on float columns — always guard with `is_float_dtype()` check first
- `_to_native()` from `pandera/api/narwhals/utils.py` must be called on any frame appearing in `SchemaError.failure_cases`
- `@validate_scope` decorator from `pandera.validation_depth` controls DATA vs SCHEMA depth gating

### Integration Points
- `pandera/backends/narwhals/base.py` — new file; consumed by `components.py` and eventually `container.py` (Phase 4)
- `pandera/backends/narwhals/components.py` — new file; consumed by container pipeline (Phase 4) and test suite
- `tests/backends/narwhals/test_components.py` — new file; parameterized against Polars and Ibis (same fixture pattern as Phase 2 test harness)

</code_context>

<specifics>
## Specific Ideas

- The wide-frame approach (keeping data + check output columns together to avoid horizontal concat) is an interesting future optimization — noted for exploration after Phase 4. Would eliminate the materialize-before-concat pattern but requires rethinking how `apply()` returns results.
- `sample=` unsupported in `subsample()` is a consequence of the always-lazy design from Phase 1 — if pandas support ever needs `sample=`, revisit the always-lazy contract first.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 03-column-backend*
*Context gathered: 2026-03-13*
