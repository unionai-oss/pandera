# Phase 2: Check Backend - Context

**Gathered:** 2026-03-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement `NarwhalsCheckBackend` (check routing and dispatch), `builtin_checks.py` (all 14 builtin checks via narwhals Expr API), and the initial test harness in `tests/backends/narwhals/` parameterized against Polars and Ibis.

No column-level validation, no container-level pipeline, no registration in this phase — those are Phase 3+. This phase is check execution only.

Covered requirements: CHECKS-01, CHECKS-02, CHECKS-03, TEST-01.

</domain>

<decisions>
## Implementation Decisions

### User-defined check dispatch
- Narwhals backend is internal plumbing — users write custom checks for their native framework, not narwhals
- User-defined checks receive the **native data container** (`PolarsData` for Polars, `IbisData` for Ibis, etc.) — consistent with what they'd receive from the native backend
- Builtin checks receive `NarwhalsData` (narwhals Expr API)
- Distinction is made via **check function signature type annotation** inspection: if first-arg annotation is `NarwhalsData`, call with `NarwhalsData`; otherwise unwrap to native container
- Unwrapping happens in `apply()`: detect non-narwhals signature → call `nw.to_native(data.frame)` → wrap into native container → call check_fn
- Result from user-defined check (native frame or bool) is then handled by `postprocess()`

### element_wise checks on SQL-lazy backends
- `map_batches` approach: call `nw.col(key).map_batches(check_fn, return_dtype=nw.Boolean)` inside a try/except
- Catch `NotImplementedError` raised by narwhals for SQL-lazy backends (Ibis, DuckDB, PySpark)
- Re-raise with a clear pandera message explaining the limitation:
  `"element_wise checks are not supported on SQL-lazy backends (Ibis, DuckDB, PySpark) because row-level Python functions cannot be applied to lazy query plans. Use a vectorized check instead."`
- For Polars (non-SQL-lazy): `map_batches` works; always pass `return_dtype=nw.Boolean` — no inference needed

### Test harness structure
- Tests live in `tests/backends/narwhals/`
- Cover both Polars and Ibis from Phase 2 (validates narwhals abstraction cross-backend)
- Parameterization via **pytest fixture with `params=["polars", "ibis"]`**: a `backend_frame` (or similar) fixture provides a frame factory callable per backend — standard, DRY, easy to extend
- Element_wise + SQL-lazy `NotImplementedError` tested in Phase 2 (CHECKS-03 must be verified)

### Claude's Discretion
- Exact fixture naming and file structure within `tests/backends/narwhals/`
- Whether to use `conftest.py` for shared fixtures or inline in test files
- Internal helper for selecting a column from `NarwhalsData.frame` by key
- `postprocess` handling for user-defined check results that return native frames

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `pandera/backends/polars/checks.py` — direct template for `NarwhalsCheckBackend`; same `preprocess` / `apply` / `postprocess` / `__call__` pattern; substitute `PolarsData` → `NarwhalsData`, `pl.LazyFrame` → `nw.LazyFrame`
- `pandera/backends/polars/builtin_checks.py` — template for narwhals `builtin_checks.py`; same `@register_builtin_check` decorator with `NarwhalsData` as first-arg type annotation; narwhals Expr API is nearly 1:1 with polars Expr (`.eq`, `.ne`, `.gt`, `.ge`, `.lt`, `.le`, `.is_in`, `.is_between`, `.str.contains`, `.str.starts_with`, `.str.ends_with`, `.str.len_chars`)
- `pandera/api/narwhals/types.py` — `NarwhalsData(frame: nw.LazyFrame, key: str)` and `NarwhalsCheckResult` already defined in Phase 1
- `pandera/api/function_dispatch.py` — `Dispatcher` dispatches on type of first arg; narwhals builtin checks register by having `NarwhalsData` as first-arg annotation

### Established Patterns
- `Check.__call__(check_obj, column)` → `get_backend(check_obj)(self)` → backend `__call__` → returns `CheckResult`
- Backend registered via `Check.register_backend(native_type, NarwhalsCheckBackend)` (happens in Phase 4 registration)
- Builtin check dispatch: `Dispatcher` dispatches on `type(args[0])` — narwhals builtins registered with `NarwhalsData` as first arg, so `NarwhalsCheckBackend.apply()` passes `NarwhalsData`
- `narwhals.stable.v1` imports throughout; `data.frame` (not `data.lazyframe`) accesses the frame

### Integration Points
- `pandera/backends/narwhals/checks.py` — new file; consumed by column backend (Phase 3) via `Check.__call__`
- `pandera/backends/narwhals/builtin_checks.py` — new file; registers narwhals implementations of all 14 builtin checks into the shared `CHECK_FUNCTION_REGISTRY` Dispatcher
- `pandera/api/narwhals/types.py` — already exists; `NarwhalsData` is the dispatch key
- `tests/backends/narwhals/` — new directory; grows across phases 2-5

</code_context>

<specifics>
## Specific Ideas

- narwhals `str.contains` supports regex patterns (confirmed via testing: `nw.col('b').str.contains('^f')` works) — `str_matches` can use the same `^`-prefix anchoring approach as the polars implementation
- narwhals `str.contains` does NOT have a `literal=False` parameter like polars — `str_contains` simply uses `str.contains(pattern)` without that flag
- `map_batches` on narwhals LazyFrame requires explicit `return_dtype=nw.Boolean` — hardcode this for all element_wise check applications

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 02-check-backend*
*Context gathered: 2026-03-09*
