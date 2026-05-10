# Phase 2: Remaining PR Review Fixes - Context

**Gathered:** 2026-03-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Address the four remaining unresolved PR #2223 review comments in `pandera/backends/narwhals/checks.py` and `pandera/backends/narwhals/components.py`. No new capabilities, no user-facing changes. All fixes are architectural clean-ups to make the Narwhals backend internally consistent.

</domain>

<decisions>
## Implementation Decisions

### Horizontal concat elimination
- `postprocess_lazyframe_output` (checks.py:134): replace `_materialize` + horizontal concat with `with_columns` — add CHECK_OUTPUT_KEY directly onto `check_obj.frame`, stays fully lazy, no round-trip. Row alignment is guaranteed since check_output is derived from check_obj.frame.
- `check_nullable` (components.py:105): compute the null boolean inline — `check_obj.with_columns(null_expr.alias(CHECK_OUTPUT_KEY))` then filter — single LazyFrame op, eliminates both `_materialize` calls.
- `_materialize` should be kept as a utility for the ibis SQL-lazy path (nw.DataFrame wrapping ibis that needs `.execute()`); just fewer call sites after refactor.

### postprocess_bool_output Polars import
- Replace `import polars as pl` + `pl.LazyFrame({...})` with `nw.get_native_namespace(check_obj.frame)` + `nw.from_dict({CHECK_OUTPUT_KEY: [check_output]}, native_namespace=ns).lazy()`.
- Makes the method correct by construction for any backend, not just Polars in practice.

### Custom check Ibis delegation
- Do NOT remove the `IbisCheckBackend` delegation yet — it is a backward-compat shim for existing Ibis user checks that expect `IbisData(table, key)`, not a fundamental NarwhalsCheckBackend limitation.
- Document explicitly in code why delegation exists: user functions written for Ibis expect `IbisData`; user functions written for Polars currently receive a raw `pl.LazyFrame` (key is dropped) — neither is ideal, both are transitional.
- Add TODO pointing at the future direction: `apply()` should unwrap `NarwhalsData` to the type the check function expects (via type annotation inspection), making `IbisCheckBackend` delegation unnecessary.
- The `import ibis as _ibis` alias is intentional convention for guarded optional imports — no change needed.

### check_dtype three-pass fallback
- Drop the multi-pass (narwhals → polars native → ibis native) entirely. Use a single narwhals engine dtype pass only, matching the structure of the Polars backend.
- Rationale: if the Narwhals backend is active, `schema.dtype` should be a narwhals engine dtype. The multi-pass compensates for a schema construction mismatch that shouldn't exist.
- If polars/ibis engine dtypes don't check correctly through the narwhals engine, that is acceptable given experimental status — users saw the warning.
- Add TODO: the root fix is in schema construction — `pandera.polars`/`pandera.ibis` should produce narwhals engine dtypes when the Narwhals backend is active.

### Claude's Discretion
- Whether `passed` in `postprocess_lazyframe_output` stays as a 1-row DataFrame or becomes a plain Python bool after the with_columns refactor.
- Exact call site cleanup for `_materialize` after horizontal concat is removed.

</decisions>

<specifics>
## Specific Ideas

- The reviewer's preferred direction for horizontal concat: "we should not have lost the relation to the check_obj when computing the check result to begin with" — `with_columns` is the direct expression of this.
- For custom check delegation: the long-term contract is that checks written for the Narwhals backend accept `NarwhalsData`; delegation to `IbisCheckBackend` is explicitly a transitional shim, not permanent design.
- For `check_dtype`: the three-pass fallback is the wrong layer to fix this — schema construction is the right layer.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_materialize` (checks.py:117): keeps its role for ibis SQL-lazy `.execute()` path; call sites in postprocess/check_nullable are removed.
- `nw.get_native_namespace()`: available in `narwhals.stable.v1`, infers backend from an existing frame — use in `postprocess_bool_output`.

### Established Patterns
- `with_columns` is lazy-safe in Narwhals for both Polars and Ibis-backed frames.
- `nw.from_dict(..., native_namespace=ns)` produces a frame in the correct backend without importing the native library directly.
- Single-pass dtype check is the Polars backend pattern: `schema.dtype.check(obj_dtype)`.

### Integration Points
- `postprocess_lazyframe_output` and `postprocess_bool_output` in `NarwhalsCheckBackend` (checks.py).
- `check_nullable` and `check_dtype` in `ColumnBackend` (components.py).
- `IbisCheckBackend` delegation in `NarwhalsCheckBackend.__call__` — leave in place with improved documentation.

</code_context>

<deferred>
## Deferred Ideas

- Schema construction fix: `pandera.polars`/`pandera.ibis` producing narwhals engine dtypes when Narwhals backend is active — future phase.
- Eliminate `IbisCheckBackend` delegation by implementing unwrap-and-coerce in `apply()` — future phase.
- `_to_native` usage in `container.py:413` (column component dispatch) — left as-is with comment explanation, not in scope for this phase.

</deferred>

---

*Phase: 02-remaining-pr-review-fixes*
*Context gathered: 2026-03-22*
