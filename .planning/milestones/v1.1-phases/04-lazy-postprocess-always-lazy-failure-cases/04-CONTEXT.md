# Phase 4: Lazy postprocess — always-lazy failure_cases - Context

**Gathered:** 2026-03-22
**Status:** Ready for planning
**Source:** Brainstorm session (conversation context)

<domain>
## Phase Boundary

Make `postprocess_lazyframe_output` fully lazy for all backends (Polars LazyFrame and ibis). Fix `failure_cases` being `pyarrow.Table` for ibis builtin check failures by never materializing the check object during postprocessing.

</domain>

<decisions>
## Implementation Decisions

### Root Cause
The bug: ibis builtin check `failure_cases` is `pyarrow.Table` instead of a lazy ibis type. Root cause: `postprocess_lazyframe_output` materializes `check_obj.frame` via `_materialize()` (which calls `ibis.Table.execute()` → pandas → `nw.from_native(pd_df)`), then `_to_native()` on the resulting pandas-backed narwhals frame returns `pyarrow.Table`.

### Architecture: Wide Table Approach (Locked)
`apply()` should attach `CHECK_OUTPUT_KEY` column to the **full frame** via `with_columns`, returning the same lazy type as the input (not just the boolean column alone). This means:
- For Polars: returns `nw.LazyFrame` with `CHECK_OUTPUT_KEY` column appended
- For ibis: returns `nw.DataFrame` wrapping `ibis.Table` with `CHECK_OUTPUT_KEY` appended

Narwhals has a **dedicated ibis backend** (`narwhals._ibis`), NOT just the interchange protocol. `with_columns`, `filter`, and `select(col.all())` all work lazily on ibis-backed narwhals frames (verified experimentally).

### postprocess_lazyframe_output: No Materialization (Locked)
`postprocess_lazyframe_output` must NOT call `_materialize(check_obj.frame)`. Instead it works lazily:
```python
# check_output IS already frame + CHECK_OUTPUT_KEY column
passed = check_output.select(nw.col(CHECK_OUTPUT_KEY).all())   # lazy scalar agg
failure_cases = check_output.filter(~nw.col(CHECK_OUTPUT_KEY)) # lazy filter
if check_obj.key != "*":
    failure_cases = failure_cases.select(check_obj.key)
if self.check.n_failure_cases is not None:
    failure_cases = failure_cases.head(self.check.n_failure_cases)
```

### Materialization Point: run_check Only (Locked)
Materialization (calling `_materialize`) only happens in `run_check` when evaluating the scalar `passed` boolean and when extracting `failure_cases`. This is the correct boundary — the check backend stays lazy, the schema backend materializes when it needs concrete values.

### failure_cases Type: Always nw.DataFrame (Locked)
`failure_cases` in `CoreCheckResult` must always be a **narwhals frame** (`nw.DataFrame`) — never unwrapped to a backend-native type (`pl.DataFrame`, `ibis.Table`, `pyarrow.Table`). This applies to both Polars and ibis paths. `_to_native` must NOT be called on `failure_cases` in `run_check`.

Rationale: `_to_native` in `run_check`'s narwhals path is the root cause of the pyarrow.Table bug for ibis. Keeping `failure_cases` as `nw.DataFrame` throughout eliminates the need for backend-specific type detection in `run_check` and keeps the narwhals abstraction intact.

`failure_cases_metadata` is the correct place to unwrap to native types when needed for backend-specific index computation.

### _is_ibis_result Guard: Do NOT Extend (Locked)
Do NOT extend the `_is_ibis_result` guard in `run_check` to detect narwhals-wrapped ibis frames. After this phase, ibis builtin checks produce narwhals frames that flow through the regular narwhals path in `run_check` — no special ibis branching needed for builtin checks.

The `_is_ibis_result` guard only remains for `native=True` custom checks that return raw `ir.BooleanScalar`/`ibis.Table` outside the narwhals type system.

### Return Type of __call__ (Decision)
`NarwhalsCheckBackend.__call__` returns whatever type the input frame is — `nw.LazyFrame` for Polars, `nw.DataFrame` wrapping ibis for ibis. No type normalization. Callers already handle both.

### Do NOT Scale Back
Do not "fix" the bug with a simple pyarrow→polars conversion after `_to_native`. The real fix is the architectural change above.

### Claude's Discretion
- Whether `_materialize` helper can be simplified or removed after the change
- How to handle `ignore_na` in the new lazy postprocess path
- Whether `postprocess_bool_output` needs changes for consistency
- Test structure for the new behavior

</decisions>

<specifics>
## Specific Ideas

- `apply()` currently returns just the boolean output column (renamed to `CHECK_OUTPUT_KEY`). Change it to return `check_obj.frame.with_columns(bool_output)` — the full frame plus the check column.
- For the `element_wise` path in `apply()`: it already does `frame.with_columns(...)` — check if it already returns the wide result or just the selected column, and align.
- `_materialize` in `checks.py` may still be needed by `run_check` in `base.py` (via the module-level `_materialize` wrapper). Keep it for now.
- The `_is_ibis_result` guard in `run_check` (base.py) that handles the custom ibis check path — check if it's still needed or can be unified with the narwhals path after this change.

</specifics>

<deferred>
## Deferred Ideas

- Removing `_is_ibis_result` guard entirely (custom ibis checks via `native=True` may still produce different result shapes — investigate separately)
- Making `postprocess_bool_output` lazy (currently produces a Polars LazyFrame for bool scalar results even for ibis — separate concern)
- `subsample()` in `base.py` also calls `_materialize` — out of scope for this phase

</deferred>

---

*Phase: 04-lazy-postprocess-always-lazy-failure-cases*
*Context gathered: 2026-03-22 via brainstorm session*
