# Phase 7: v1.0 Tech Debt Cleanup - Context

**Gathered:** 2026-03-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix 6 tech debt items identified in the v1.0 milestone audit: 2 correctness bugs, 1 stale docstring,
5 xpassed test markers, stale ROADMAP checkboxes, and 1 pre-existing ibis API rename. All items are
clearly scoped — this phase does not introduce new functionality.

</domain>

<decisions>
## Implementation Decisions

### Dead branch in `_count_failure_cases`
- Replace the dead `isinstance(failure_cases, (nw.LazyFrame, nw.DataFrame))` branch AND the
  ibis-specific `try/import ibis/isinstance(ibis.Table)` guard with a single unified narwhals count:
  `int(nw.from_native(failure_cases, eager_only=False).lazy().select(nw.len()).collect()["len"][0])`
- Rationale: `failure_cases` is always native (pl.DataFrame or ibis.Table) at the SchemaError
  boundary (Phase 6 contract) — `nw.from_native()` wraps it uniformly without backend-specific logic
- `SchemaError.failure_cases` stays native (user-facing contract unchanged) — the `nw.from_native()`
  call is internal to `_count_failure_cases` only

### `drop_invalid_rows` check_output fix
- **Dropped from Phase 7 scope.** The whole function needs a narwhals-idiomatic rethink:
  ibis delegates to `IbisSchemaBackend`, polars uses a hardcoded `pl.DataFrame()` constructor,
  and there is no path for other backends (pandas, PySpark). Fixing the narrow
  `co[CHECK_OUTPUT_KEY]` unwrap would be patching the wrong layer.
- Track as a future phase: narwhals-idiomatic `drop_invalid_rows` covering all backends uniformly.

### xpassed test promotion
- **Promote 4 tests** by removing their `xfail` markers (they are genuinely passing):
  - `test_postprocess_lazyframe_no_materialization_ibis`
  - `test_failure_cases_metadata`
  - `test_ibis_narwhals_auto_activated`
  - `test_ibis_backend_is_narwhals`
- **Delete `test_drop_invalid_rows`** — the test uses a fake handler without `schema_errors`,
  so `errors = []` and the function returns early without exercising any real logic. A strengthened
  version would fail due to the `co[CHECK_OUTPUT_KEY]` / `nw.LazyFrame` bug. Restore a proper
  test when `drop_invalid_rows` is implemented narwhals-idiomatically in a future phase.

### `Check.native` docstring
- Update to reflect the Phase 5 expression protocol: `native=False` passes `nw.col(key)` (a
  `nw.Expr`) to the check function, not the old `(nw_frame, key)` tuple signature.

### ibis API rename (`DatabaseTable` → `Table`)
- Fix `test_custom_check_receives_table_and_key`: update the assertion from
  `table_type == "DatabaseTable"` to `table_type == "Table"` (ibis renamed the class).

### ROADMAP.md checkbox cleanup
- Mark all plan checkboxes as complete for phases 02, 03, 05, 06 — documentation drift only,
  no code changes required.

### Claude's Discretion
- Exact wording of the updated `Check.native` docstring
- Whether to add a brief inline comment explaining the `nw.from_native()` approach in
  `_count_failure_cases`

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `nw.from_native(..., eager_only=False)` + `.lazy().select(nw.len()).collect()`: already used
  throughout the narwhals backend for backend-agnostic frame operations
- `pandera/api/narwhals/error_handler.py`: 25-line file — the `_count_failure_cases` fix is
  a self-contained 2-line replacement of lines 14-23

### Established Patterns
- Phase 6 contract: `failure_cases` is always native at SchemaError boundary — all counting
  logic must accept native types as input
- Narwhals-is-internal principle: `nw.from_native()` wraps native types for internal operations;
  user-facing outputs remain native

### Integration Points
- `pandera/api/narwhals/error_handler.py` lines 14-23: dead branch + ibis guard → replace with
  unified narwhals count
- `tests/backends/narwhals/test_checks.py` line 456, 477: `xfail` markers to remove
- `tests/backends/narwhals/test_container.py` lines 28, 48, 219, 233: `xfail` markers — 4 to
  remove, `test_drop_invalid_rows` to delete entirely
- `pandera/api/checks.py`: `Check.native` docstring update
- `tests/backends/narwhals/test_e2e.py` line 478: `"DatabaseTable"` → `"Table"`
- `.planning/ROADMAP.md`: stale checkboxes for phases 02, 03, 05, 06

</code_context>

<specifics>
## Specific Ideas

- For `_count_failure_cases`: the 2-line unified approach was preferred over keeping the ibis
  guard — "not ideal to have backend-specific code here"
- `drop_invalid_rows` future phase should address ibis delegation, polars `pl.DataFrame()`
  coupling, and other backends in one coherent narwhals-idiomatic implementation

</specifics>

<deferred>
## Deferred Ideas

- Full narwhals-idiomatic `drop_invalid_rows`: ibis delegation to `IbisSchemaBackend` removed,
  polars-specific `pl.DataFrame()` constructor replaced, all backends handled uniformly — future phase
- Nyquist validation audits for phases 01-06 (all VALIDATION.md files exist but are `status: draft`) —
  run `/gsd:validate-phase {N}` for each

</deferred>

---

*Phase: 07-v1.0-tech-debt-cleanup*
*Context gathered: 2026-03-24*
