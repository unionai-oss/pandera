# Phase 5: Ibis Registration and Integration - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Register the narwhals backend for `ibis.Table`, close known xfail gaps (coerce_dtype, column unique), and complete the full test suite (TEST-02, TEST-04) covering both Polars and Ibis backends. Covers REGISTER-03, TEST-02, TEST-04.

</domain>

<decisions>
## Implementation Decisions

### Ibis registration trigger
- `register_ibis_backends()` gains narwhals auto-detection, mirroring the polars pattern exactly: detect narwhals at registration time, register narwhals backends for `ibis.Table` when narwhals is installed
- Emits the same `UserWarning` as `register_polars_backends()` when narwhals auto-activates
- `tests/backends/narwhals/conftest.py` calls `register_ibis_backends()` alongside `register_polars_backends()` (same autouse module fixture) — narwhals conftest needs explicit registration because tests hit backends directly without going through schema.validate()
- `pandera/backends/narwhals/register.py` is deleted — it's a dead file (nothing imports it); registration responsibility stays co-located with each library's own register.py

### Multi-column uniqueness for Ibis
- Both container-level (`check_column_values_are_unique`) and column-level (`check_unique`) use `group_by().agg(nw.len())` — SQL-lazy safe, works across Polars and Ibis without requiring collect()
- Column-level `check_unique` is not implemented in the ibis backend at all; Phase 5 implements it in the narwhals `ColumnBackend` via `group_by().agg(nw.len())`
- This replaces/supersedes the Polars-only `collect()`-then-`is_duplicated()` approach for Ibis paths

### drop_invalid_rows for Ibis
- Detect if the underlying native frame is an `ibis.Table` (after `nw.to_native()`), delegate to `IbisSchemaBackend.drop_invalid_rows()`, re-wrap with `nw.from_native()` — one `try/except ImportError` guard in `base.py`
- Rationale: `err.check_output` is already native (ibis boolean column table) by this point; everything else in the pipeline went through narwhals; delegation is clean post-processing
- Fix the existing Polars path in `drop_invalid_rows` at the same time: replace `pl.fold` with `nw.all_horizontal()` to use narwhals-native boolean reduction

### TEST-04 parity test structure
- New file: `tests/backends/narwhals/test_parity.py`
- Draw from: `test_polars_container.py`, `test_ibis_container.py`, `test_polars_decorators.py`, `test_ibis_decorators.py`, `test_polars_model.py`, `test_ibis_model.py`
- Exclude: dtype-specific tests (test_polars_dtypes.py, test_ibis_dtypes.py), strategy tests (test_polars_strategies.py), pydantic/typing tests, builtin check tests (already covered by test_checks.py)
- Coerce-dependent tests: include as `xfail(strict=True)` — coerce is a clearly bounded v2 feature; strict=True ensures CI breaks when coerce lands, forcing mark cleanup rather than letting stale xfails accumulate silently

### Claude's Discretion
- Exact narwhals `group_by().agg()` expression for uniqueness checks
- Whether the ibis-delegation branch in `drop_invalid_rows` uses `isinstance(native, ibis.Table)` or a string backend name check
- Exact structure/grouping of tests within test_parity.py

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `pandera/backends/polars/register.py` — direct template for adding narwhals auto-detection to `register_ibis_backends()`; same pattern: try narwhals import, register narwhals backends on success, emit UserWarning
- `pandera/backends/ibis/base.py: IbisSchemaBackend.drop_invalid_rows()` — used directly for Ibis delegation; handles both positional-join and row_number-index backends
- `pandera/backends/narwhals/base.py: NarwhalsSchemaBackend` — `drop_invalid_rows()` to be fixed (pl.fold → nw.all_horizontal for Polars, delegate to IbisSchemaBackend for Ibis)
- `tests/backends/narwhals/conftest.py` — autouse module fixture already calls `register_polars_backends()`; add `register_ibis_backends()` call here
- `tests/polars/test_polars_container.py`, `tests/ibis/test_ibis_container.py` — source for test_parity.py; container-level validation, strict/filter, lazy=True, drop_invalid_rows, unique

### Established Patterns
- `register_polars_backends()` pattern: `lru_cache`, `try/except ImportError`, direct `BACKEND_REGISTRY` writes — replicate for ibis
- `err.check_output` is already native (`_to_native()` called in `base.py: run_check()` before storing)
- `xfail(strict=False)` for unimplemented stubs — established throughout narwhals test suite
- `narwhals/register.py` currently has no imports and nothing imports it — safe to delete

### Integration Points
- `pandera/backends/ibis/register.py` — add narwhals auto-detection block
- `pandera/backends/narwhals/base.py` — fix `drop_invalid_rows` (nw.all_horizontal + ibis delegation)
- `pandera/backends/narwhals/components.py` — add `check_unique` via `group_by().agg(nw.len())`
- `pandera/backends/narwhals/container.py` — update `check_column_values_are_unique` to use `group_by().agg(nw.len())` for SQL-lazy safety
- `tests/backends/narwhals/conftest.py` — add `register_ibis_backends()` call
- `tests/backends/narwhals/test_parity.py` — new file

</code_context>

<specifics>
## Specific Ideas

- The narwhals backend is internal plumbing — users don't see it. When `register_ibis_backends()` runs with narwhals installed, `schema.validate(ibis_table)` from `pandera.ibis` routes through narwhals internally with no API change.
- `drop_invalid_rows` delegation preserves the invariant: narwhals handles validation, ibis handles ibis-specific post-processing where narwhals has no equivalent abstraction (positional joins, row_number windows).
- The `group_by().agg(nw.len())` uniqueness approach mirrors what the existing ibis backend does with window functions — stays SQL-side, no materialization until a failure is found.

</specifics>

<deferred>
## Deferred Ideas

- None — discussion stayed within phase scope.

</deferred>

---

*Phase: 05-ibis-registration-and-integration*
*Context gathered: 2026-03-14*
