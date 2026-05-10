---
phase: 09-accumulate-check-outputs-into-single-wide-table-for-narwhals-idiomatic-drop-invalid-rows
verified: 2026-03-25T06:30:00Z
status: passed
score: 7/7 must-haves verified
re_verification: false
---

# Phase 9: Accumulate Check Outputs via nw.Expr — Verification Report

**Phase Goal:** Refactor the narwhals backend check loop so that per-check boolean outputs accumulate into a single wide table during iteration, enabling drop_invalid_rows to be a pure narwhals all_horizontal filter — no backend-specific isinstance checks, no IbisSchemaBackend delegation.
**Verified:** 2026-03-25T06:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (from 09-02-PLAN must_haves)

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | polars lazy drop_invalid_rows filters to only valid rows (lazy=True path) | VERIFIED | 14 polars test_drop_invalid_rows variants: 14 passed (including 8 lazy=True parametrized cases) |
| 2 | polars nullable drop_invalid_rows respects ignore_na (rows with null pass if nullable=True) | VERIFIED | test_drop_invalid_rows_nullable: 2/2 passed |
| 3 | ibis duckdb drop_invalid_rows filters correctly — no IbisSchemaBackend delegation | VERIFIED | test_drop_invalid_rows[duckdb-*]: 6/6 passed |
| 4 | ibis sqlite drop_invalid_rows filters correctly — no IbisSchemaBackend delegation | VERIFIED | test_drop_invalid_rows[sqlite-*]: 6/6 passed |
| 5 | narwhals backend suite has zero regressions (208+ tests pass) | VERIFIED | 209 passed, 8 skipped, 1 xfailed — up from 208 pre-phase-09 |
| 6 | polars container suite has zero regressions | VERIFIED | Pre-phase-09: 30 failures; post-phase-09: 22 failures (8 drop_invalid_rows tests fixed, 0 new failures) |
| 7 | ibis container suite has zero regressions | VERIFIED | Pre-phase-09: 26 failures; post-phase-09: 14 failures (12 drop_invalid_rows tests fixed, 0 new failures) |

**Score: 7/7 truths verified**

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pandera/backends/narwhals/checks.py` | apply() returns nw.Expr; postprocess_expr_output() stores expr+None failure_cases (deferred) | VERIFIED | apply() native=False path returns `expr` directly (line 78); element_wise path returns `expr` (line 56); `postprocess_expr_output()` defined at line 115; `postprocess()` dispatches to it via `isinstance(check_output, nw.Expr)` at line 105 |
| `pandera/backends/narwhals/base.py` | drop_invalid_rows() uses nw.all_horizontal; failure_cases_metadata() reconstructs failure_cases from stored nw.Expr | VERIFIED | `nw.all_horizontal` used at line 457; failure_cases_metadata() reconstructs nw.Expr check_output at lines 205-236; run_check() also handles deferred expr path at lines 107-124 |
| `pandera/backends/narwhals/container.py` | config_context(SCHEMA_AND_DATA) when drop_invalid_rows=True to ensure data checks run on lazy frames | VERIFIED | Lines 132-135: `config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA)` applied when `drop_invalid_rows=True` |
| `tests/backends/narwhals/test_parity.py` | parity test promoted from xfail to strict passing | VERIFIED | `test_drop_invalid_rows_expr_accumulation` (line 277) has no xfail mark and passes; tests row-drop behavior AND nw.Expr check_output contract |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `checks.py::apply()` native=False path | `nw.Expr` returned | `return expr` (line 78) | WIRED | `check_fn(nw.col(key))` or `check_fn(frame)` result returned directly — no `frame.with_columns()` wrapping |
| `checks.py::apply()` element_wise path | `nw.Expr` returned | `return expr` (line 56) | WIRED | `selector.map_batches(...)` result returned directly — no `frame.with_columns()` wrapping |
| `checks.py::postprocess()` | `postprocess_expr_output()` | `isinstance(check_output, nw.Expr)` (line 105) | WIRED | nw.Expr branch added first, before existing LazyFrame/bool branches |
| `base.py::drop_invalid_rows()` | `nw.all_horizontal` accumulation | accumulate nw.Expr entries from schema_errors (lines 414-457) | WIRED | Iterates errors, collects `(col_name, expr, check)` tuples, builds wide frame with single `with_columns` call, filters via `nw.all_horizontal` |
| `base.py::failure_cases_metadata()` | reconstructs failure_cases from stored nw.Expr | `err.failure_cases is False and isinstance(err.check_output, nw.Expr)` (lines 205-236) | WIRED | Reconstructs wide table from `err.data + err.check_output` when SchemaErrors raised |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| DIR-01 | 09-02-PLAN | polars lazy drop_invalid_rows filters correctly | SATISFIED | 14 polars drop_invalid_rows tests pass (8 lazy=True, 6 eager=False variants); test_drop_invalid_rows_expr_accumulation passes (2 rows returned, row -1 dropped) |
| DIR-02 | 09-02-PLAN | polars nullable drop_invalid_rows + ignore_na | SATISFIED | test_drop_invalid_rows_nullable: 2/2 passed; ignore_na handled at column level in `postprocess_expr_output()` and `drop_invalid_rows()` |
| DIR-03 | 09-02-PLAN | ibis duckdb drop_invalid_rows filters correctly | SATISFIED | test_drop_invalid_rows[duckdb-*]: 6/6 passed; no IbisSchemaBackend delegation in drop_invalid_rows |
| DIR-04 | 09-02-PLAN | ibis sqlite drop_invalid_rows filters correctly | SATISFIED | test_drop_invalid_rows[sqlite-*]: 6/6 passed |
| DIR-05 | 09-02-PLAN | No regression in narwhals backend suite | SATISFIED | 209 passed, 8 skipped, 1 xfailed — same xfail pattern maintained (coerce not yet implemented) |
| DIR-06 | 09-02-PLAN | No regression in polars container suite | SATISFIED | Pre-phase: 30 failures; post-phase: 22 failures — 8 fewer (drop_invalid_rows fixed), 0 new failures |
| DIR-07 | 09-02-PLAN | No regression in ibis container suite | SATISFIED | Pre-phase: 26 failures; post-phase: 14 failures — 12 fewer (drop_invalid_rows fixed), 0 new failures |

No orphaned requirements — all DIR-01 through DIR-07 appear in 09-02-PLAN.md `requirements` field and are accounted for. 09-01-PLAN.md declared DIR-01 through DIR-04 for the TDD RED baseline (establishing the failing test count), which is correct.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `pandera/backends/narwhals/base.py` | 7 | `import polars as pl` module-level | Info | Not an anti-pattern — polars is used legitimately in `failure_cases_metadata()` eager path (pl.from_arrow, pl.concat) and scalar path (pl.DataFrame). Module-level import is correct. |

No blocker or warning anti-patterns found:
- No `isinstance(native, _ibis.Table)` remaining in `drop_invalid_rows()`
- No `IbisSchemaBackend` delegation
- No `co[CHECK_OUTPUT_KEY]` LazyFrame indexing
- No `TODO`/`FIXME`/`PLACEHOLDER` comments in modified files
- No empty implementations or stub returns in `drop_invalid_rows()`, `postprocess_expr_output()`, or `failure_cases_metadata()`

---

### Human Verification Required

None. All phase behaviors are programmatically verifiable via existing test suites. The test runs confirm:
- Row filtering produces correct output (test_drop_invalid_rows_expr_accumulation asserts `result_df["a"].to_list() == [1, 2]`)
- nw.Expr contract enforced by assertion in same test
- Cross-backend parity (polars + ibis duckdb + ibis sqlite) covered by parametrized test suites

---

### Gaps Summary

No gaps. All 7 must-have truths verified, all 4 key artifacts confirmed substantive and wired, all 5 key links verified, all 7 requirements satisfied.

**Notable deviation from plan (auto-fixed, no gaps):** The SUMMARY documents two auto-fixes that deviated from the original 09-02-PLAN scope:
1. `container.validate()` required `config_context(SCHEMA_AND_DATA)` when `drop_invalid_rows=True` — polars LazyFrame defaulted to SCHEMA_ONLY, skipping all data checks. This fix was essential and is verified present at container.py lines 132-135.
2. The parity test was redesigned (from asserting SchemaErrors raised to asserting rows dropped) — correct behavior, test passes.

Both deviations were auto-fixed within the plan execution. The final codebase is correct and complete.

---

_Verified: 2026-03-25T06:30:00Z_
_Verifier: Claude (gsd-verifier)_
