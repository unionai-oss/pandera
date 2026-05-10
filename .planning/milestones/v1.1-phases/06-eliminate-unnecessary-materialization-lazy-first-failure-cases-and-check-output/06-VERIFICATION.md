---
phase: 06-eliminate-unnecessary-materialization-lazy-first-failure-cases-and-check-output
verified: 2026-03-23T23:00:00Z
status: passed
score: 16/16 must-haves verified
gaps: []
---

# Phase 6: Eliminate Unnecessary Materialization — Verification Report

**Phase Goal:** Enforce a single principle throughout the narwhals backend: execution is triggered only once — to evaluate the scalar boolean "did the check pass?" — and everything else is returned in the user's original type. failure_cases and check_output must stay lazy until the SchemaError boundary, where they are unwrapped to native types.

**Verified:** 2026-03-23T23:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | run_check() has one code path — no _is_ibis_result bifurcation — materializes only the scalar bool | VERIFIED | base.py lines 87-131: single path, `_materialize(passed_lf)[CHECK_OUTPUT_KEY][0]` only |
| 2 | run_check() returns failure_cases as a narwhals wrapper (nw.LazyFrame or nw.DataFrame), never calling fc.collect() | VERIFIED | base.py line 112: `failure_cases = fc` — no collect(); grep confirms empty |
| 3 | run_check() returns check_output as a narwhals wrapper — never calls _materialize(check_result.check_output) | VERIFIED | base.py line 127: `check_output=check_result.check_output` — no _materialize call |
| 4 | subsample() calls check_obj.head(n) / check_obj.tail(n) directly without _materialize(check_obj) | VERIFIED | base.py lines 81,83: `check_obj.head(head)` and `check_obj.tail(tail)` directly |
| 5 | subsample() raises NotImplementedError for tail= on SQL-lazy backends, detected via hasattr(..., 'execute') | VERIFIED | base.py lines 70-77: guard with hasattr(native, "execute"); test_subsample_ibis_tail_raises PASSES |
| 6 | check_nullable() materializes only the scalar .any() via _materialize(combined_lf.select(...any())) | VERIFIED | components.py line 132: `_materialize(combined_lf.select(nw.col(CHECK_OUTPUT_KEY).any()))` — full-frame materialize gone |
| 7 | check_nullable() returns failure_cases and check_output as narwhals wrappers in CoreCheckResult | VERIFIED | components.py lines 145-154: `failure_cases = combined_lf.filter(...)`, `check_output=combined_lf` — no _to_native |
| 8 | failure_cases_metadata() builds enriched frame using narwhals ops (nw.lit, nw.concat_str) for lazy/SQL path | VERIFIED | base.py lines 189-214: lazy/SQL branch uses nw.lit(), nw.concat_str(), no Arrow roundtrip |
| 9 | failure_cases_metadata() returns native ibis.Table for ibis inputs via backend-aware concat | VERIFIED | base.py lines 285-292: `functools.reduce(a.union(b))` for ibis; test_failure_cases_metadata_ibis_returns_ibis_table PASSES |
| 10 | Row index is None for lazy/SQL backends — no forced ordering | VERIFIED | base.py line 212: `nw.lit(None).cast(nw.Int32).alias("index")` in lazy/SQL path |
| 11 | SchemaError.failure_cases is native at SchemaError construction in components.py | VERIFIED | components.py lines 338-358: manual detection pattern — ibis LazyFrame -> ibis.Table, polars lazy -> collect+to_native -> pl.DataFrame |
| 12 | SchemaError.failure_cases is native at SchemaError construction in container.py | VERIFIED | container.py lines 144-159: same detection pattern; _to_native import at line 13 |
| 13 | check_output stays as narwhals wrapper at SchemaError construction in both files | VERIFIED | components.py line 356: `check_output=result.check_output`; container.py line 162: `check_output=result.check_output` — no unwrap |
| 14 | All 5 subsample tests GREEN: head stays lazy, tail stays lazy, head+tail stays lazy, ibis tail raises, no-params unchanged | VERIFIED | pytest tests/backends/narwhals/test_components.py — 24 passed |
| 15 | All 4 e2e failure_cases type assertions GREEN: polars SchemaError.failure_cases is pl.DataFrame, ibis SchemaError.failure_cases is ibis.Table, ibis SchemaErrors.failure_cases is ibis.Table | VERIFIED | pytest tests/backends/narwhals/test_e2e.py — 44 passed, 1 pre-existing failure (deferred) |
| 16 | NarwhalsErrorHandler._count_failure_cases handles native ibis.Table via .count().execute() | VERIFIED | error_handler.py lines 18-23: ibis.Table branch with count().execute() |

**Score:** 16/16 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/backends/narwhals/test_components.py` | TestSubsample (5 tests) + test_failure_cases_metadata_ibis_returns_ibis_table | VERIFIED | Lines 222-326; all 6 tests pass |
| `tests/backends/narwhals/test_e2e.py` | Updated assertions for Phase 6 type contracts; test_ibis_lazy_failure_cases_is_ibis_table | VERIFIED | Lines 188-284, 628-640; all assertions in Phase 6 contract form |
| `pandera/backends/narwhals/base.py` | Unified run_check(); lazy subsample(); _is_lazy_or_sql() helper; redesigned failure_cases_metadata() | VERIFIED | _is_ibis_result absent (only in comment); fc.collect() absent; nw.lit/nw.concat_str present |
| `pandera/backends/narwhals/components.py` | Scalar-only check_nullable(); boundary unwrap in run_checks_and_handle_errors() | VERIFIED | check_nullable uses _materialize(...select(...any())); failure_cases unwrapped at SchemaError construction |
| `pandera/backends/narwhals/container.py` | Boundary unwrap at SchemaError construction; subsample normalization block compatible | VERIFIED | _to_native imported; manual detection pattern at lines 144-159; subsample normalization unchanged |
| `pandera/api/narwhals/error_handler.py` | _count_failure_cases handles native ibis.Table | VERIFIED | Lines 18-23 present and correct |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| base.py run_check() | _materialize(check_result.check_passed) | single materialization point — scalar bool only | VERIFIED | Line 98: `_materialize(passed_lf)[CHECK_OUTPUT_KEY][0]`; no other _materialize calls on check_result |
| base.py subsample() | check_obj.head(n) | direct lazy call — no _materialize(check_obj) | VERIFIED | Lines 81,83 call head/tail directly; no _materialize(check_obj) anywhere in subsample |
| components.py check_nullable() | _materialize(combined_lf.select) | scalar-only materialization | VERIFIED | Line 132: `_materialize(combined_lf.select(nw.col(CHECK_OUTPUT_KEY).any()))` |
| base.py failure_cases_metadata() | nw.lit(value).alias(name) | narwhals literal attach, works for polars AND ibis | VERIFIED | Lines 207-213: with_columns(nw.lit(...).alias(...)) |
| components.py run_checks_and_handle_errors() | native fc at SchemaError | boundary unwrap | VERIFIED | Lines 338-358: ibis path -> ibis.Table; polars lazy -> collect+to_native |
| container.py validate() | native fc at SchemaError | boundary unwrap | VERIFIED | Lines 144-159: same pattern; `failure_cases=fc` at line 159 |

---

### Requirements Coverage

All three plans declare requirement `LAZY-FIRST-01`. No REQUIREMENTS.md exists in the project — the ROADMAP.md notes "Requirements: TBD" for Phase 6. Requirement tracking is informal; the plans document `LAZY-FIRST-01` as the governing requirement and all three summaries mark it completed.

| Requirement | Source Plans | Description | Status |
|-------------|-------------|-------------|--------|
| LAZY-FIRST-01 | 06-01, 06-02, 06-03 | Lazy-first invariant: materialization only for scalar pass/fail boolean; failure_cases and check_output stay lazy until boundary | SATISFIED — implementation confirmed across all three files; full test suite passing |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `pandera/backends/narwhals/base.py` | 90 | String "_is_ibis_result" in docstring comment only | Info | Not a code anti-pattern — docstring notes "no _is_ibis_result bifurcation" which is correct and intentional |
| `pandera/backends/narwhals/base.py` | 221 | `pl.from_arrow(fc_eager.to_arrow())` | Info | This is in the eager-polars path of failure_cases_metadata(), which is intentionally retained for polars-eager inputs. Not a violation — the lazy/SQL path was redesigned to avoid this. |
| `tests/backends/narwhals/test_e2e.py` | 478 | Pre-existing test failure: `test_custom_check_receives_table_and_key` asserts `"DatabaseTable"` but ibis now uses `"Table"` | Warning | Pre-existing failure deferred in Plan 01, documented as out of scope. Does not affect Phase 6 contracts. |

No blocker anti-patterns found.

---

### Human Verification Required

None — all Phase 6 contracts are mechanically verifiable via grep and test execution.

The one item that could benefit from human review:

**Multi-column lazy failure_cases format:** The lazy/SQL path in failure_cases_metadata() produces `"col=value, col=value"` strings via nw.concat_str(), intentionally different from the eager polars JSON struct format. The decision was accepted in Plan 03 as "no shared contract requires JSON." A human reviewer may want to confirm this format difference is acceptable from a user-facing API perspective — but this is a design decision, not a correctness gap.

---

### Gaps Summary

No gaps. All Phase 6 must-haves are verified against the actual codebase.

**Roadmap discrepancy noted:** The ROADMAP.md still shows Plan 03 as unchecked (`[ ] 06-03-PLAN.md`). The commits `66c7f6e` and `724e9db` confirm Plan 03 executed successfully and all tests pass. The ROADMAP.md checkbox was not updated — this is a documentation artifact only and does not affect goal achievement.

---

_Verified: 2026-03-23T23:00:00Z_
_Verifier: Claude (gsd-verifier)_
