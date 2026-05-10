---
phase: 07-v1.0-tech-debt-cleanup
verified: 2026-03-24T00:00:00Z
status: passed
score: 9/9 must-haves verified
re_verification: false
---

# Phase 7: v1.0 Tech Debt Cleanup Verification Report

**Phase Goal:** Address all tech debt identified in the v1.0 milestone audit — fix dead code in `_count_failure_cases`, update the `Check.native` docstring to reflect the current expression-based API, fix the ibis API rename in `test_custom_check_receives_table_and_key`, promote 4 xpassed tests to strict passing, delete one hollow test, and mark stale ROADMAP.md plan checkboxes as complete.
**Verified:** 2026-03-24
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | `_count_failure_cases` uses `nw.from_native(failure_cases, eager_only=False)` with no isinstance branch and no ibis import | VERIFIED | `error_handler.py` is 22 lines; only `nw.from_native` path present |
| 2  | No unused `_materialize` import in `error_handler.py` | VERIFIED | Grep for `_materialize` in file returns no matches |
| 3  | `test_custom_check_receives_table_and_key` asserts `table_type == "Table"` | VERIFIED | Line 478 of `test_e2e.py`: `assert table_type == "Table"` |
| 4  | `Check.native` docstring describes `nw.col(key)` / `nw.Expr` protocol for `native=False` | VERIFIED | Lines 89-90 of `checks.py` mention `nw.col(key)` and `nw.Expr` |
| 5  | `test_failure_cases_metadata` has no xfail marker | VERIFIED | No xfail markers found anywhere in `test_container.py` |
| 6  | `test_ibis_narwhals_auto_activated` has no xfail marker | VERIFIED | No xfail markers found anywhere in `test_container.py` |
| 7  | `test_ibis_backend_is_narwhals` has no xfail marker | VERIFIED | No xfail markers found anywhere in `test_container.py` |
| 8  | `test_postprocess_lazyframe_no_materialization_ibis` has no xfail marker | VERIFIED | Function exists at line 477 of `test_checks.py` with no preceding xfail decorator; three other xfail markers (lines 456, 501, 535) are for different tests and untouched |
| 9  | `test_drop_invalid_rows` does not exist in `test_container.py` | VERIFIED | Grep returns no matches |
| 10 | All plan checkboxes for phases 02, 03, 05, 06, 07 show `[x]` in ROADMAP.md | VERIFIED | No `- [ ]` lines found anywhere in ROADMAP.md; all 13 plan lines across those phases show `[x]` |

**Score:** 10/10 truths verified (9 must-haves from plans + 1 derived from phase goal)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pandera/api/narwhals/error_handler.py` | Unified `_count_failure_cases` using `nw.from_native` | VERIFIED | 22-line file; single `nw.from_native(failure_cases, eager_only=False).lazy().select(nw.len()).collect()["len"][0]` path; no `_materialize` import; no `ibis` import |
| `tests/backends/narwhals/test_e2e.py` | `table_type == "Table"` assertion | VERIFIED | Line 478 contains the corrected assertion |
| `pandera/api/checks.py` | Accurate `native=False` docstring | VERIFIED | Lines 89-90 reference `nw.col(key)` and `nw.Expr` |
| `tests/backends/narwhals/test_container.py` | 3 xfail markers removed; `test_drop_invalid_rows` deleted | VERIFIED | Zero xfail markers in file; `test_drop_invalid_rows` absent |
| `tests/backends/narwhals/test_checks.py` | `test_postprocess_lazyframe_no_materialization_ibis` xfail removed | VERIFIED | Function present without xfail; remaining three xfail markers are for different tests |
| `.planning/ROADMAP.md` | All phase 02/03/05/06/07 plan checkboxes `[x]` | VERIFIED | No unchecked `[ ]` plan lines anywhere in file |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `pandera/api/narwhals/error_handler.py` | `narwhals.stable.v1.from_native` | `nw.from_native(failure_cases, eager_only=False)` | WIRED | Pattern found at line 17 of `error_handler.py` |
| `pandera/api/checks.py` | Phase 5 expression protocol | `native=False` docstring mentions `nw.Expr` and `nw.col(key)` | WIRED | Lines 89-90 confirm both terms present |

---

### Requirements Coverage

No requirement IDs declared for this phase (hygiene/cleanup work only).

---

### Anti-Patterns Found

None. No TODO/FIXME/placeholder comments, no stub implementations, no empty return values in the modified files.

---

### Human Verification Required

None. All changes are structural (code/text content) and verifiable programmatically.

---

### Commits Verified

All four task commits exist in the repository history:

| Commit | Description |
|--------|-------------|
| `931de58` | fix(07-01): replace `_count_failure_cases` dead branch with unified `nw.from_native` count |
| `6ccb4f6` | fix(07-01): update ibis table type assertion from `DatabaseTable` to `Table` |
| `f46d124` | fix(07-02): update `Check.native` docstring and promote 4 xfail tests |
| `f31dc13` | chore(07-02): mark all completed plan checkboxes in ROADMAP.md |

---

## Summary

All must-haves from both plans verified against the actual codebase. Phase 7 goal is fully achieved:

- `error_handler.py` cleaned from 25 lines to 22 lines: dead isinstance branch, ibis try/import guard, and unused `_materialize` import all removed; replaced by a single `nw.from_native` unified count.
- `test_e2e.py` assertion corrected from `"DatabaseTable"` to `"Table"`.
- `checks.py` docstring updated to describe the Phase 5 `nw.col(key)` / `nw.Expr` protocol.
- All 4 xfail markers promoted (3 in `test_container.py`, 1 in `test_checks.py`); the 3 xfail markers that were explicitly required to remain in `test_checks.py` are still present.
- Hollow `test_drop_invalid_rows` deleted from `test_container.py`.
- ROADMAP.md has zero unchecked plan lines.

---

_Verified: 2026-03-24_
_Verifier: Claude (gsd-verifier)_
