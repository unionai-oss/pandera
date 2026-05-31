---
phase: 08-fix-lazy-true-critical-regressions
verified: 2026-03-24T00:00:00Z
status: passed
score: 5/5 must-haves verified
re_verification: false
---

# Phase 8: Fix lazy=True Critical Regressions Verification Report

**Phase Goal:** Close the two critical integration breaks found in the v1.0 post-audit: (1) `failure_cases_metadata()` collapsing N polars lazy failure rows to a single repr string, and (2) `_count_failure_cases()` crashing with `TypeError` when `failure_cases` is a bool scalar. Both fixes must be narwhals-idiomatic — no native type-dependent `isinstance` checks — and the lazy=True path must work for both polars and ibis.
**Verified:** 2026-03-24
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | polars lazy=True SchemaErrors.failure_cases has N per-row values (not a single repr string) | VERIFIED | `test_lazy_failure_cases_per_row_polars` PASSES — `len(fc) == 3`, values are numeric-castable strings (not a DataFrame repr) |
| 2 | ibis lazy=True SchemaErrors.failure_cases is a native ibis.Table with N rows | VERIFIED | `test_lazy_failure_cases_per_row_ibis` PASSES — `isinstance(fc, ibis.Table)` and `fc.count().execute() == 3` |
| 3 | `_count_failure_cases()` does not crash with TypeError when failure_cases is a bool scalar | VERIFIED | `test_lazy_bool_output_check_does_not_crash` PASSES — `SchemaErrors` raised, not `TypeError` |
| 4 | All three regression tests from Plan 01 are GREEN | VERIFIED | `pytest tests/backends/narwhals/test_lazy_regressions.py` → 3 passed |
| 5 | Full narwhals backend test suite passes with no new failures | VERIFIED | `pytest tests/backends/narwhals/` → 208 passed, 8 skipped, 1 xfailed |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/backends/narwhals/test_lazy_regressions.py` | Regression tests for MISSING-01 and MISSING-02 | VERIFIED | Exists, 105 lines, 3 test functions with substantive assertions |
| `pandera/backends/narwhals/base.py` | `failure_cases_metadata()` with unified `nw.from_native` try/except rewrap | VERIFIED | Lines 183-187: `try: fc = nw.from_native(fc, eager_or_interchange_only=False) / except TypeError: pass` — ibis-specific block replaced |
| `pandera/api/narwhals/error_handler.py` | `_count_failure_cases()` with `try/except TypeError` scalar fallback | VERIFIED | Lines 18-26: full count expression wrapped in try/except TypeError; `isinstance(str)` guard removed; fallback `0 if None else 1` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `test_lazy_regressions.py` | `pandera/backends/narwhals/base.py` | `schema.validate(lazy=True)` → `failure_cases_metadata()` | WIRED | Tests call `schema.validate(..., lazy=True)` which exercises `failure_cases_metadata()` — confirmed by 3 passing tests |
| `pandera/backends/narwhals/base.py` | `nw.from_native` | unified `try: nw.from_native(fc, eager_or_interchange_only=False) / except TypeError: pass` | WIRED | Pattern present at lines 184-187; replaces old ibis-specific 6-line block |
| `test_lazy_regressions.py` | `pandera/api/narwhals/error_handler.py` | `SchemaErrors` accumulation → `_count_failure_cases()` | WIRED | MISSING-02 test exercises bool scalar path which calls `_count_failure_cases(False)` — test passes |
| `pandera/api/narwhals/error_handler.py` | `nw.from_native` | `try: nw.from_native(...) / except TypeError` wrapping count expression | WIRED | Pattern `except TypeError` present at line 25 |

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|--------------|-------------|--------|----------|
| MISSING-01 | 08-01-PLAN.md, 08-02-PLAN.md | `failure_cases_metadata()` drops per-row content for polars lazy=True mode — N rows collapsed to 1 repr string | SATISFIED | Unified `nw.from_native` try/except in `base.py` lines 183-187 routes `pl.DataFrame` to eager polars path; `test_lazy_failure_cases_per_row_polars` and `test_lazy_failure_cases_per_row_ibis` both PASS |
| MISSING-02 | 08-01-PLAN.md, 08-02-PLAN.md | `_count_failure_cases()` raises `TypeError` for bool scalar `failure_cases` | SATISFIED | try/except TypeError in `error_handler.py` lines 18-26 returns `0 if None else 1`; `test_lazy_bool_output_check_does_not_crash` PASSES |

**Note on FLOW-BROKEN-01 and FLOW-BROKEN-02:** The ROADMAP lists these flow IDs as gap closures for Phase 8, but neither PLAN declares them as requirement IDs. They are flow-level descriptions of the same two bugs as MISSING-01 and MISSING-02 respectively (documented in `v1.0-MILESTONE-AUDIT.md` lines 68-77). Their resolution is fully covered by the MISSING-01 and MISSING-02 fixes — no orphaned requirement.

### Narwhals-Idiomatic Constraint Verification

The phase goal explicitly requires "no native type-dependent `isinstance` checks." Verified:

- `failure_cases_metadata()` in `base.py`: old ibis-specific `import ibis as _ibis; if isinstance(fc, _ibis.Table)` block (6 lines) is gone. Replaced by `try: nw.from_native(fc, ...) / except TypeError: pass` (4 lines) — narwhals defines what it accepts; TypeError signals non-wrappable scalars.
- `_count_failure_cases()` in `error_handler.py`: old `isinstance(failure_cases, str)` guard removed. No native type checks remain in the method.
- The `isinstance(native, _ibis.Table)` at `base.py` line 341 is in the unrelated `drop_invalid_rows()` method (line 322), which is outside Phase 8's scope and was not modified.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | — | — | No anti-patterns found in modified files |

No TODO/FIXME/placeholder comments in the modified files. No empty implementations. No stub returns. No console-only handlers.

### Human Verification Required

None. All three observable truths are verifiable programmatically via pytest, and all three tests pass with substantive assertions (row counts, type checks, exception type).

### Gaps Summary

No gaps. All five must-have truths are verified against the actual codebase:

1. The production fix to `failure_cases_metadata()` exists at `base.py` lines 183-187 — confirmed via Read.
2. The production fix to `_count_failure_cases()` exists at `error_handler.py` lines 18-26 — confirmed via Read.
3. All three regression tests exist in `test_lazy_regressions.py` with substantive assertions — confirmed via Read.
4. All three regression tests PASS — confirmed by running pytest (3 passed in 1.38s).
5. No regressions in the broader suite — confirmed by running pytest (208 passed, 8 skipped, 1 xfailed).

Both fixes are narwhals-idiomatic (no native type-dependent isinstance checks in modified methods). Commits `f8b9993` (MISSING-01 fix) and `4c9d526` (MISSING-02 fix) both exist and are correctly described.

---

_Verified: 2026-03-24_
_Verifier: Claude (gsd-verifier)_
