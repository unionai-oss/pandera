---
phase: 01-pr-review-architecture-fixes
verified: 2026-03-22T02:10:00Z
status: passed
score: 12/12 must-haves verified
re_verification: true
  previous_status: gaps_found
  previous_score: 9/12
  gaps_closed:
    - "All docstrings/comments in pandera/backends/narwhals/*.py use 'Narwhals' (capital N) where referring to the framework by name"
    - "ROADMAP.md plan marker for 01-02 reflects completed status"
    - "validate() defers native materialization until after subsampling — subsampling operates on narwhals LazyFrame, not native frame"
  gaps_remaining: []
  regressions: []
human_verification:
  - test: "Run narwhals test suite"
    expected: "All 125+ narwhals tests pass (125 passed, 1 skipped, 3 xfailed, 4 xpassed per summary)"
    why_human: "Test execution environment and dependencies (polars, ibis) required"
  - test: "Run ibis test suite after NarwhalsErrorHandler wiring"
    expected: "No new failures vs. pre-phase baseline (~88 pre-existing failures)"
    why_human: "Ibis integration tests require ibis + SQLite test environment"
---

# Phase 1: PR Review Architecture Fixes — Verification Report

**Phase Goal:** Address architectural feedback from PR Review #2223 — separate ibis logic from base ErrorHandler, create NarwhalsErrorHandler, remove polars-specific coupling from narwhals container backend, fix misleading comments, and fix Narwhals capitalization.
**Verified:** 2026-03-22T02:10:00Z
**Status:** passed
**Re-verification:** Yes — after gap closure (Plan 03 committed as `42981c1` and `c4cc0e8`)

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Base ErrorHandler._count_failure_cases contains no ibis-specific imports or isinstance checks | VERIFIED | `grep -n "ibis" pandera/api/base/error_handler.py` returns nothing |
| 2 | NarwhalsErrorHandler exists in pandera/api/narwhals/error_handler.py and overrides _count_failure_cases | VERIFIED | File exists, subclasses base ErrorHandler, overrides _count_failure_cases with guarded ibis import |
| 3 | NarwhalsErrorHandler handles ibis.Table failure cases via .count().to_pyarrow().as_py() | VERIFIED | Guarded import + isinstance check + .count().to_pyarrow().as_py() in narwhals error_handler.py |
| 4 | Existing ibis ErrorHandler unchanged (pandera/api/ibis/error_handler.py) | VERIFIED | File still has hard import ibis, still overrides _count_failure_cases with ibis.Table branch |
| 5 | container.py imports NarwhalsErrorHandler (not base ErrorHandler) | VERIFIED | `cont_mod.ErrorHandler is NarwhalsEH` -> True (confirmed programmatically) |
| 6 | components.py imports NarwhalsErrorHandler (not base ErrorHandler) | VERIFIED | `comp_mod.ErrorHandler is NarwhalsEH` -> True (confirmed programmatically) |
| 7 | base.py failure_cases_metadata uses NarwhalsErrorHandler (not base ErrorHandler) | VERIFIED | `base_mod.ErrorHandler is NarwhalsEH` -> True; ErrorHandler() instantiates NarwhalsEH |
| 8 | _to_frame_kind_nw uses duck-typing instead of issubclass(return_type, pl.DataFrame) | VERIFIED | `hasattr(return_type, "collect")` present; no `issubclass(return_type, pl.DataFrame)` anywhere in container.py |
| 9 | import polars as pl is removed from container.py | VERIFIED | No `import polars` or `issubclass.*pl.` in container.py |
| 10 | collect_schema_components detects ibis vs polars schema dynamically | VERIFIED | `_schema_module = schema.__class__.__module__`; conditional Column import at lines 309-313 |
| 11 | The misleading comment 'Convert to native pl.LazyFrame for column component dispatch' is updated | VERIFIED | Old comment absent; replaced with "Convert to native frame for column component dispatch." |
| 12 | All docstrings/comments use 'Narwhals' (capital N) where referring to the framework by name | VERIFIED | grep for `narwhals LazyFrame\|narwhals APIs\|narwhals collect\(\)\|narwhals equivalent` across all 3 backend files returns no matches (import paths and identifiers excluded) |

**Score:** 12/12 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pandera/api/narwhals/error_handler.py` | NarwhalsErrorHandler subclass | VERIFIED | Exists, subclasses base ErrorHandler, overrides _count_failure_cases with guarded ibis import |
| `pandera/api/base/error_handler.py` | Cleaned base ErrorHandler with no ibis logic | VERIFIED | Zero ibis references; _count_failure_cases handles str, len(), None/scalar only |
| `pandera/backends/narwhals/container.py` | Fixed container backend with no polars-specific type checks; validate() subsamples on nw.LazyFrame | VERIFIED | No `import polars as pl`, no `issubclass(return_type, pl.DataFrame)`, subsample() receives check_lf directly (line 104-110), _to_frame_kind_nw deferred to return statements only (lines 159, 168, 171) |
| `pandera/backends/narwhals/components.py` | Column backend using NarwhalsErrorHandler; 'Narwhals APIs' capitalized | VERIFIED | `from pandera.api.narwhals.error_handler import ErrorHandler` at line 10; "Narwhals APIs" in class docstring at line 31 |
| `pandera/backends/narwhals/base.py` | Base backend with NarwhalsErrorHandler; 'Narwhals collect()' capitalized | VERIFIED | `from pandera.api.narwhals.error_handler import ErrorHandler` at line 9; "Narwhals collect()" at line 229 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| pandera/api/narwhals/error_handler.py | pandera/api/base/error_handler.py | import and subclass | VERIFIED | `from pandera.api.base.error_handler import ErrorHandler as _ErrorHandler` + `class ErrorHandler(_ErrorHandler)` |
| pandera/api/ibis/error_handler.py | pandera/api/base/error_handler.py | import and subclass (unchanged) | VERIFIED | `from pandera.api.base.error_handler import ErrorHandler as _ErrorHandler` still present |
| pandera/backends/narwhals/container.py | pandera/api/narwhals/error_handler.py | import | VERIFIED | `from pandera.api.narwhals.error_handler import ErrorHandler` at line 12 |
| pandera/backends/narwhals/components.py | pandera/api/narwhals/error_handler.py | import | VERIFIED | `from pandera.api.narwhals.error_handler import ErrorHandler` at line 10 |
| pandera/backends/narwhals/base.py | pandera/api/narwhals/error_handler.py | import | VERIFIED | `from pandera.api.narwhals.error_handler import ErrorHandler` at line 9 |
| container.py validate() | subsample() | passes check_lf (nw.LazyFrame) | VERIFIED | Line 104: `self.subsample(check_lf, ...)` — no _to_frame_kind_nw call before this |
| container.py validate() | _to_frame_kind_nw | single call at return | VERIFIED | _to_frame_kind_nw appears only at lines 159, 168, 171 — all after subsample; no call before subsample |

### Requirements Coverage

Requirements ARCH-01 through ARCH-04 are declared in ROADMAP.md for this phase. There is no separate REQUIREMENTS.md file — the requirement IDs are informational labels referencing PR review feedback categories.

| Requirement | Source Plan | Description (inferred) | Status | Evidence |
|-------------|-------------|----------------------|--------|---------|
| ARCH-01 | 01-01, 01-02 | ErrorHandler class hierarchy: no ibis in base, NarwhalsErrorHandler subclass | SATISFIED | Base ErrorHandler clean; NarwhalsErrorHandler created and wired into all 3 backends |
| ARCH-02 | 01-02, 01-03 | Remove polars-specific coupling from narwhals container backend; validate() defers materialization | SATISFIED | No `import polars as pl`; duck-typing for return type; subsample receives nw.LazyFrame; _to_frame_kind_nw only at return |
| ARCH-03 | 01-02 | Fix misleading comment in run_schema_component_checks | SATISFIED | Comment replaced with "Convert to native frame for column component dispatch." |
| ARCH-04 | 01-02, 01-03 | Narwhals capitalization across all backend files | SATISFIED | All 6 previously-identified lowercase proper-noun instances now capitalized; grep confirms no remaining instances |

### Anti-Patterns Found

No new anti-patterns. No blockers. No stubs. No empty implementations. No unguarded ibis imports in the base class.

Previously-flagged cosmetic issues (lowercase "narwhals" as proper noun) are now resolved per Plan 03 commit `c4cc0e8`.

### Human Verification Required

#### 1. Narwhals Test Suite

**Test:** Run `python -m pytest tests/core/test_polars.py tests/narwhals/ -x -q`
**Expected:** All narwhals tests pass (summary reports 125 passed, 1 skipped, 3 xfailed, 4 xpassed)
**Why human:** Test execution requires polars and narwhals installed in the environment; validate() subsample ordering change alters the execution path for head/tail subsampling cases

#### 2. Ibis Regression Check

**Test:** Run `python -m pytest tests/ibis/ -x -q 2>&1 | tail -20` and compare failure count to pre-phase baseline
**Expected:** No new failures introduced by NarwhalsErrorHandler wiring (pre-existing ~88 failures from SQLite/ibis integration issues are acceptable)
**Why human:** Requires ibis + SQLite test environment

### Re-verification Summary

All three gaps from the initial verification are now closed:

**Gap 1 — Capitalization (ARCH-04, now SATISFIED):** Plan 03 commit `c4cc0e8` fixed all 6 remaining lowercase proper-noun instances: container.py lines 29, 37, 65, 254; components.py line 31; base.py line 229. Grep across all three backend files returns zero matches for the flagged patterns.

**Gap 2 — ROADMAP marker (now SATISFIED):** ROADMAP.md lines 40-42 confirm all three plans marked `[x]` complete. The SUMMARY noted the marker was already correct before Plan 03 ran — the initial verification observed a stale state.

**Gap 3 — validate() premature materialization (ARCH-02, now SATISFIED):** Plan 03 commit `42981c1` corrected the ordering. `self.subsample()` now receives `check_lf` (nw.LazyFrame) directly at line 104. `_to_frame_kind_nw` is called only at the three return/raise points (lines 159, 168, 171) — never before subsampling. The intermediate `check_obj_parsed` variable is now scoped entirely inside the `drop_invalid_rows` branch.

All automated checks pass. Phase goal fully achieved. Human test suite execution remains pending.

---

_Verified: 2026-03-22T02:10:00Z_
_Verifier: Claude (gsd-verifier)_
