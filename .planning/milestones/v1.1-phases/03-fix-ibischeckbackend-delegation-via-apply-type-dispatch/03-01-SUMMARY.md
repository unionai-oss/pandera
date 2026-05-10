---
phase: 03-fix-ibischeckbackend-delegation-via-apply-type-dispatch
plan: "01"
subsystem: api
tags: [narwhals, checks, builtin-checks, check-backend]

# Dependency graph
requires:
  - phase: 02-remaining-pr-review-fixes
    provides: NarwhalsCheckBackend refactored (apply() dispatch via Dispatcher)
provides:
  - "Check.__init__ native: bool parameter (default True) stored as self.native"
  - "from_builtin_check_name always creates Check with native=False"
  - "All 14 narwhals builtin check functions use (frame: nw.LazyFrame, key: str, ...) signature"
affects:
  - "03-02: apply() rewrite uses native=False path to call check_fn(frame, key)"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "native=False flag on Check signals builtin checks that receive (nw_frame, key) rather than native frame"
    - "Builtin check functions directly accept (frame: nw.LazyFrame, key: str, ...) — no NarwhalsData wrapper"

key-files:
  created:
    - tests/core/__init__.py
    - tests/core/test_checks.py
  modified:
    - pandera/api/checks.py
    - pandera/api/base/checks.py
    - pandera/backends/narwhals/builtin_checks.py

key-decisions:
  - "native=False is passed as an explicit keyword in from_builtin_check_name cls() call — before **kws — so user-provided native kwarg cannot override it"
  - "NarwhalsData import removed from builtin_checks.py — no longer needed as annotation after signature refactor"
  - "test_builtin_checks_pass/fail in narwhals backend test suite are expected RED after this plan — plan 02 fixes apply() dispatch"

patterns-established:
  - "TDD: write failing tests first, commit RED, implement GREEN, commit implementation"
  - "Builtin check functions use (frame: nw.LazyFrame, key: str, extra_kwargs...) signature contract"

requirements-completed: []

# Metrics
duration: 4min
completed: 2026-03-22
---

# Phase 03 Plan 01: native Flag and Builtin Signature Refactor Summary

**Check.native flag added (default True, False for all builtins) and all 14 narwhals builtin check functions refactored from (data: NarwhalsData, ...) to (frame: nw.LazyFrame, key: str, ...) signature**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-22T16:49:45Z
- **Completed:** 2026-03-22T16:53:25Z
- **Tasks:** 2
- **Files modified:** 5 (3 source, 2 test)

## Accomplishments
- Added `native: bool = True` parameter to `Check.__init__` with docstring and `self.native = native` assignment
- Propagated `native=False` in `from_builtin_check_name` so all 14 builtin factory methods return `Check` with `native=False`
- Refactored all 14 builtin functions in `builtin_checks.py`: `data: NarwhalsData` -> `frame: nw.LazyFrame, key: str`, all `data.frame` -> `frame`, all `data.key` -> `key`
- Removed `NarwhalsData` import from `builtin_checks.py`
- Created `tests/core/test_checks.py` with 33 tests (TDD RED/GREEN for both tasks)

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: failing native flag tests** - `c2e43f5` (test)
2. **Task 1 GREEN: Add native param to Check.__init__ and from_builtin_check_name** - `13fa2d8` (feat)
3. **Task 2 RED: failing builtin signature tests** - `21e46d7` (test)
4. **Task 2 GREEN: refactor all 14 builtin check signatures** - `44ef28d` (refactor)

_Note: TDD tasks have multiple commits (test RED -> feat/refactor GREEN)_

## Files Created/Modified
- `pandera/api/checks.py` - Added `native: bool = True` param, `self.native = native`, docstring
- `pandera/api/base/checks.py` - Added `native=False` in `from_builtin_check_name` cls() call
- `pandera/backends/narwhals/builtin_checks.py` - All 14 functions refactored to `(frame, key, ...)` signature; NarwhalsData import removed
- `tests/core/__init__.py` - New package init
- `tests/core/test_checks.py` - 33 tests: TestNativeFlag (5), TestBuiltinNativeFalse (14), TestBuiltinCheckSignatures (14)

## Decisions Made
- `native=False` is placed before `**kws` in the `cls(...)` call in `from_builtin_check_name` — making it an explicit keyword that cannot be overridden by user-provided kwargs in `kws`
- `NarwhalsData` import fully removed from `builtin_checks.py` since it was only used as a type annotation
- `test_builtin_checks_pass` and `test_builtin_checks_fail` in `tests/backends/narwhals/test_checks.py` are expected to be RED after this plan — this is intentional; plan 02 fixes `apply()` to call `check_fn(frame, key)` via the `native=False` path

## Deviations from Plan

**1. [Rule 2 - Missing Critical] Created tests/core/ package and test_checks.py**
- **Found during:** Task 1 setup
- **Issue:** Plan specified `pytest tests/core/test_checks.py` as the verify command but the `tests/core/` directory and file did not exist
- **Fix:** Created `tests/core/__init__.py` and `tests/core/test_checks.py` with full TDD test suite for both tasks
- **Files modified:** tests/core/__init__.py, tests/core/test_checks.py (new files)
- **Verification:** All 33 tests pass
- **Committed in:** c2e43f5, 21e46d7 (task RED commits)

---

**Total deviations:** 1 auto-fixed (missing test infrastructure)
**Impact on plan:** The test file creation was required to run the plan's own verify command — a necessary prerequisite, not scope creep.

## Issues Encountered
None - plan executed smoothly. The `tests/core/` directory was created as part of the TDD process since the plan referenced it.

## Next Phase Readiness
- `Check.native` flag is set and ready for plan 02's `apply()` rewrite
- All 14 builtin functions have the `(frame, key, ...)` signature that plan 02's `native=False` path will invoke as `check_fn(frame, key)`
- `test_builtin_checks_pass` / `test_builtin_checks_fail` are expected to be RED until plan 02 fixes `apply()` dispatch

---
*Phase: 03-fix-ibischeckbackend-delegation-via-apply-type-dispatch*
*Completed: 2026-03-22*
