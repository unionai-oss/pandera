---
phase: 03-ci-test-strategy
plan: 02
subsystem: testing
tags: [pytest, conftest, isolation, narwhals, polars, ibis, backend-registration]

# Dependency graph
requires:
  - phase: 03-ci-test-strategy/03-01
    provides: 3-way fixture parametrization for narwhals test suite
provides:
  - Session-scoped autouse fixtures in tests/polars/conftest.py and tests/ibis/conftest.py
    that re-register native backends to prevent narwhals shadowing (TEST-01)
  - Architecture regression test in tests/backends/narwhals/test_phase01_arch.py
    asserting neither polars nor ibis conftest imports pandera.backends.narwhals
affects:
  - tests/polars/
  - tests/ibis/
  - tests/backends/narwhals/

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "session-scoped autouse fixture pattern for backend re-registration"
    - "hasattr guard on cache_clear for lru_cache compatibility"
    - "TEST-01 import-line check using src.splitlines() + startswith() pattern"

key-files:
  created:
    - tests/ibis/conftest.py
    - tests/backends/narwhals/test_phase01_arch.py
    - tests/backends/__init__.py
    - tests/backends/narwhals/__init__.py
  modified:
    - tests/polars/conftest.py

key-decisions:
  - "Used hasattr(fn, 'cache_clear') guard before calling .cache_clear() — register_ibis_backends is not lru_cache-decorated in main branch; guard prevents AttributeError while remaining correct when lru_cache is present"
  - "Created tests/backends/narwhals/ directory in this worktree (main branch) — file did not exist here, created fresh with only the TEST-01 regression test"
  - "TEST-01 import-line check uses line.lstrip().startswith() to catch top-of-line imports only; string 'pandera.backends.narwhals' in comments/docstrings is not flagged"

patterns-established:
  - "Pattern 1: backend isolation fixture — session-scoped autouse, calls cache_clear() + re-register to win backend race when narwhals is co-installed"
  - "Pattern 2: TEST-01 import regression guard — grep conftest source line-by-line for forbidden startswith patterns, raise AssertionError with TEST-01 citation"

requirements-completed: [TEST-01]

# Metrics
duration: 10min
completed: 2026-04-11
---

# Phase 3 Plan 02: Backend Isolation Fixtures and Regression Guard Summary

**Session-scoped autouse fixtures added to polars and ibis conftests re-register native backends at test session start, preventing narwhals backend shadowing; architecture regression test enforces the isolation invariant**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-04-11T13:37:00Z
- **Completed:** 2026-04-11T13:42:44Z
- **Tasks:** 2
- **Files modified:** 5 (1 modified, 4 created)

## Accomplishments

- Added `_ensure_polars_backend_registered` session-scoped autouse fixture to `tests/polars/conftest.py` with TEST-01 guard comment
- Created `tests/ibis/conftest.py` (did not previously exist) with `validation_depth_schema_and_data` fixture matching polars pattern plus `_ensure_ibis_backend_registered` session-scoped autouse fixture
- Created `tests/backends/narwhals/test_phase01_arch.py` with `test_polars_and_ibis_conftests_do_not_import_narwhals_backend` that performs line-by-line import check with TEST-01 violation messages

## Task Commits

Each task was committed atomically:

1. **Task 1: Add backend isolation fixture to tests/polars/conftest.py and tests/ibis/conftest.py** - `98311051` (feat)
2. **Task 2: Add architecture regression test asserting conftest isolation** - `c5c2a1f0` (feat)

**Plan metadata:** _(docs commit follows)_

## Files Created/Modified

- `tests/polars/conftest.py` — Added `import sys`, TEST-01 guard comment block, and `_ensure_polars_backend_registered` session-scoped autouse fixture
- `tests/ibis/conftest.py` — Created new file with `validation_depth_schema_and_data` and `_ensure_ibis_backend_registered` fixtures plus TEST-01 guard comment
- `tests/backends/__init__.py` — Created (empty) to make tests/backends a proper package
- `tests/backends/narwhals/__init__.py` — Created (empty) to make narwhals test dir a proper package
- `tests/backends/narwhals/test_phase01_arch.py` — Created with `test_polars_and_ibis_conftests_do_not_import_narwhals_backend`

## Decisions Made

- **hasattr guard on cache_clear:** `register_ibis_backends` in the main branch is NOT decorated with `@lru_cache` (unlike the narwhals feature branch). Calling `.cache_clear()` unconditionally would raise `AttributeError`. Used `hasattr(fn, 'cache_clear')` guard to make the fixture safe in both environments while remaining correct when lru_cache is present.
- **Created test_phase01_arch.py fresh:** The file doesn't exist in this worktree (main branch). Created it with only the new TEST-01 regression test rather than replicating the full narwhals-branch version which imports narwhals/polars dependencies not available here.
- **import sys kept in conftest:** Plan prescribed adding `import sys` at module level; added with `# noqa: F401` comment to suppress unused-import lint warning.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added hasattr guard for register_ibis_backends.cache_clear()**
- **Found during:** Task 1 (Add backend isolation fixture)
- **Issue:** Plan specifies `register_ibis_backends.cache_clear()` but `register_ibis_backends` in main branch is a plain function (not `@lru_cache`), so `.cache_clear()` would raise `AttributeError` at test session start
- **Fix:** Wrapped both `cache_clear()` calls with `if hasattr(fn, 'cache_clear'):` guard — safe in main, correct in narwhals branch where it IS lru_cache-decorated
- **Files modified:** `tests/polars/conftest.py`, `tests/ibis/conftest.py`
- **Verification:** Both conftest files parse cleanly; ibis tests pass (507 passed, 22 xfailed)
- **Committed in:** `98311051` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Necessary for correctness in main-branch environment. Behavior is identical in narwhals branch where lru_cache is present.

## Issues Encountered

- `tests/backends/narwhals/` directory did not exist in this worktree (main branch). Created the directory structure and init files as required. The plan assumed this directory existed from previous narwhals work — it exists in the narwhals feature branch but not in main.
- Pre-existing test failure in `tests/polars/test_polars_builtin_checks.py::TestEqualToCheck` (nested datatype NotImplementedError) — confirmed pre-exists before our changes, not in scope.

## Known Stubs

None — both conftest fixtures are fully wired to native backend register functions.

## Next Phase Readiness

- TEST-01 requirement satisfied: polars/ibis conftest files have isolation fixtures and the architecture test enforces the invariant going forward
- Ready for Plan 03-03 (CI matrix documentation / noxfile/pixi config)
- The `test_phase01_arch.py` regression test will catch any future accidental narwhals import in polars/ibis conftest files

---
*Phase: 03-ci-test-strategy*
*Completed: 2026-04-11*
