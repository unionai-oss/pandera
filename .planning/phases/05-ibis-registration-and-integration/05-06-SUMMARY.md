---
phase: 05-ibis-registration-and-integration
plan: "06"
subsystem: testing
tags: [ibis, error-handler, narwhals, regression-fix]

# Dependency graph
requires:
  - phase: 05-ibis-registration-and-integration
    provides: "run_check ibis path returns ibis.Table as failure_cases (plan 05-04)"
provides:
  - "ibis.Table branch in _count_failure_cases using .count().execute()"
  - "regression test test_custom_check_ibis_lazy in test_parity.py"
affects: [error-handler, ibis-backend, narwhals-backend]

# Tech tracking
tech-stack:
  added: []
  patterns: [try/except ImportError guard for optional ibis dependency in shared code]

key-files:
  created: []
  modified:
    - pandera/api/base/error_handler.py
    - tests/backends/narwhals/test_parity.py

key-decisions:
  - "ibis guard in _count_failure_cases uses try/except ImportError so ibis remains fully optional from shared base code"
  - "test_custom_check_ibis_lazy uses IbisData + ibis.selectors to produce ibis.Table failure_cases — the exact crash path, not a workaround"
  - "ibis.Table detection placed BEFORE the try/len() block to intercept ExpressionError before it propagates"

patterns-established:
  - "Pattern: optional-dependency guard in shared code — try: import X; if isinstance(obj, X.Type): ...; except ImportError: pass"

requirements-completed: [TEST-02, TEST-04]

# Metrics
duration: 5min
completed: 2026-03-15
---

# Phase 5 Plan 06: Guard _count_failure_cases against ibis.Table len() crash

**ibis.Table branch added to _count_failure_cases using .count().execute(), fixing test_dataframe_level_checks regression from plan 05-04 and reducing ibis test failures from 95 to 89**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-15T23:20:00Z
- **Completed:** 2026-03-15T23:26:55Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Fixed `_count_failure_cases` to handle `ibis.Table` using `.count().execute()` instead of `len()` — ibis raises `ExpressionError("Use .count() instead")` which was not caught by the existing `except TypeError` clause
- Added regression test `test_custom_check_ibis_lazy` using a proper `IbisData`-based custom check that produces `ibis.Table` failure_cases, correctly triggering the crash path
- Resolved the sole remaining UAT blocker for Phase 5 — `test_dataframe_level_checks` in `tests/ibis/test_ibis_container.py` now passes
- Reduced ibis test failure count from 95 to 89 (also fixed 2 `test_*_n_failure_cases` and 3 other tests that hit the same len() path)

## Task Commits

Each task was committed atomically:

1. **Task 1: Guard _count_failure_cases against ibis.Table (TDD)** - `cc0aa07` (fix)
2. **Task 2: Verify no regression in broader test suites** - (verification only, no code changes)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `pandera/api/base/error_handler.py` - Added ibis.Table branch before the try/len() block using .count().execute(), wrapped in try/except ImportError
- `tests/backends/narwhals/test_parity.py` - Added test_custom_check_ibis_lazy regression test using IbisData wrapping and ibis.selectors

## Decisions Made
- Used `try: import ibis; if isinstance(...): ...; except ImportError: pass` pattern to keep ibis optional in shared base code
- Test uses `IbisData.table.select(s.across(s.all(), ibis_deferred == 0))` pattern (same as `test_dataframe_level_checks`) to produce a genuine ibis.Table failure_cases — the original plan test with `lambda df: df["a"] > 100` was insufficient as it caused a TypeError before reaching `_count_failure_cases`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test RED phase needed a different check pattern to trigger the actual crash**
- **Found during:** Task 1 (TDD RED step)
- **Issue:** The plan's proposed test `Check(lambda df: df["a"] > 100)` passed immediately — the lambda caused a TypeError in check execution (ibis tables don't support `["a"]` dict-style indexing), so failure_cases was set to a string error message, never reaching the `len()` crash in `_count_failure_cases`
- **Fix:** Rewrote the test check to use `IbisData` wrapping with `ibis.selectors` — `data.table.select(s.across(s.all(), ibis_deferred == 0))` — which matches the pattern in `test_dataframe_level_checks` and correctly produces an `ibis.Table` as failure_cases
- **Files modified:** tests/backends/narwhals/test_parity.py
- **Verification:** Test now properly fails RED with `ExpressionError: Use .count() instead`, then passes GREEN after the fix
- **Committed in:** cc0aa07 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug in test design)
**Impact on plan:** Required changing the test check pattern to actually exercise the crash path. No scope creep — same fix target, same files, same outcome.

## Issues Encountered
None beyond the test design deviation documented above.

## Next Phase Readiness
- Phase 5 UAT blocker resolved — lazy=True custom ibis check validation works end-to-end
- ibis test failure count at 89 (down from 95 at start of plan 06; down from 103 at start of phase 05)
- Remaining 89 ibis failures are pre-existing (not introduced by phase 05 work)
- No remaining blockers for Phase 5 completion

---
*Phase: 05-ibis-registration-and-integration*
*Completed: 2026-03-15*

## Self-Check: PASSED

- FOUND: pandera/api/base/error_handler.py
- FOUND: tests/backends/narwhals/test_parity.py
- FOUND: .planning/phases/05-ibis-registration-and-integration/05-06-SUMMARY.md
- FOUND: cc0aa07 (fix commit)
