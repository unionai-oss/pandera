---
phase: 05-ibis-registration-and-integration
plan: 01
subsystem: testing
tags: [narwhals, ibis, polars, xfail, test-scaffold, parity]

# Dependency graph
requires:
  - phase: 04-container-backend-and-polars-registration
    provides: DataFrameSchemaBackend and register_polars_backends() with narwhals auto-activation
provides:
  - Wave 0 xfail test scaffolds for Phase 5 ibis registration and parity coverage
  - test_ibis_narwhals_auto_activated and test_ibis_backend_is_narwhals stubs in test_container.py (REGISTER-03)
  - test_parity.py with 11 test functions covering TEST-04 Polars/Ibis behavioral parity
affects:
  - 05-ibis-registration-and-integration/05-02 (implements register_ibis_backends with narwhals)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - xfail(strict=False) for ibis-backend-dependent stubs that will pass when registration lands
    - xfail(strict=True) for v2 coerce feature gate to force cleanup when coerce is implemented
    - Polars parity baseline tests (non-xfail) run immediately as regression anchors

key-files:
  created:
    - tests/backends/narwhals/test_parity.py
  modified:
    - tests/backends/narwhals/test_container.py

key-decisions:
  - "xfail(strict=False) used for all ibis stubs — XPASS acceptable as ibis backend may already support behaviors"
  - "test_coerce_ibis uses xfail(strict=True) as v2 coerce feature gate — CI must break when coerce lands to force cleanup"
  - "Polars parity baseline tests run unconditionally (no xfail) — these serve as regression anchors for the narwhals backend"

patterns-established:
  - "Wave 0 scaffolds: write xfail stubs before implementation to establish test contract (Nyquist compliance)"
  - "Separate polars baseline tests from ibis xfail stubs to distinguish working vs pending behavior"

requirements-completed: [REGISTER-03, TEST-04]

# Metrics
duration: 5min
completed: 2026-03-15
---

# Phase 5 Plan 01: Wave 0 Test Scaffolds for Ibis Registration and Parity Summary

**xfail test scaffolds establishing the TEST-04 parity contract and REGISTER-03 ibis narwhals stubs before implementation — 11 tests collected, suite exits 0, 8 ibis tests XPASS indicating ibis backend is more complete than anticipated**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-15T04:52:42Z
- **Completed:** 2026-03-15T04:57:30Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Appended REGISTER-03 ibis stubs (test_ibis_narwhals_auto_activated, test_ibis_backend_is_narwhals) to test_container.py
- Created test_parity.py with 11 test functions covering container validation, strict mode, lazy mode, decorators, DataFrameModel, coerce, and polars baseline
- Discovered that 8 of 9 ibis xfail stubs are already XPASS — ibis backend validates via existing path, only register_ibis_backends() narwhals integration remains

## Task Commits

Each task was committed atomically:

1. **Task 1: Add REGISTER-03 ibis stubs to test_container.py** - `8b02846` (test)
2. **Task 2: Create test_parity.py with TEST-04 xfail stubs** - `612067c` (test)

**Plan metadata:** (docs commit to follow)

## Files Created/Modified
- `tests/backends/narwhals/test_container.py` - Appended test_ibis_narwhals_auto_activated and test_ibis_backend_is_narwhals stubs
- `tests/backends/narwhals/test_parity.py` - New file with 11 parity test functions covering TEST-04

## Decisions Made
- xfail(strict=False) used for all ibis integration stubs so XPASS does not break CI as implementation lands incrementally
- test_coerce_ibis uses strict=True as a v2 feature gate — must fail until coerce is implemented to force marker cleanup
- Polars baseline tests (test_validate_polars_parity, test_lazy_mode_polars_parity) run unconditionally as regression anchors

## Deviations from Plan

None - plan executed exactly as written. The discovery that most ibis tests XPASS (ibis backend already handles these paths) is informational context for Plan 02, not a deviation.

## Issues Encountered

Noteworthy discovery: 8 of 9 ibis-tagged xfail stubs in test_parity.py are already XPASS (the existing ibis backend handles validation through its own path). Only the narwhals-specific registration (register_ibis_backends() emitting Narwhals UserWarning) remains truly unimplemented. This is valuable signal for Plan 02 scoping.

## Next Phase Readiness
- Wave 0 test contract established — Plan 02 can implement register_ibis_backends() with narwhals integration against concrete test targets
- The XPASS stubs in test_parity.py indicate Plan 02's scope may be narrower than expected (primarily the registration function itself)
- test_ibis_narwhals_auto_activated (in test_container.py) remains XFAIL and is the key target for Plan 02

---
*Phase: 05-ibis-registration-and-integration*
*Completed: 2026-03-15*
