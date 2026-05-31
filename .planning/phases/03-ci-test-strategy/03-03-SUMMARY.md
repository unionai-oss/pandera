---
phase: 03-ci-test-strategy
plan: "03"
subsystem: testing
tags: [ci, github-actions, nox, narwhals, polars, ibis]

# Dependency graph
requires:
  - phase: 03-ci-test-strategy/03-01
    provides: 3-way make_narwhals_frame fixture parametrizing pl.DataFrame/pl.LazyFrame/ibis.Table
  - phase: 03-ci-test-strategy/03-02
    provides: TEST-01 isolation guard in polars/ibis conftest; backend-specific TEST-02 annotations
provides:
  - Documented CI matrix: narwhals not installed in polars/ibis jobs; narwhals+polars+ibis co-installed in narwhals job
  - New unit-tests-narwhals GitHub Actions job running tests/backends/narwhals/ with narwhals extra
  - Nox session support for --extra narwhals mapping to tests/backends/narwhals/ with polars+ibis co-installed
  - conftest.py module docstring explaining the full 3-way CI matrix (TEST-01/02/03)
affects: [ci-tests, nox, test-isolation, narwhals-backend]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "DATAFRAME_EXTRAS set in noxfile.py: narwhals added so it gets the no-pandas-version matrix entry"
    - "test_dir special-case for narwhals -> backends/narwhals (non-flat test directory)"
    - "CI job scope: ubuntu-only + 2 Python versions for experimental narwhals backend (not full OS matrix)"

key-files:
  created: []
  modified:
    - tests/backends/narwhals/conftest.py
    - noxfile.py
    - .github/workflows/ci-tests.yml

key-decisions:
  - "ubuntu-latest only for unit-tests-narwhals job (Python 3.11/3.12) — narwhals backend is experimental, full OS matrix is premature"
  - "polars+ibis co-installed in narwhals nox session via _testing_requirements augmentation — narwhals extra alone does not list them"
  - "test_dir for narwhals is backends/narwhals (not narwhals) — reflects non-flat location under tests/backends/"

patterns-established:
  - "test_dir special-case: use a conditional after the default assignment, not a new parameter"
  - "CI Matrix documentation: cross-reference TEST-XX requirements in both conftest docstrings and workflow YAML comments"

requirements-completed: [TEST-03]

# Metrics
duration: 2min
completed: 2026-04-11
---

# Phase 3 Plan 03: CI Test Strategy Matrix Documentation and Wiring Summary

**narwhals CI job (unit-tests-narwhals) wired with polars+ibis co-install, TEST-01/02/03 matrix documented in conftest and workflow YAML**

## Performance

- **Duration:** 2 min
- **Started:** 2026-04-11T13:45:24Z
- **Completed:** 2026-04-11T13:47:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Replaced one-liner conftest docstring with full CI Matrix documentation covering TEST-01/02/03, the 3-way parametrization, and cross-references to workflow/requirements
- Added narwhals to DATAFRAME_EXTRAS and special-cased test_dir so `--extra narwhals` runs `tests/backends/narwhals/` with polars+ibis co-installed
- Added `unit-tests-narwhals` GitHub Actions job (ubuntu-latest, Python 3.11/3.12) and cross-referencing comment blocks in ci-tests.yml

## Task Commits

Each task was committed atomically:

1. **Task 1: Add CI matrix docstring and narwhals nox session support** - `be5ee142` (feat)
2. **Task 2: Add unit-tests-narwhals CI job with TEST-01/02/03 matrix docs** - `41a67838` (feat)

## Files Created/Modified

- `tests/backends/narwhals/conftest.py` - Module docstring expanded from one-liner to full CI Matrix documentation (TEST-01/02/03)
- `noxfile.py` - Added "narwhals" to DATAFRAME_EXTRAS; added polars+ibis co-install in _testing_requirements; added test_dir special-case for backends/narwhals
- `.github/workflows/ci-tests.yml` - Added TEST-01 comment before unit-tests-dataframe-extras; added unit-tests-narwhals job with TEST-03 CI Matrix comment block

## Decisions Made

- ubuntu-latest only for unit-tests-narwhals — narwhals backend is experimental; adding full OS matrix prematurely adds CI cost without benefit
- polars+ibis are appended to requirements inside `_testing_requirements()` via `.get()` calls — safe even if the extras are renamed later
- test_dir special-case added after the default `test_dir = "base" if extra is None else extra` line, before the modin check — minimal and readable

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 03 (ci-test-strategy) is complete: TEST-01 (isolation guard), TEST-02 (3-way parametrization), TEST-03 (CI matrix documentation + wiring) all satisfied
- The narwhals CI job will begin running tests/backends/narwhals/ on every PR once merged to the target branch
- No blockers for closing the v1.2 milestone

---
*Phase: 03-ci-test-strategy*
*Completed: 2026-04-11*
