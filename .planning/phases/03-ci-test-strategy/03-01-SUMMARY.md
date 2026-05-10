---
phase: 03-ci-test-strategy
plan: 01
subsystem: testing
tags: [narwhals, pytest, parametrize, polars, ibis, fixture, test-infrastructure]

# Dependency graph
requires:
  - phase: 02-documentation-polish
    provides: Narwhals capitalization and documentation polish complete
provides:
  - Three-way parametrized make_narwhals_frame fixture (polars_eager, polars_lazy, ibis_table)
  - TEST-02 compliance: all narwhals backend checks tested across pl.DataFrame, pl.LazyFrame, and ibis.Table
  - Strategy C annotations for intentionally backend-specific E2E tests
affects: [03-02, 03-03, future narwhals test additions]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "make_narwhals_frame fixture parametrizes across 3 native types via request.param branching"
    - "Backend-specific tests use pytest.skip() with type inspection (nw.to_native + isinstance)"
    - "Intentionally type-specific tests annotated with '# TEST-02: intentionally {type}-specific'"

key-files:
  created: []
  modified:
    - tests/backends/narwhals/conftest.py
    - tests/backends/narwhals/test_e2e.py
    - tests/backends/narwhals/test_lazy_regressions.py

key-decisions:
  - "Used eager_only=True for pl.DataFrame wrapping in narwhals (not eager_or_interchange_only=False)"
  - "Strategy C (class-level docstring annotation) preferred over per-line comments for class-based tests in test_e2e.py"
  - "Existing skip gates in test_checks.py (type inspection via nw.to_native) work correctly with 3-way parametrization — no changes needed"

patterns-established:
  - "Pattern: 3-way fixture covers polars_eager (pl.DataFrame), polars_lazy (pl.LazyFrame), ibis_table (ibis.Table)"
  - "Pattern: backend-specific tests skip via nw.to_native() type check + pytest.skip(reason=...)"
  - "Pattern: TEST-02 annotation in class docstring covers all methods in a backend-specific class"

requirements-completed: [TEST-02]

# Metrics
duration: 5min
completed: 2026-04-11
---

# Phase 03 Plan 01: CI Test Strategy — 3-Way Fixture Parametrization Summary

**`make_narwhals_frame` expanded from 2-way (polars/ibis) to 3-way (polars_eager/polars_lazy/ibis_table), running test_checks.py 108 tests across all three native frame types with 12 explicit skips**

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-11T13:31:15Z
- **Completed:** 2026-04-11T13:36:17Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Expanded `make_narwhals_frame` fixture in `tests/backends/narwhals/conftest.py` from 2-way (polars, ibis) to 3-way (polars_eager, polars_lazy, ibis_table) parametrization
- All 14 builtin check tests now run 3x each (once per frame type) for 42 passing iterations per test group
- Annotated all intentionally backend-specific tests in `test_e2e.py` and `test_lazy_regressions.py` with `TEST-02: intentionally {type}-specific` docstring comments
- All 274 narwhals backend tests pass (12 skipped with explicit reasons, 1 xfailed for coerce)

## Task Commits

Each task was committed atomically:

1. **Task 1: Expand make_narwhals_frame to 3-way parametrization** - `4a283c1b` (feat)
2. **Task 2: Annotate intentionally type-specific tests with TEST-02 comments** - `ac7fe893` (chore)

## Files Created/Modified

- `tests/backends/narwhals/conftest.py` - Expanded fixture params from `["polars", "ibis"]` to `["polars_eager", "polars_lazy", "ibis_table"]` with 3-way `_make()` branching
- `tests/backends/narwhals/test_e2e.py` - Added `TEST-02: intentionally {type}-specific` annotations to all class-level docstrings and standalone test functions
- `tests/backends/narwhals/test_lazy_regressions.py` - Added `TEST-02: intentionally polars_eager-specific` / `ibis_table-specific` annotations to all regression tests

## Decisions Made

- Used `eager_only=True` for `pl.DataFrame` in narwhals wrapping (matches narwhals API requirement for eager frames)
- Existing skip gates in `test_checks.py` work correctly with 3-way parametrization: `test_element_wise_sql_lazy_raises` skips on `polars_eager`/`polars_lazy`, `test_native_true_user_check_ibis` skips on both polars variants, etc.
- Strategy C (class-level docstring) preferred over per-line inline comments in `test_e2e.py` — the class structure already communicates intent and per-line comments would be very noisy
- `test_ignore_na_lazy` and `test_postprocess_lazyframe_no_materialization_polars` correctly run on `polars_eager` as well as `polars_lazy` — the assertions are frame-type-agnostic within the polars family

## Deviations from Plan

None — plan executed exactly as written. Existing tests in `test_checks.py` already had proper skip gates via type inspection (`nw.to_native()` + `isinstance`), so no modifications were needed there beyond the fixture change.

## Issues Encountered

None — the 3-way fixture change worked cleanly. All existing backend-specific skip logic using `type(native).__module__` (for polars variants) and `isinstance(native, ibis.Table)` correctly handled the third parametrization.

## Known Stubs

None — all tests are concrete and functional. The `xfail` for coerce is intentional (v2 feature gate).

## Next Phase Readiness

- 03-01 complete: `make_narwhals_frame` fixture is now the single source of truth for 3-way parametrized narwhals backend tests
- Ready for 03-02 and 03-03 which may add more tests using the updated fixture
- Any new test using `make_narwhals_frame` automatically runs against all 3 frame types

---
*Phase: 03-ci-test-strategy*
*Completed: 2026-04-11*
