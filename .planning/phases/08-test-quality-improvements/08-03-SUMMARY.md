---
phase: 08-test-quality-improvements
plan: "03"
subsystem: testing
tags: [pyspark, narwhals, pytest, behavioral-tests, schema-driven-dispatch]

# Dependency graph
requires:
  - phase: 04-pyspark-narwhals-components
    provides: ColumnBackend.check_dtype with schema-driven dispatch (ARCH-03)
provides:
  - Behavioral test suite for ARCH-03 schema-driven dispatch (TQ-03 closed)
  - Documentation comment explaining intentional nw.DataFrame omission in PySpark register (TQ-04 closed)
affects: [future test phases, 08-test-quality-improvements]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "PySpark-gated behavioral tests: HAS_PYSPARK guard + pyspark_only skipif marker + inline spark fixture"
    - "comment-only documentation of intentional registration omission citing ibis precedent"

key-files:
  created: []
  modified:
    - tests/narwhals/test_arch03_schema_driven_dispatch.py
    - pandera/backends/pyspark/register.py

key-decisions:
  - "TQ-03: delete 4 brittle inspect.getsource source-inspection tests; add 2 PySpark-gated behavioral tests directly testing ColumnBackend.check_dtype with real PySpark frames"
  - "TQ-04: comment-only in register.py; no nw.DataFrame registration added; cites ibis precedent"

patterns-established:
  - "Behavioral dispatch tests: use SimpleNamespace schema stub + direct ColumnBackend().check_dtype() calls with real frames, not inspect.getsource assertions"
  - "PySpark inline fixtures: copy spark + _spark_env_vars from test_e2e.py when narwhals conftest.py lacks them"

requirements-completed: [TQ-03, TQ-04]

# Metrics
duration: 2min
completed: 2026-05-26
---

# Phase 08 Plan 03: Test Quality Improvements (TQ-03, TQ-04) Summary

**Replaced 4 brittle source-inspection tests in test_arch03 with PySpark-gated behavioral tests that directly exercise ColumnBackend.check_dtype schema-driven dispatch; added register.py comment documenting intentional nw.DataFrame omission citing ibis precedent**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-05-26T14:06:51Z
- **Completed:** 2026-05-26T14:09:05Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Deleted 4 source-inspection tests that asserted internal variable names via the inspect module — they tested implementation state, not behavioral contract
- Removed `import inspect` (no remaining uses)
- Kept existing `test_check_dtype_narwhals_schema_takes_narwhals_engine_path` behavioral test verbatim
- Added `HAS_PYSPARK` guard, `pyspark_only` skipif marker, inline `spark` module-scoped fixture, and `_spark_env_vars` autouse fixture copied from `test_e2e.py`
- Added `test_check_dtype_pyspark_schema_pass` and `test_check_dtype_pyspark_schema_fail` — two PySpark-gated behavioral tests that call `ColumnBackend().check_dtype()` directly with real PySpark frames and pyspark_engine-wrapped dtypes
- Added comment block in `pandera/backends/pyspark/register.py` before `Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend)` documenting why `nw.DataFrame` is intentionally omitted, citing ibis as precedent

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace 4 source-inspection tests with 2 PySpark-gated behavioral tests** - `2e0eddeb` (test)
2. **Task 2: Document intentional nw.DataFrame omission in pandera/backends/pyspark/register.py** - `d5cf95c1` (docs)

## Files Created/Modified
- `tests/narwhals/test_arch03_schema_driven_dispatch.py` - Rewrote to contain only behavioral tests; deleted 4 inspect-based tests, added 2 PySpark-gated tests + fixtures/guards
- `pandera/backends/pyspark/register.py` - Added 4-line comment block before nw.LazyFrame registration explaining nw.DataFrame omission

## Decisions Made
- Followed D-03: placed new PySpark behavioral tests in `test_arch03_schema_driven_dispatch.py` (same file) to preserve topical cohesion with ARCH-03 contract
- Followed D-04: no code change to register.py; comment-only, references ibis precedent by name and contrasts with polars

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- `grep -c 'inspect.getsource'` criterion needed care: plan asked for 0 occurrences but the action spec said to mention the old tests in the docstring. Resolved by rewording the docstring to avoid the literal string `inspect.getsource` while still conveying the historical context.

## Known Stubs

None - no stubs introduced.

## Threat Flags

None - no new network endpoints, auth paths, or schema changes introduced. The spark fixture uses loopback-only Spark config identical to test_e2e.py.

## Next Phase Readiness
- TQ-03 and TQ-04 complete; `test_arch03_schema_driven_dispatch.py` now contains only behavioral tests
- All 3 behavioral tests pass (1 narwhals-engine path + 2 PySpark-engine path)
- Existing PySpark narwhals registration tests still pass

---

## Self-Check: PASSED

- `tests/narwhals/test_arch03_schema_driven_dispatch.py` exists and passes 3 tests (FOUND)
- `pandera/backends/pyspark/register.py` contains nw.DataFrame comment (FOUND)
- Commit `2e0eddeb` exists (FOUND)
- Commit `d5cf95c1` exists (FOUND)

---
*Phase: 08-test-quality-improvements*
*Completed: 2026-05-26*
