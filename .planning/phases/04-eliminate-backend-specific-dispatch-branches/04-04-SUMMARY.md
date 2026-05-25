---
phase: 04-eliminate-backend-specific-dispatch-branches
plan: "04"
subsystem: narwhals-backend
tags:
  - refactoring
  - pyspark
  - extract-method
  - arch-04
dependency_graph:
  requires:
    - "04-01 (base.py run_check / _materialize fix)"
    - "04-02 (_concat_failure_cases narwhals-native concat)"
    - "04-03 (check_dtype schema-driven dispatch)"
  provides:
    - "_handle_pyspark_validation_result method on DataFrameSchemaBackend"
    - "ARCH-04 requirement satisfied"
  affects:
    - "pandera/backends/narwhals/container.py"
tech_stack:
  added: []
  patterns:
    - "Extract-method refactoring: inline PySpark protocol blocks extracted to named method"
    - "TDD: RED/GREEN cycle with unit tests using MagicMock for isolated contract verification"
key_files:
  created: []
  modified:
    - "pandera/backends/narwhals/container.py"
    - "tests/narwhals/test_phase01_arch.py"
decisions:
  - "TDD unit tests use MagicMock rather than real PySpark (no Java runtime in dev env)"
  - "Tests added to tests/narwhals/test_phase01_arch.py alongside existing ARCH tests"
  - "is_pyspark detection preserved in validate() — genuine protocol difference cannot be eliminated"
metrics:
  duration: "~3 minutes"
  completed: "2026-05-25"
  tasks_completed: 1
  files_changed: 2
---

# Phase 04 Plan 04: Extract PySpark Validation Result Handler Summary

**One-liner:** Extract inline PySpark error-setting blocks in `DataFrameSchemaBackend.validate` to `_handle_pyspark_validation_result` method, satisfying ARCH-04 with a docstring explaining the genuine protocol difference.

## What Was Built

Added `_handle_pyspark_validation_result(self, check_obj, error_handler, schema, has_errors: bool)` to `DataFrameSchemaBackend` in `pandera/backends/narwhals/container.py`. The method:

- Sets `check_obj.pandera.errors = error_handler.summarize(schema_name=schema.name)` on the error path (`has_errors=True`)
- Sets `check_obj.pandera.errors = {}` on the success path (`has_errors=False`)
- Returns `check_obj` (the original PySpark frame) in both cases
- Has a 5-line docstring naming: (a) the protocol difference, (b) why it differs from the narwhals standard, (c) the ARCH-04 reference

Replaced two inline `is_pyspark` blocks in `validate()`:
- Error path (former lines 231-236): `elif is_pyspark:` with 3-line inline body → `return self._handle_pyspark_validation_result(..., has_errors=True)`
- Success path (former lines 244-246): `if is_pyspark:` with 2-line inline body → `return self._handle_pyspark_validation_result(..., has_errors=False)`

The `is_pyspark` boolean detection at lines 102-106 is preserved — it remains the legitimate dispatch mechanism.

## TDD Gate Compliance

| Gate | Commit | Notes |
|------|--------|-------|
| RED | 81bad172 | 5 failing tests added to `tests/narwhals/test_phase01_arch.py` |
| GREEN | faf53c76 | Method implemented; all 5 tests pass |
| REFACTOR | N/A | No cleanup needed — implementation matches the research code example exactly |

## Source Assertions (all passing)

- `grep -c "_handle_pyspark_validation_result" container.py` → 3 (1 def + 2 call sites)
- `grep -n "protocol" container.py` → line 260 in docstring ✓
- Inline `check_obj.pandera.errors =` assignments: 0 in `validate()`, 2 in `_handle_pyspark_validation_result` ✓
- `is_pyspark = check_lf.implementation in` survives at line 102 ✓
- `grep -c "elif is_pyspark:\|if is_pyspark:" container.py` → 2 ✓

## Test Results

- `tests/narwhals/` (210 tests): 210 passed, 12 skipped, 4 xfailed ✓
- `tests/polars/ tests/ibis/` (502 tests): 502 passed, 1 skipped, 134 xfailed ✓
- `tests/pyspark/` (PySpark tests): skipped — no Java runtime in dev environment; PySpark tests run only in CI with PySpark nox session

## Deviations from Plan

**1. [Rule 1 - Accepted Limitation] PySpark integration tests could not be run**

- **Found during:** Task 1 acceptance criteria verification
- **Issue:** The acceptance criteria specifies running `PANDERA_USE_NARWHALS_BACKEND=True python -m pytest tests/pyspark/...` but the dev environment has no Java runtime (`Unable to locate a Java Runtime`), causing PySpark tests to fail at session start.
- **Mitigation:** This is expected per STATE.md ("PySpark install in CI may require special nox handling (Java runtime, JAVA_HOME)"). The unit tests for `_handle_pyspark_validation_result` use MagicMock and are fully independent of Java. The method is a pure Python extract-method refactor with no logic changes — the same code that previously passed PySpark CI tests now lives in a named method.
- **Files modified:** N/A
- **Risk:** LOW — behavioral contract is identical; only location of code changed.

## Known Stubs

None — pure refactoring with no new stubs or placeholders.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes. This is a pure extract-method refactor moving existing code to a named method. Trust boundaries are unchanged (see plan threat model: `validate() → _handle_pyspark_validation_result` is an internal method call, no new trust boundary).

## Self-Check: PASSED

- [x] `pandera/backends/narwhals/container.py` modified (method added, validate() updated)
- [x] `tests/narwhals/test_phase01_arch.py` modified (5 new tests added)
- [x] RED commit 81bad172 exists
- [x] GREEN commit faf53c76 exists
- [x] All 5 ARCH-04 unit tests pass
- [x] All 502 polars + ibis integration tests pass
- [x] No accidental file deletions
