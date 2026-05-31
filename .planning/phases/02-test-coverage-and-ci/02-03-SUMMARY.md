---
plan: 02-03
phase: 02-test-coverage-and-ci
status: complete
completed: 2026-05-11
requirements: [TEST-01, TEST-03]
phase_split_recommended: true
follow_on_plan: 02-04
---

# Plan 02-03: PySpark Triage Under Narwhals Backend

## What Was Built

Ran the full `tests/pyspark/` suite under `PANDERA_USE_NARWHALS_BACKEND=True`, produced `02-03-TRIAGE.md` classifying every test outcome, fixed one XPASS(strict) regression introduced by Plan 01's xfail scope, and identified 4 root-cause backend bugs requiring a follow-on plan.

## Key Files

### Created
- `.planning/phases/02-test-coverage-and-ci/02-03-TRIAGE.md` — Per-test triage record: 109 table rows, all 7 RESEARCH.md assumptions resolved, Category B/C/D fully documented with verbatim pytest output

### Modified
- `tests/pyspark/test_pyspark_check.py` — Narrowed `TestUniqueValuesEqCheck::test_failed_unaccepted_datatypes` xfail from class-level to per-parametrization (BooleanType data actually passes; StringType/IntegerType fail as expected)

## Commits
- `f0b5793c` docs(phase-2): record triage outcomes for tests/pyspark under narwhals backend
- `da57d8af` fix(02-03): correct XPASS(strict) for TestUniqueValuesEqCheck::test_failed_unaccepted_datatypes

## Triage Summary

**Run:** `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/ --ignore=test_schemas_on_pyspark_pandas.py -k "spark and not spark_connect"`

**Results:** 59 passed · 27 xfailed · 300 failed · 360 deselected (spark_connect)

**Assumptions resolved:**
- A1 (test_cache_dataframe_requirements passes): confirmed
- A2 (test_cache_dataframe_settings fails): confirmed
- A3 (test_pyspark_dtypes.py mostly passes): **refuted** — all 58 tests failed
- A4 (test_pyspark_error.py mostly passes): **refuted** — 4/5 failed
- A5 (spark_connect pre-existing): confirmed
- A6 (pyspark pulls pandas transitively): confirmed
- A7 (TestCustomCheck tests fail due to API mismatch): confirmed

**Category B (new xfails needed):** 1 XPASS correction (no new xfails; the 300 failures are all Category C backend bugs, not SQL-lazy limitations)

**Category C (backend bugs):** 4 root causes, all [DEFERRED — exceeded per-run cap]:
1. `container.py`: `validate()` raises `SchemaErrors` instead of setting `df.pandera.errors` (affects ~290 tests)
2. `base.py`: `_concat_failure_cases()` uses `pl.concat()` which breaks on PySpark DataFrames
3. `components.py`: dtype string format — narwhals reports `Int64` where PySpark expects `IntegerType()`
4. `base.py`: unnecessary PySpark materialization during validation

## Decisions Made
- PHASE SPLIT RECOMMENDED: Plan 02-04 must be created to fix the 4 Category C bugs before phase 02 closes
- xfail scope narrowed from class-level to per-parametrization for `test_failed_unaccepted_datatypes`

## Self-Check: PASSED

- TRIAGE.md exists with 109 table rows covering all filtered tests ✓
- All 7 RESEARCH.md assumptions (A1-A7) explicitly resolved ✓
- Category B/C/D sections present with verbatim pytest output ✓
- XPASS(strict) corrected in test_pyspark_check.py ✓
- All xfail markers in tests/pyspark/ use CONFIG.use_narwhals_backend + strict=True (AST-verified) ✓
- No os.getenv() in xfail conditions ✓
- Human verification checkpoint approved by user ✓
- PHASE SPLIT RECOMMENDED signal recorded — Plan 02-04 required before phase closes ✓
