# Phase 8: Test Quality Improvements - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-25
**Phase:** 08-test-quality-improvements
**Areas discussed:** _cmp_errors placement, pl_items polars branch fix, ARCH-03 behavioral test design, nw.DataFrame registration for PySpark

---

## _cmp_errors placement

| Option | Description | Selected |
|--------|-------------|----------|
| `tests/pyspark/conftest.py` | Module-level function, auto-importable by all pyspark tests | ✓ |
| Duplicate in `test_pyspark_error.py` | Simple but two copies to maintain | |
| Keep in `test_pyspark_config.py`, import explicitly | Cross-test coupling, unusual pattern | |

**User's choice:** `tests/pyspark/conftest.py`
**Notes:** `TestPySparkConfig._cmp_errors` in `test_pyspark_config.py` should be updated to delegate to the conftest version (remove `@staticmethod` body). Only DATA error assertions with `error` ternaries use `_cmp_errors`; SCHEMA assertions (no CONFIG ternaries) stay as direct equality.

---

## pl_items polars branch fix

| Option | Description | Selected |
|--------|-------------|----------|
| Fix: collect + pl.concat with pl_items | Best for coverage; polars has no SparkSession barrier | ✓ (if pl_items confirmed non-empty) |
| Comment-only: prove pl_items always empty | Simpler changeset; only valid if provably true | Try first |

**User's choice:** Try to prove pl_items is always empty; fix if not.
**Notes:** User asked what approach is most aligned with ibis and PySpark backends. Analysis: PySpark warns+drops (SparkSession barrier), ibis silently drops (same). For polars, no barrier exists — pl_items ARE polars DataFrames, so they can be merged cleanly. Fix: `pl.concat([nw.to_native(nw.concat(nw_items)).collect()] + pl_items)`. No warning needed. User noted the collect-then-concat approach loses laziness — acknowledged as acceptable since `failure_cases` is error-reporting only and already eager in the all-polars path.

---

## ARCH-03 behavioral test design

| Option | Description | Selected |
|--------|-------------|----------|
| Behavioral tests in same file (needs PySpark) | PySpark-gated pass/fail tests for PySpark-dtype path | ✓ |
| Delete file, cover in test_components.py or e2e | Removes dedicated ARCH-03 file | (planner's call on file placement) |
| Keep only the 5th existing test | No PySpark coverage in this file | |

**User's choice:** Whatever guarantees no loss of test coverage and is correct.
**Notes:** User was unclear why there's a PySpark branch in `check_dtype`. Explained: PySpark SQL types (T.IntegerType()) and narwhals dtypes are from different type systems — narwhals can't compare them directly, so `check_dtype` falls back to native PySpark string comparison when `schema.dtype` is a PySpark DataType. Decision: delete 4 source-inspection tests, keep 5th behavioral test, add 2 PySpark-gated behavioral tests (pass/fail dtype check using real PySpark narwhals frame). Planner chooses file placement between `test_arch03_schema_driven_dispatch.py` and `test_components.py`.

---

## nw.DataFrame registration for PySpark

| Option | Description | Selected |
|--------|-------------|----------|
| Add comment explaining why not needed | Documents intentional omission | Only if necessary |
| Register nw.DataFrame defensively | Matches polars exactly; unnecessary for SQL-lazy | |

**User's choice:** Add comment only if necessary; if ibis doesn't have it, probably not needed.
**Notes:** Confirmed ibis registration also omits `nw.DataFrame` with no comment. Decision: no change required — consistent with ibis pattern. Planner should explicitly close this SC as "intentional, consistent with ibis."

---

## Claude's Discretion

- File placement for new PySpark-gated `check_dtype` tests: planner chooses between `test_arch03_schema_driven_dispatch.py` and `tests/narwhals/test_components.py`

## Deferred Ideas

- Further test suite cleanup / deduplication across `tests/narwhals/` — user mentioned intent; not in scope for Phase 8
