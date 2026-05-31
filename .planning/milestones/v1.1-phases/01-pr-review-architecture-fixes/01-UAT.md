---
status: testing
phase: 01-pr-review-architecture-fixes
source: [01-01-SUMMARY.md, 01-02-SUMMARY.md]
started: 2026-03-21T23:00:00Z
updated: 2026-03-21T23:00:00Z
---

## Current Test

number: 1
name: NarwhalsErrorHandler is importable
expected: |
  `from pandera.api.narwhals.error_handler import NarwhalsErrorHandler` succeeds
  and `issubclass(NarwhalsErrorHandler, ErrorHandler)` is True.
awaiting: user response

## Tests

### 1. NarwhalsErrorHandler is importable
expected: `from pandera.api.narwhals.error_handler import NarwhalsErrorHandler` succeeds and `issubclass(NarwhalsErrorHandler, ErrorHandler)` is True.
result: [pending]

### 2. Base ErrorHandler has no ibis logic
expected: The base `ErrorHandler._count_failure_cases` handles str, list, and None/scalar cases only — no ibis import or ibis.Table handling in `pandera/api/base/error_handler.py`.
result: [pending]

### 3. Narwhals tests pass (all 125)
expected: Running `pytest tests/narwhals/ -x -q` completes with 125 passed, 1 skipped, and 3 xfailed — no failures.
result: [pending]

### 4. Polars LazyFrame validation returns LazyFrame
expected: Validating a Polars LazyFrame against a narwhals schema returns a LazyFrame (not an eagerly-collected DataFrame). The `test_validate_polars_lazyframe` test passes.
result: [pending]

### 5. container.py has no polars import
expected: `pandera/backends/narwhals/container.py` contains no `import polars` line. The backend uses duck-typing (`hasattr(return_type, "collect")`) instead of polars-specific type checks.
result: [pending]

## Summary

total: 5
passed: 0
issues: 0
pending: 5
skipped: 0

## Gaps

[none yet]
