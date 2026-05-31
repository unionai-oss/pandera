---
status: complete
phase: 03-documentation
source: 03-01-SUMMARY.md
started: 2026-05-18T00:00:00Z
updated: 2026-05-18T00:10:00Z
---

## Current Test

[testing complete]

## Tests

### 1. PySpark Listed as Supported Backend
expected: Open docs/source/supported_libraries.md and navigate to the narwhals-backends section. PySpark SQL should be named explicitly as a supported SQL-lazy backend alongside Ibis (and Polars). The opening note box and paragraph should both enumerate PySpark SQL.
result: pass

### 2. SQL-lazy Limitations Documented
expected: In both docs/source/pyspark_sql.md (the note block) and docs/source/supported_libraries.md (the Known Gaps section), the documentation should state that PySpark SQL does not support element-wise checks and does not support the sample= / tail= row-sampling parameters. The limitations should match what is documented for Ibis.
result: pass

### 3. Opt-in Instructions Self-Contained
expected: A user reading only docs/source/pyspark_sql.md should be able to determine how to enable the narwhals backend for PySpark without consulting source code. The page should show: (a) install command — pip install 'pandera[pyspark,narwhals]' pyspark, (b) env-var opt-in — PANDERA_USE_NARWHALS_BACKEND=True, (c) programmatic opt-in — pandera.config.CONFIG.use_narwhals_backend = True, and (d) a reference/link to the narwhals-backends page for further details.
result: pass

## Summary

total: 3
passed: 3
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none]
