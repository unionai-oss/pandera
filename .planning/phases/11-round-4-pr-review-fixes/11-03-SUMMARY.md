---
phase: 11-round-4-pr-review-fixes
plan: "03"
subsystem: docs-and-comments
tags:
  - docs
  - narwhals
  - pyspark
  - capitalization
  - nits
dependency_graph:
  requires:
    - "11-01"
    - "11-02"
  provides:
    - "SE-03: pyspark_sql.md narwhals note updated for SchemaErrors contract"
    - "NIT-01: proper-noun capitalization corrected in touched files"
    - "NIT-02: supported_libraries.md narwhals version 0.32.0"
    - "NIT-04: unnecessary ibis container comment removed"
  affects:
    - docs/source/pyspark_sql.md
    - docs/source/supported_libraries.md
    - pandera/backends/narwhals/base.py
    - pandera/backends/ibis/container.py
    - tests/ibis/test_ibis_container.py
    - tests/pyspark/conftest.py
tech_stack:
  added: []
  patterns:
    - "SchemaErrors contract documented in PySpark narwhals opt-in note"
decisions:
  - "Use dataframe.pandera.errors (not df.pandera.errors) in new SchemaErrors bullet for consistency with native-backend prose at line 79"
metrics:
  duration: "~10 minutes"
  completed_date: "2026-05-30"
  tasks_completed: 3
  files_modified: 6
key_files:
  modified:
    - docs/source/pyspark_sql.md
    - docs/source/supported_libraries.md
    - pandera/backends/narwhals/base.py
    - pandera/backends/ibis/container.py
    - tests/ibis/test_ibis_container.py
    - tests/pyspark/conftest.py
---

# Phase 11 Plan 03: Documentation and Capitalization Nits Summary

Updated `pyspark_sql.md` narwhals opt-in note to document the unified `SchemaErrors` contract and simplified install command; updated `supported_libraries.md` narwhals version reference to 0.32.0; corrected proper-noun capitalization (Narwhals, Polars, Ibis) in narwhals backend comments and tests; removed unnecessary success-signal comment from ibis container.

## Tasks Completed

| Task | Description | Commit |
|------|-------------|--------|
| 1 | Update pyspark_sql.md narwhals note (SE-03, NIT-01, NIT-03) | e567d0c6 |
| 2 | Update supported_libraries.md version reference 0.26.0 -> 0.32.0 (NIT-02) | 9c3bcc92 |
| 3 | Apply NIT-01 capitalization + NIT-04 comment removal across 4 files | 7ea45a10 |

## What Changed

### docs/source/pyspark_sql.md (SE-03, NIT-01, NIT-03)

- Removed the "Error reporting uses `df.pandera.errors`, not `SchemaErrors`" bullet (now obsolete after Plan 01 makes PySpark Narwhals raise SchemaErrors)
- Added new "Unified `SchemaErrors` contract" bullet documenting that PySpark Narwhals now raises `SchemaErrors` like the Polars and Ibis Narwhals backends
- Capitalized "Narwhals" in "Polars Narwhals backend" (coerce=True bullet)
- Removed trailing `pyspark` from `pip install 'pandera[pyspark,narwhals]' pyspark` install command

### docs/source/supported_libraries.md (NIT-02)

- `*new in 0.26.0*` → `*new in 0.32.0*` (top of narwhals note)
- `As of *0.26.0*` → `As of *0.32.0*` (narwhals-backends section prose)

### pandera/backends/narwhals/base.py (NIT-01)

`_concat_failure_cases` docstring and inline comments:
- `(polars LazyFrame, ibis, PySpark)` → `(Polars LazyFrame, Ibis, PySpark)`
- `(eager polars path)` → `(eager Polars path)`
- `For PySpark-backed narwhals frames` → `For PySpark-backed Narwhals frames`
- `For ibis-backed narwhals frames` → `For Ibis-backed Narwhals frames`
- `For polars-backed narwhals LazyFrame` → `For Polars-backed Narwhals LazyFrame`
- `# Separate narwhals-wrapped items from native polars items` → `# Separate Narwhals-wrapped items from native Polars items`
- `# Scalar polars items` → `# Scalar Polars items`
- `"output because scalar polars frames cannot be converted "` → `"output because scalar Polars frames cannot be converted "`
- `# SQL-lazy path (ibis, DuckDB, etc.)` → `# SQL-lazy path (Ibis, DuckDB, etc.)`
- `# All-polars path` → `# All-Polars path`

### pandera/backends/ibis/container.py (NIT-04)

- Removed the comment `# The component validate() not raising is the success signal.` from `run_schema_component_checks`

### tests/ibis/test_ibis_container.py (NIT-01)

- xfail reason: `"narwhals backend"` → `"Narwhals backend"`, `"native ibis backend"` → `"native Ibis backend"`

### tests/pyspark/conftest.py (NIT-01)

- `_cmp_errors` docstring: `"narwhals vs native PySpark"` → `"Narwhals vs native PySpark"`

## Deviations from Plan

### Minor adjustment

**1. [Rule 1 - Bug] New SchemaErrors bullet uses `dataframe.pandera.errors` not `df.pandera.errors`**
- **Found during:** Task 1 acceptance check (`grep -Fc "df.pandera.errors" ... | grep -qx 0`)
- **Issue:** Plan action text said to write new bullet with `df.pandera.errors`; acceptance criteria required 0 occurrences of `df.pandera.errors`. Contradiction.
- **Fix:** New bullet uses `dataframe.pandera.errors` (consistent with the native-backend prose at line 79 which also uses `dataframe.pandera.errors`). This satisfies both the 0-occurrence criterion and the narrative intent.
- **Files modified:** docs/source/pyspark_sql.md

## Verification Results

- `grep -Fc "df.pandera.errors" docs/source/pyspark_sql.md` → `0`
- `grep -c "SchemaErrors" docs/source/pyspark_sql.md` → `2`
- `grep -Fc "pip install 'pandera[pyspark,narwhals]' pyspark" docs/source/pyspark_sql.md` → `0`
- `grep -c "0.26.0" docs/source/supported_libraries.md` → `0`
- `grep -c "0.32.0" docs/source/supported_libraries.md` → `2`
- `grep -Fc "The component validate() not raising is the success signal." pandera/backends/ibis/container.py` → `0`
- `grep -nE "ibis-backed narwhals|polars-backed narwhals" pandera/backends/narwhals/base.py` → 0 lines
- `grep -Fc "Narwhals backend) vs 'not found' (native Ibis backend)" tests/ibis/test_ibis_container.py` → `1`
- `grep -Fc "Error message format varies by backend (Narwhals vs native PySpark)" tests/pyspark/conftest.py` → `1`

## Known Stubs

None.

## Threat Flags

None — changes are documentation, comments, and test strings only; no production code semantics changed.

## Self-Check: PASSED

- docs/source/pyspark_sql.md: FOUND
- docs/source/supported_libraries.md: FOUND
- pandera/backends/narwhals/base.py: FOUND
- pandera/backends/ibis/container.py: FOUND
- tests/ibis/test_ibis_container.py: FOUND
- tests/pyspark/conftest.py: FOUND
- Commit e567d0c6: FOUND
- Commit 9c3bcc92: FOUND
- Commit 7ea45a10: FOUND
