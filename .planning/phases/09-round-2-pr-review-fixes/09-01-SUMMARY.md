---
phase: 09-round-2-pr-review-fixes
plan: "01"
subsystem: narwhals-backend-pyspark
tags:
  - pyspark
  - narwhals
  - bugfix
  - exception-handling
  - type-cleanup
dependency_graph:
  requires: []
  provides:
    - "ColumnBackend.validate() SQL-lazy guard (B-01)"
    - "ImportError-only PySpark import guards (M-01)"
    - "Deduped PySparkDtypeInputTypes Union (M-05)"
    - "Simplified supported_types() without dead try/except (M-06)"
  affects:
    - pandera/backends/narwhals/components.py
    - pandera/api/pyspark/types.py
    - pandera/backends/pyspark/register.py
    - pandera/accessors/pyspark_sql_accessor.py
tech_stack:
  added: []
  patterns:
    - "_is_sql_lazy() helper guards .lazy() conversion in ColumnBackend"
key_files:
  created: []
  modified:
    - pandera/backends/narwhals/components.py
    - pandera/api/pyspark/types.py
    - pandera/backends/pyspark/register.py
    - pandera/accessors/pyspark_sql_accessor.py
decisions:
  - "Do not add new import for _is_sql_lazy — already imported on line 14 of components.py"
  - "Do not add AttributeError to except clauses — only ImportError is semantically correct for missing deps"
  - "PySparkConnectDataFrame always bound at module top (either real type or fallback) — try/except in supported_types() was dead code"
metrics:
  duration: "~10 minutes"
  completed: "2026-05-29"
---

# Phase 09 Plan 01: Round 2 PR Review Production-Code Fixes Summary

Fixed four Round 2 PR review findings: B-01 (PySpark .lazy() correctness), M-01 (broad except Exception guards in three PySpark import sites), M-05 (duplicate type entry in PySparkDtypeInputTypes), M-06 (dead try/except in supported_types()).

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Guard .lazy() in ColumnBackend.validate() with _is_sql_lazy | c833f72f | pandera/backends/narwhals/components.py |
| 2 | Narrow three except Exception guards to except ImportError | bae7e827 | pandera/api/pyspark/types.py, pandera/backends/pyspark/register.py, pandera/accessors/pyspark_sql_accessor.py |
| 3 | Dedupe type in PySparkDtypeInputTypes; remove dead ImportError block in supported_types() | 67468044 | pandera/api/pyspark/types.py |

## Findings Fixed

### B-01 — ColumnBackend.validate() unconditional .lazy() (components.py:68)

**File:** `pandera/backends/narwhals/components.py`, line 68

**Root cause:** `if isinstance(check_lf, nw.DataFrame): check_lf = check_lf.lazy()` ran unconditionally, converting PySpark-backed `nw.DataFrame` into `nw.LazyFrame`. This collapsed the `.first()` bounded-row materialization path into `.collect()`, which can crash or unbounded-materialize a PySpark frame.

**Fix:** Added `and not _is_sql_lazy(check_lf)` guard. `_is_sql_lazy` was already imported on line 14 — no new import needed.

```python
# Before
if isinstance(check_lf, nw.DataFrame):
    check_lf = check_lf.lazy()

# After
if isinstance(check_lf, nw.DataFrame) and not _is_sql_lazy(check_lf):
    check_lf = check_lf.lazy()
```

### M-01 — Three bare `except Exception:` guards (three files)

**Files and lines:**
- `pandera/api/pyspark/types.py` line 23 — Spark Connect import block
- `pandera/backends/pyspark/register.py` line 16 — pyspark_connect import block
- `pandera/accessors/pyspark_sql_accessor.py` line 159 — register_connect_dataframe_accessor call

**Fix:** Changed `except Exception:` to `except ImportError:` in all three sites. The comments in all three cite missing `grpcio-status`/Spark Connect deps, which raise `ImportError`. Bare `except Exception:` would silently swallow genuine bugs.

### M-05 — Duplicate `type` entry in PySparkDtypeInputTypes (types.py:71)

**File:** `pandera/api/pyspark/types.py`, line 71 (second occurrence)

**Fix:** Removed the second `type,` from the Union. The Union now lists `type` exactly once at its original position (line 69).

### M-06 — Dead try/except ImportError in supported_types() (types.py:104-108)

**File:** `pandera/api/pyspark/types.py`, lines 104-108

**Root cause:** `PySparkConnectDataFrame` is bound unconditionally at module top — either to the real Spark Connect type or falls back to `pyspark.sql.DataFrame`. The `table_types.append(PySparkConnectDataFrame)` call cannot raise `ImportError` because the name is always defined.

**Fix:** Replaced the `try/except` wrapper with a direct list literal:

```python
# Before
table_types = [PySparkSQLDataFrame]
try:
    table_types.append(PySparkConnectDataFrame)
except ImportError:  # pragma: no cover
    pass

# After
table_types = [PySparkSQLDataFrame, PySparkConnectDataFrame]
```

The `# pragma: no cover` annotation was also removed (it was on the dead `except` branch).

## Verification Results

All plan-level verification checks passed:

1. `grep -n "except Exception" ...` returns no matches (exit=1) in all three PySpark files
2. `grep -n "and not _is_sql_lazy(check_lf)" components.py` returns exactly one match at line 68
3. `grep -cE "^\s*type,\s*$" pandera/api/pyspark/types.py` returns 1
4. `grep -c "# pragma: no cover" pandera/api/pyspark/types.py` returns 0
5. `PANDERA_USE_NARWHALS_BACKEND=True python -c "import pandera"` completes without error
6. `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/test_pyspark_dtypes.py --collect-only -q` collects 116 tests, exit 0

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None.

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or schema changes introduced. Changes are all defensive narrowing of existing exception handling.

## Self-Check: PASSED

- pandera/backends/narwhals/components.py: FOUND
- pandera/api/pyspark/types.py: FOUND
- pandera/backends/pyspark/register.py: FOUND
- pandera/accessors/pyspark_sql_accessor.py: FOUND
- Commit c833f72f: FOUND (Task 1)
- Commit bae7e827: FOUND (Task 2)
- Commit 67468044: FOUND (Task 3)
