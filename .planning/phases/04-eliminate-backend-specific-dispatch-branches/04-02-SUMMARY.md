---
phase: 04-eliminate-backend-specific-dispatch-branches
plan: "02"
subsystem: narwhals-backend
tags:
  - arch-02
  - failure-cases
  - schema-warning
  - pyspark
dependency_graph:
  requires:
    - 04-01 (nw.Implementation dispatch already in place)
  provides:
    - ARCH-02 SC2 (no silent scalar frame drops in _concat_failure_cases)
  affects:
    - pandera/backends/narwhals/base.py
tech_stack:
  added: []
  patterns:
    - SchemaWarning emission for mixed-backend failure-case drops
    - TODO comment for future SparkSession-mediated Approach B
key_files:
  created: []
  modified:
    - pandera/backends/narwhals/base.py
decisions:
  - "Approach A: emit SchemaWarning naming dropped scalar columns instead of silently skipping"
  - "Added TODO(ARCH-02 follow-up) comment for SparkSession-mediated Approach B conversion"
metrics:
  duration: "14 minutes"
  completed: "2026-05-25"
  tasks_completed: 1
  tasks_total: 1
  files_changed: 1
---

# Phase 04 Plan 02: Eliminate Mixed-Backend Silent Drop — Summary

Emit `SchemaWarning` in `_concat_failure_cases` when scalar `pl.DataFrame` items
from `_build_scalar_failure_case` are dropped in the PySpark path (no SparkSession
available to convert), naming the affected columns explicitly.

## What Was Already Done (by Plan 04-01)

The context note was accurate — Plan 04-01 (`db69c48e`) already completed most of
the ARCH-02 SC2 work:

- `_concat_failure_cases` already dispatches on `nw.Implementation` (not
  `type(item).__module__.startswith("pyspark")` module-string sniffing)
- `_build_lazy_failure_case` already returns narwhals-wrapped `enriched` (no
  `nw.to_native(enriched)` call)
- `failure_cases_metadata` does not call `nw.to_native` on individual
  `failure_case_collection` items before passing to `_concat_failure_cases`

## What This Plan Added

The remaining gap was the silent drop of `pl_items` in the PySpark path: when
both PySpark `nw_items` and scalar polars `pl_items` (from
`_build_scalar_failure_case`, produced by dtype/column-presence errors) were
present, `pl_items` were silently skipped with no user-visible indication.

**Change made** (`pandera/backends/narwhals/base.py`, `_concat_failure_cases`):

- When the PySpark path encounters `pl_items`, it now emits a `SchemaWarning`
  naming the columns whose failure cases are being dropped (extracted from the
  `column` field of each `pl.DataFrame` item).
- Added a `# TODO(ARCH-02 follow-up): SparkSession-mediated conversion
  (Approach B) when a SparkSession reference is available` comment embedded
  in the warning string so the future work is discoverable.
- Updated the docstring to mention the `SchemaWarning` emission.

## Acceptance Criteria Verification

| Check | Result |
|-------|--------|
| `grep -n "startswith.*pyspark\|__module__" base.py \| grep -v '#'` | No matches |
| `grep -n "nw.to_native(enriched)" base.py` | No matches |
| `grep -n "nw.Implementation.PYSPARK" base.py \| grep -v '#'` | 2 matches (lines 78-79) |
| `grep -n "nw.to_native" base.py` (no match in failure_cases_metadata) | All in _concat_failure_cases only |
| `PANDERA_USE_NARWHALS_BACKEND=True python -m pytest tests/polars/ -q` | 361 passed, 1 skipped, 68 xfailed |
| `PANDERA_USE_NARWHALS_BACKEND=True python -m pytest tests/ibis/ -q` | 141 passed, 66 xfailed |
| `tests/pyspark/test_pyspark_error.py` | Not runnable (no Java in CI env) |

Note: `tests/pyspark/test_pyspark_error.py` could not be executed in this
environment because Java is not installed (PySpark requires JVM). The code
change is an additive warning only — it does not modify the return value or
the existing PySpark union path.

## Task Commits

| Task | Description | Commit |
|------|-------------|--------|
| Task 1 | Emit SchemaWarning for mixed-backend failure case drops | `546d369c` |

## Deviations from Plan

### Scope Reduction (Plan Written Before 04-01 Executed)

The plan was written before 04-01 ran. Most of the listed sub-tasks were already
implemented by `db69c48e` (refactor(04-01): SC2):

- `_build_lazy_failure_case` return type change: already done
- `_concat_failure_cases` nw.Implementation dispatch: already done
- `failure_cases_metadata` caller audit: already clean

Only the SchemaWarning for the mixed-backend case was genuinely missing.

### Environmental Constraint (Java Not Available)

`tests/pyspark/test_pyspark_error.py` requires Java/PySpark which is not
installed in the current environment. The verify step could not be run
against the primary regression test file. Polars and ibis test suites (which
exercise `_concat_failure_cases` for their respective backends) pass cleanly.

## Known Stubs

None — no placeholder values or TODO stubs in the production code path (the
TODO comment is in a warning string explaining a known limitation and pointing
to future work, not a stub that blocks the plan's goal).

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes
introduced. The only new surface is `SchemaWarning` text which names column
identifiers already present in `df.pandera.errors` — accepted per T-04-02-02.
