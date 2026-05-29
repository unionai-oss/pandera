---
phase: 10-pr-review-response-documentation-and-code-comment-fixes-from
plan: "02"
subsystem: narwhals-backend
tags: [docs, code-comments, pyspark, narwhals]
dependency_graph:
  requires: []
  provides: [COMMENT-L1, COMMENT-L2]
  affects:
    - pandera/backends/narwhals/container.py
    - pandera/api/narwhals/utils.py
tech_stack:
  added: []
  patterns: []
key_files:
  created: []
  modified:
    - pandera/backends/narwhals/container.py
    - pandera/api/narwhals/utils.py
decisions:
  - "Retained PySpark nw.DataFrame branch in _materialize() as a defensive fallback rather than removing it, since direct callers bypassing container.py could still reach it"
metrics:
  duration: "~10 minutes"
  completed: "2026-05-29T22:35:41Z"
  tasks_completed: 2
  files_modified: 2
---

# Phase 10 Plan 02: Code Comment Clarifications (COMMENT-L1 / COMMENT-L2) Summary

**One-liner:** Added accessor-protocol rationale at three `is_pyspark` dispatch sites in container.py (COMMENT-L1), and updated `_materialize()` docstring plus inline comment to explain PySpark `nw.DataFrame` branch is dead code under normal validation flow (COMMENT-L2).

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | Add accessor-protocol rationale comment to is_pyspark dispatch (COMMENT-L1) | 3a1c7b38 | pandera/backends/narwhals/container.py |
| 2 | Add dead-code clarification to _materialize() PySpark nw.DataFrame branch (COMMENT-L2) | 0e7c72b0 | pandera/api/narwhals/utils.py |

## What Was Done

### Task 1: COMMENT-L1 — container.py accessor-protocol comments

Added three comment additions to `pandera/backends/narwhals/container.py`:

1. **Above `is_pyspark = check_lf.implementation in (...)`** — A 7-line block comment explaining that PySpark uses an accessor-error protocol (`df.pandera.errors`) rather than raising `SchemaErrors`, and that this is a genuine protocol difference (not a backend-capability difference) that cannot be abstracted via `_is_sql_lazy(check_lf)` — because ibis is also SQL-lazy but uses the raise-SchemaErrors protocol. References `_handle_pyspark_validation_result` for the full rationale.

2. **Above `elif is_pyspark:` error-path block** — A 2-line reminder comment noting the accessor protocol and cross-referencing the `is_pyspark` definition above.

3. **Above `if is_pyspark:` success-path block** — A 2-line reminder comment noting that PySpark sets `df.pandera.errors = {}` while other backends just return the parsed frame.

No logic changes were made — all three additions are pure documentation.

### Task 2: COMMENT-L2 — utils.py _materialize() dead-code clarification

Updated `pandera/api/narwhals/utils.py` `_materialize()` with two changes:

1. **Updated docstring** — Replaced the stale "the only context _materialize is called for PySpark" claim with an accurate description of the three branches: (a) `nw.LazyFrame` (Polars, or PySpark after `_to_lazy_nw`), (b) `nw.DataFrame` wrapping SQL-lazy non-PySpark backends (Ibis, DuckDB), and (c) `nw.DataFrame` wrapping PySpark — which is now explicitly documented as a defensive fallback that is effectively dead code under `DataFrameSchemaBackend.validate` because `_to_lazy_nw` in `container.py` eagerly converts every incoming PySpark `nw.DataFrame` to `nw.LazyFrame` before `_materialize` is reached.

2. **Inline comment inside PySpark sub-branch** — Added `# Defensive fallback: see docstring — unreachable under DataFrameSchemaBackend.validate because _to_lazy_nw converts PySpark nw.DataFrame to nw.LazyFrame before _materialize runs.` immediately before the existing PySpark branch implementation comment.

No implementation lines were changed — the `import pyarrow as pa`, `native.first()`, `row.asDict()`, and empty-frame fallback logic are byte-identical.

## Verification

- Both files parse as valid Python (verified via `ast.parse`)
- `pandera/backends/narwhals/container.py` contains `accessor-error protocol`, `_is_sql_lazy` (in a comment), `PySpark error path`, and `PySpark success path`
- `pandera/api/narwhals/utils.py` contains `dead code`, `_to_lazy_nw`, and `Defensive fallback`
- `git diff` for both files shows only comment/docstring additions — no code lines changed
- Both modules import cleanly

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None.

## Threat Flags

None — changes are documentation-only with no new code paths, endpoints, or data flows introduced.

## Self-Check: PASSED

- pandera/backends/narwhals/container.py: FOUND (modified)
- pandera/api/narwhals/utils.py: FOUND (modified)
- Commit 3a1c7b38: FOUND
- Commit 0e7c72b0: FOUND
