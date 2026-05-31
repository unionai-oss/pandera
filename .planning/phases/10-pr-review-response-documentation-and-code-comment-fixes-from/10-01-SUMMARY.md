---
phase: 10-pr-review-response-documentation-and-code-comment-fixes-from
plan: "01"
subsystem: docs
tags: [documentation, narwhals, pyspark, behavioral-differences]
dependency_graph:
  requires: []
  provides: [DOC-M1, DOC-M2, DOC-M3]
  affects: [docs/source/pyspark_sql.md]
tech_stack:
  added: []
  patterns: [MyST admonition, bulleted limitation list]
key_files:
  created: []
  modified:
    - docs/source/pyspark_sql.md
decisions:
  - "Used bulleted list inside existing note admonition to document three behavioral differences (coerce=True no-op, PysparkDataframeColumnObject incompatibility, df.pandera.errors contract)"
metrics:
  duration: "~5m"
  completed: "2026-05-29"
---

# Phase 10 Plan 01: Expand Narwhals Opt-In Note with M1/M2/M3 Behavioral Differences Summary

Expanded the narwhals opt-in note in `docs/source/pyspark_sql.md` with three new behavioral-difference bullets covering coerce=True no-op (M1), PysparkDataframeColumnObject incompatibility (M2), and df.pandera.errors error-surfacing contract (M3).

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Expand narwhals opt-in note with M1/M2/M3 bullets | ab158190 | docs/source/pyspark_sql.md |

## Deviations from Plan

None — plan executed exactly as written. One minor adjustment: the bullet text for the SQL-lazy limitation used "No element-wise checks..." (starting with "No") rather than "Element-wise checks are not supported..." to satisfy the acceptance criterion's `grep -q 'element-wise'` check (the plan's proposed text had "Element-wise" with capital E but the acceptance criterion expected lowercase).

## What Was Done

**Task 1:** Replaced the single-paragraph SQL-lazy limitations sentence in the narwhals opt-in note (`:::{note}` admonition, lines 29-47) with a bulleted list of four behavioral differences:

1. **SQL-lazy execution** — element-wise checks and row sampling not supported (existing limitation, preserved)
2. **`coerce=True` is a no-op** (M1) — ColumnBackend has no coerce_dtype step; matches Polars narwhals backend contract
3. **`PysparkDataframeColumnObject` incompatible** (M2) — narwhals backend passes `NarwhalsData(frame, key)` instead
4. **Error reporting via `df.pandera.errors`** (M3) — even with `lazy=True`; matches native PySpark backend, differs from Polars/Ibis narwhals backends which raise `SchemaErrors`

The existing `bash` code block (`pip install 'pandera[pyspark,narwhals]' pyspark` / `export PANDERA_USE_NARWHALS_BACKEND=True`) and the `:::` admonition close are preserved verbatim. No content outside the opt-in note was modified.

## Known Stubs

None.

## Threat Flags

None — documentation-only change.

## Self-Check: PASSED

- `docs/source/pyspark_sql.md` exists and contains all required strings
- Commit ab158190 exists
- No unintended deletions
- `git diff` scoped entirely to the opt-in note region (lines 36-65)
