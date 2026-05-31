---
phase: 07-ci-fixes-and-post-review-quick-fixes
plan: 02
subsystem: narwhals-backend
tags: [polish, nits, docs, assert-removal, noxfile, capitalisation]
dependency_graph:
  requires: []
  provides: [NITS-02]
  affects:
    - pandera/backends/narwhals/components.py
    - noxfile.py
    - docs/source/supported_libraries.md
tech_stack:
  added: []
  patterns: []
key_files:
  created: []
  modified:
    - pandera/backends/narwhals/components.py
    - noxfile.py
    - docs/source/supported_libraries.md
decisions:
  - "Trust the uses_pyspark_dtype guard instead of a redundant non-None assert; the ternary binding mirrors the condition exactly"
metrics:
  duration: "~5 minutes"
  completed: "2026-05-25"
  tasks_completed: 3
  tasks_total: 3
---

# Phase 07 Plan 02: Post-Review Nits (NITS-02) Summary

Three mechanical post-review polish edits: remove structurally unreachable `assert native_pyspark_schema is not None` from `check_dtype`, add clarifying inline comments above both `tests/common/` exclusion gates in noxfile.py, and correct three `Pyspark SQL` capitalisation errors to `PySpark SQL` in supported_libraries.md.

## Tasks Completed

### Task 1: Remove redundant non-None assert from check_dtype

**Commit:** 533e4826

Deleted the single line `assert native_pyspark_schema is not None` (formerly line 297) from the `if uses_pyspark_dtype:` block inside `check_dtype` in `pandera/backends/narwhals/components.py`.

The binding at line 284 reads:
```python
native_pyspark_schema = (
    nw.to_native(check_obj).schema if uses_pyspark_dtype else None
)
```

The `if uses_pyspark_dtype:` guard on the next line entering the branch proves the binding is non-None. The assert was structurally unreachable as a failure path and has been removed per the phase decision to "simply trust the guard".

The surrounding ARCH-03 comment block (lines 290-296) and the `pyspark_dtype = native_pyspark_schema[column].dataType` statement immediately below the removed line are unchanged.

### Task 2: Add inline comment to noxfile tests/common/ exclusion

**Commit:** ba13a9ca

Added the comment:
```
# tests/common/ has no pyspark marker — pytest -m pyspark would deselect every test there
```

at two locations in `noxfile.py`:
1. Immediately above `if not session.posargs and extra in ("polars", "ibis"):` in the `unit_tests` session (line ~323)
2. Immediately above `if extra in ("polars", "ibis"):` in the `tests_narwhals_backend` session (line ~389)

Both comments are placed as header lines at the same indentation as the `if` statement below them. No functional code was changed.

### Task 3: Correct "Pyspark SQL" capitalisation in supported_libraries.md

**Commit:** d48b2c5d

Replaced all three `Pyspark SQL` occurrences with `PySpark SQL`:
- Line 26: `{ref}\`Pyspark SQL <native-pyspark>\`` → `{ref}\`PySpark SQL <native-pyspark>\``
- Line 49: `Pyspark SQL <pyspark_sql>` (toctree entry) → `PySpark SQL <pyspark_sql>`
- Line 128: `{ref}\`Pyspark SQL <native-pyspark>\` integrations…` → `{ref}\`PySpark SQL <native-pyspark>\` integrations…`

Sphinx reference labels (`<native-pyspark>`, `<pyspark_sql>`) were not modified — only the display text changed. The file now contains 10 occurrences of `PySpark SQL` (7 pre-existing + 3 corrected) and 0 occurrences of `Pyspark SQL`.

## Verification Commands Passed

```
grep -c "Pyspark SQL" docs/source/supported_libraries.md  → 0
grep -v '^[[:space:]]*#' pandera/backends/narwhals/components.py | grep -c "assert native_pyspark_schema" → 0
grep -B 1 'extra in ("polars", "ibis"):' noxfile.py | grep -c "tests/common/ has no pyspark marker" → 2
python -c "import pandera.backends.narwhals.components; import ast; ast.parse(open('noxfile.py').read())" → exits 0
grep -c "PySpark SQL" docs/source/supported_libraries.md  → 10
grep -c "<native-pyspark>" docs/source/supported_libraries.md  → 2
grep -c "<pyspark_sql>" docs/source/supported_libraries.md  → 1
```

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None.

## Threat Flags

None. All three changes are non-behavioral polish edits with no new network endpoints, auth paths, file access patterns, or schema changes at trust boundaries.

## Self-Check: PASSED

- pandera/backends/narwhals/components.py: FOUND (no assert, guard and subscript preserved)
- noxfile.py: FOUND (2 comments, 2 conditions, valid Python)
- docs/source/supported_libraries.md: FOUND (0 Pyspark SQL, 10 PySpark SQL, refs untouched)
- Commit 533e4826: FOUND
- Commit ba13a9ca: FOUND
- Commit d48b2c5d: FOUND
