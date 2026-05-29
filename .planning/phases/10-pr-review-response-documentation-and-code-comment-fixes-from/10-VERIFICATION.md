---
phase: 10-pr-review-response-documentation-and-code-comment-fixes-from
verified: 2026-05-29T00:00:00Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
re_verification: false
---

# Phase 10: Round-3 PR Review Fixes — Verification Report

**Phase Goal:** Documentation and code-comment fixes for Round-3 PR review — expand narwhals opt-in note with behavioral differences and add code-comment clarifications for maintainability
**Verified:** 2026-05-29
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `docs/source/pyspark_sql.md` narwhals opt-in note explicitly states that `coerce=True` is a no-op under the narwhals backend (M1) | VERIFIED | Lines 45-49 of `docs/source/pyspark_sql.md` contain "**`coerce=True` is a no-op.**" bullet inside the `:::{note}` admonition at line 29 |
| 2 | `docs/source/pyspark_sql.md` narwhals opt-in note explicitly states that custom checks using `PysparkDataframeColumnObject` are incompatible with the narwhals backend (M2) | VERIFIED | Lines 50-56 contain "**Custom checks using `PysparkDataframeColumnObject` are incompatible.**" with `NarwhalsData(frame, key)` named as the replacement |
| 3 | `docs/source/pyspark_sql.md` narwhals opt-in note explicitly states that error reporting uses `df.pandera.errors`, not `SchemaErrors`, under the narwhals backend (M3) | VERIFIED | Lines 57-62 contain "**Error reporting uses `df.pandera.errors`, not `SchemaErrors`.**" with `lazy=True` explicitly covered |
| 4 | `pandera/backends/narwhals/container.py` `is_pyspark` dispatch carries a comment explaining the accessor-protocol is PySpark-only and cannot be abstracted via `_is_sql_lazy` (L1) | VERIFIED | Lines 102-108 of `container.py` have a 7-line block comment naming the accessor-error protocol and explicitly stating `_is_sql_lazy(check_lf)` cannot be used; lines 238-239 add "PySpark error path" reminder; lines 251-252 add "PySpark success path" reminder |
| 5 | `pandera/api/narwhals/utils.py` `_materialize()` docstring explains the PySpark `nw.DataFrame` branch is effectively dead code because PySpark frames are converted to `nw.LazyFrame` by `_to_lazy_nw` before `_materialize()` is called (L2) | VERIFIED | Lines 43-60 of `utils.py` contain updated docstring with "effectively dead code" and `_to_lazy_nw` named; lines 68-70 contain inline "Defensive fallback" comment inside the PySpark sub-branch |

**Score: 5/5 truths verified**

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `docs/source/pyspark_sql.md` | Narwhals opt-in note with four behavioral-difference bullets (SQL-lazy, coerce=True, PysparkDataframeColumnObject, df.pandera.errors) inside existing `:::{note}` admonition | VERIFIED | All four bullets present at lines 43-62 inside the note opened at line 29; bash code block (pip install / export) preserved at lines 64-67; admonition closed at line 68 |
| `pandera/backends/narwhals/container.py` | Inline comments at `is_pyspark` definition (above line 109) and at error-path/success-path dispatch sites, explaining accessor-protocol rationale | VERIFIED | Block comment at lines 102-108; reminder at lines 238-239 above `elif is_pyspark:`; reminder at lines 251-252 above `if is_pyspark:` |
| `pandera/api/narwhals/utils.py` | Updated docstring and inline comment in `_materialize()` PySpark `nw.DataFrame` branch declaring it dead code and naming `_to_lazy_nw` | VERIFIED | Docstring at lines 43-60; inline comment at lines 68-70 |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `docs/source/pyspark_sql.md` narwhals opt-in note | Existing limitations bullet (SQL-lazy / element-wise / sample= / tail=) | Extended inside same `:::{note}` admonition using bulleted list | WIRED | Original limitation language preserved at line 43-44 ("No element-wise checks… no row sampling via `sample=` / `tail=`"); new bullets follow in the same admonition block |
| `container.py` is_pyspark dispatch (line 109) / error-path (line 240) / success-path (line 253) | `pandera/api/narwhals/utils.py::_is_sql_lazy` | Comment explicitly states `_is_sql_lazy(check_lf)` cannot replace the check | WIRED | Line 106: "cannot be abstracted via `_is_sql_lazy(check_lf)`" — string `_is_sql_lazy` confirmed present in a comment line |
| `pandera/api/narwhals/utils.py` `_materialize()` PySpark `nw.DataFrame` branch | `pandera/backends/narwhals/container.py::_to_lazy_nw` | Docstring names `_to_lazy_nw` as the function that lazifies PySpark frames before `_materialize` | WIRED | Lines 46, 51, 69 of `utils.py` each reference `_to_lazy_nw`; line 69 is the inline comment inside the dead-code branch |

---

### Data-Flow Trace (Level 4)

Not applicable — this phase is documentation-only (Markdown) and code-comment-only changes. No dynamic data rendering artifacts exist.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `container.py` parses as valid Python | `python -c "import ast, pathlib; ast.parse(pathlib.Path('pandera/backends/narwhals/container.py').read_text())"` | exit 0 | PASS |
| `utils.py` parses as valid Python | `python -c "import ast, pathlib; ast.parse(pathlib.Path('pandera/api/narwhals/utils.py').read_text())"` | exit 0 | PASS |

---

### Probe Execution

No probes declared or conventionally applicable for a documentation-and-comments phase.

---

### Requirements Coverage

The requirement IDs DOC-M1, DOC-M2, DOC-M3, COMMENT-L1, COMMENT-L2 are referenced in PLAN frontmatter and defined in ROADMAP.md Phase 10 success criteria. They are **not registered in `.planning/REQUIREMENTS.md`** (which covers the v1.3 milestone functional requirements only — registration, testing, CI, docs, architecture, correctness). These IDs are PR-review-specific documentation/comment findings that were not formally added to REQUIREMENTS.md.

This does not block the phase goal — the success criteria are clearly defined in ROADMAP.md and are all met — but the IDs are orphaned from REQUIREMENTS.md traceability.

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| DOC-M1 | 10-01-PLAN.md | `coerce=True` no-op documented in narwhals opt-in note | SATISFIED | `docs/source/pyspark_sql.md` line 45: "`coerce=True` is a no-op." bullet present in note |
| DOC-M2 | 10-01-PLAN.md | `PysparkDataframeColumnObject` incompatibility documented | SATISFIED | `docs/source/pyspark_sql.md` lines 50-56: incompatibility bullet with `NarwhalsData` present |
| DOC-M3 | 10-01-PLAN.md | `df.pandera.errors` vs `SchemaErrors` contract documented | SATISFIED | `docs/source/pyspark_sql.md` lines 57-62: error-reporting bullet present |
| COMMENT-L1 | 10-02-PLAN.md | Accessor-protocol rationale comment at `is_pyspark` dispatch sites in `container.py` | SATISFIED | Lines 102-108, 238-239, 251-252 of `container.py` contain the three comment additions |
| COMMENT-L2 | 10-02-PLAN.md | Dead-code clarification in `_materialize()` PySpark branch in `utils.py` | SATISFIED | `utils.py` docstring (lines 43-60) and inline comment (lines 68-70) present |

**Note — REQUIREMENTS.md orphan:** DOC-M1/M2/M3 and COMMENT-L1/L2 are not listed as rows in `.planning/REQUIREMENTS.md`. They appear only in the ROADMAP.md Phase 10 section. This is a documentation gap in REQUIREMENTS.md (the IDs cannot be traced from that file to this phase) but does not affect the correctness of the implementation.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | No anti-patterns found in modified files | — | — |

All three modified files (`docs/source/pyspark_sql.md`, `pandera/backends/narwhals/container.py`, `pandera/api/narwhals/utils.py`) contain only substantive comment/documentation additions. No TBD, FIXME, XXX, HACK, or placeholder markers were introduced.

---

### Human Verification Required

None. All success criteria are observable in the codebase via static analysis. Documentation changes are prose-only in an existing admonition; code changes are comment/docstring-only with no behavior modification.

---

### Gaps Summary

No gaps. All five roadmap success criteria for Phase 10 are met with direct codebase evidence.

The only informational note is that DOC-M1/M2/M3 and COMMENT-L1/L2 are not reflected in `.planning/REQUIREMENTS.md` — they exist only in ROADMAP.md. This is an administrative gap (traceability), not a correctness gap.

---

_Verified: 2026-05-29_
_Verifier: Claude (gsd-verifier)_
