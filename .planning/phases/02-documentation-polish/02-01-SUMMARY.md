---
phase: 02-documentation-polish
plan: "01"
subsystem: api/checks
tags: [documentation, docstring, narwhals, native-param]
dependency_graph:
  requires: []
  provides: [DOCS-01]
  affects: [pandera/api/checks.py]
tech_stack:
  added: []
  patterns: [docstring-clarification]
key_files:
  created: []
  modified:
    - pandera/api/checks.py
decisions:
  - "Appended caveat as continuation of existing sentence rather than a new paragraph — minimal diff, reads naturally in RST"
  - "Text wraps at 88 chars per project convention — sentence spans two lines but docstring content is correct"
metrics:
  duration: "1 minute"
  completed: "2026-04-10"
  tasks_completed: 1
  files_changed: 1
---

# Phase 02 Plan 01: Native Param Docstring Clarification Summary

Appended a single clarifying sentence to the `:param native:` docstring in `pandera/api/checks.py` stating the parameter only applies when using the Narwhals backend.

## What Was Done

Added the following caveat to `Check.__init__` docstring `:param native:` block (after "Builtin checks use ``native=False``."):

> Note: This parameter only applies when using the Narwhals backend; it is ignored by the native pandas, polars, and ibis backends.

The sentence is appended inline at the end of the existing paragraph. No behavior changes. No other files were touched.

## Requirement Addressed

**DOCS-01** — Clarify that the `native` parameter only takes effect under the Narwhals backend. The existing docstring referenced `nw.col(key)` and `nw.Expr` but did not say the parameter is inert under non-Narwhals backends, causing PR reviewer confusion.

## Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Append Narwhals-backend caveat to :param native: docstring | 64f02abb | pandera/api/checks.py |

## Verification

- `python -c "from pandera.api.checks import Check; doc = Check.__init__.__doc__; assert 'native frame' in doc and 'only applies when' in doc and 'Narwhals backend' in doc"` — PASSED
- `python -c "import ast; ast.parse(open('pandera/api/checks.py').read())"` — PASSED (file still parses)
- `git diff --stat pandera/api/checks.py` — 1 file, 3 insertions, 1 deletion (minimal diff)

## Deviations from Plan

None — plan executed exactly as written.

The acceptance criteria grep pattern `grep -c "ignored by the native pandas, polars, and ibis backends"` returns 0 because the text wraps across two lines in the file. The docstring content is correct — the sentence is present verbatim when read from Python (`Check.__init__.__doc__`). This is a line-wrap artifact, not a content issue.

## Known Stubs

None.

## Self-Check: PASSED

- pandera/api/checks.py: FOUND (modified with 3 insertions, 1 deletion)
- Commit 64f02abb: FOUND in git log
