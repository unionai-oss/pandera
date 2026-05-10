---
phase: 02-documentation-polish
plan: "02"
subsystem: documentation
tags: [narwhals, docstrings, comments, capitalization]

requires:
  - phase: 02-01
    provides: DOCS-01 caveat appended to :param native: in api/checks.py (prerequisite for capitalization edit)

provides:
  - Consistent "Narwhals" capitalization (capital N) in all prose comments and docstrings across pandera/
  - Zero lowercase prose "narwhals" references remaining in pandera/**/*.py

affects:
  - any phase editing narwhals backend files (capitalization convention now established)

tech-stack:
  added: []
  patterns:
    - "Prose references to the Narwhals library always capitalize: 'Narwhals backend', 'Narwhals frame', 'Narwhals wrapper', 'Narwhals expression'"
    - "Code identifiers stay lowercase: imports (import narwhals.stable.v1 as nw), variable names (narwhals_data), module paths (pandera.backends.narwhals.*)"

key-files:
  created: []
  modified:
    - pandera/backends/polars/register.py
    - pandera/backends/ibis/register.py
    - pandera/api/checks.py
    - pandera/backends/narwhals/checks.py
    - pandera/backends/narwhals/base.py
    - pandera/backends/narwhals/container.py
    - pandera/backends/narwhals/components.py
    - pandera/engines/narwhals_engine.py
    - pandera/api/narwhals/types.py
    - pandera/api/narwhals/utils.py

key-decisions:
  - "api/narwhals/types.py and api/narwhals/utils.py prose also capitalized — plan listed them as out of scope but sweep found them; Rule 2 applied (consistency correctness)"
  - "':class:`~pandera.api.narwhals.types.NarwhalsData`' dotted path in narwhals_engine.py left unchanged — code identifier rule, not prose"

patterns-established:
  - "Narwhals (capital N) in prose; narwhals (lowercase) in identifiers and import paths"

requirements-completed:
  - DOCS-02

duration: 4min
completed: 2026-04-10
---

# Phase 02 Plan 02: Narwhals Prose Capitalization Summary

**Capitalized all prose "narwhals" → "Narwhals" references across 10 pandera Python files, leaving code identifiers, imports, and module paths untouched.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-10T21:30:56Z
- **Completed:** 2026-04-10T21:34:42Z
- **Tasks:** 3
- **Files modified:** 10

## Accomplishments

- All prose references to the Narwhals library are now consistently capitalized with capital N in docstrings and inline comments
- Code identifiers (imports, variable names, module paths, backtick-quoted symbols) remain lowercase and untouched
- DOCS-02 requirement fully satisfied; zero lowercase prose "narwhals" patterns remain in pandera/**/*.py
- DOCS-01 caveat from plan 02-01 preserved in pandera/api/checks.py

## Task Commits

Each task was committed atomically:

1. **Task 1: Capitalize Narwhals prose in register.py and api/checks.py** - `400430ff` (docs)
2. **Task 2: Capitalize Narwhals prose in narwhals backends and engine** - `fc54aea9` (docs)
3. **Task 3: Sweep and fix remaining prose in api/narwhals/types.py and utils.py** - `eadedba1` (docs)

**Plan metadata:** _(docs commit follows)_

## Files Created/Modified

- `pandera/backends/polars/register.py` - 3 prose edits in docstring (Auto-detects, ColumnBackend, not installed)
- `pandera/backends/ibis/register.py` - same 3 prose edits as polars register
- `pandera/api/checks.py` - 1 edit: "receives a Narwhals expression"
- `pandera/backends/narwhals/checks.py` - 2 edits: raises NotImplementedError, types docstring
- `pandera/backends/narwhals/base.py` - 7 edits: wrappers, narwhals ops, pure Narwhals, Narwhals in/out
- `pandera/backends/narwhals/container.py` - 3 edits: Unwrap, CoreCheckResult carries, returns a Narwhals frame
- `pandera/backends/narwhals/components.py` - 5 edits: wrappers lazy, produce Narwhals, Convert, CoreCheckResult carries, without Narwhals knowledge
- `pandera/engines/narwhals_engine.py` - 2 edits: stays in Narwhals, Narwhals-compatible
- `pandera/api/narwhals/types.py` - 2 edits: Narwhals-backed, Narwhals frames
- `pandera/api/narwhals/utils.py` - 1 edit: Narwhals frame in _to_native docstring

## Decisions Made

The broad sweep in Task 3 found prose occurrences in `pandera/api/narwhals/types.py` and `pandera/api/narwhals/utils.py` that the plan did not enumerate. These were capitalized under the same D-04 rule (Rule 2 — missing critical correctness — consistency applies across all files).

The `:class:~pandera.api.narwhals.types.NarwhalsData` dotted path in `narwhals_engine.py` line 32 was correctly left untouched per the plan's explicit instruction — it is a code identifier, not prose.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Capitalized prose in api/narwhals/types.py and utils.py**
- **Found during:** Task 3 (broad sweep)
- **Issue:** Plan enumerated 8 files but the D-04 capitalization rule applies to all `.py` files under `pandera/`. The broad grep sweep found lowercase prose "narwhals" in two api/narwhals files not listed in the plan.
- **Fix:** Applied same prose capitalization rule: `narwhals-backed` → `Narwhals-backed`, `narwhals frames` → `Narwhals frames`, `narwhals frame` → `Narwhals frame`
- **Files modified:** pandera/api/narwhals/types.py, pandera/api/narwhals/utils.py
- **Verification:** Broad grep sweep returns 0 for all prose patterns after fix
- **Committed in:** eadedba1 (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (Rule 2 — missing critical completeness)
**Impact on plan:** Required for DOCS-02 completeness. No scope creep — these files are part of the narwhals backend subsystem.

## Issues Encountered

None — all files parsed cleanly before and after edits.

## Known Stubs

None — this plan makes only prose changes (comments and docstrings). No runtime behavior is affected.

## Next Phase Readiness

- DOCS-01 and DOCS-02 requirements fully satisfied
- All narwhals backend files now have consistent "Narwhals" capitalization throughout
- Phase 02 documentation polish is complete

## Self-Check: PASSED

- SUMMARY.md file exists
- Commit 400430ff (Task 1) exists
- Commit fc54aea9 (Task 2) exists
- Commit eadedba1 (Task 3) exists

---
*Phase: 02-documentation-polish*
*Completed: 2026-04-10*
