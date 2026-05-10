---
gsd_state_version: 1.0
milestone: v1.2
milestone_name: PR Review Cleanup & Test Strategy
status: verifying
stopped_at: Completed 03-03-PLAN.md
last_updated: "2026-04-11T13:55:43.577Z"
last_activity: 2026-04-11
progress:
  total_phases: 3
  completed_phases: 3
  total_plans: 11
  completed_plans: 11
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-29 after v1.2 milestone start)

**Core value:** Users can validate any Narwhals-supported dataframe library through a single, consistent backend — reducing maintenance burden and unlocking lazy validation and future library support for free.
**Current focus:** Phase 03 — ci-test-strategy

## Current Position

Phase: 03
Plan: Not started
Status: Phase complete — ready for verification
Last activity: 2026-04-11

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: —
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

*Updated after each plan completion*
| Phase 01 P03 | 8 minutes | 2 tasks | 1 files |
| Phase 01 P06 | 3 | 2 tasks | 3 files |
| Phase 02 P01 | 1 | 1 tasks | 1 files |
| Phase 02 P02 | 4 | 3 tasks | 10 files |
| Phase 03 P01 | 5 | 2 tasks | 3 files |
| Phase 03 P02 | 10 | 2 tasks | 5 files |
| Phase 03 P03 | 2 | 2 tasks | 3 files |

## Accumulated Context

### Decisions

Key decisions from v1.1 still relevant:

- `polars` imported lazily in `base.py` — polars is optional dep; ibis-only users should not need it
- `nw.from_native(failure_cases, eager_only=False)` unified pattern for failure case counting
- Single materialization point for scalar bool pass/fail; failure_cases stay lazy through check loop
- [Phase 01]: All inner polars imports in base.py guarded with try/except ImportError; functools hoisted to module level
- [Phase 01]: DataFrameSchema import guarded by TYPE_CHECKING — polars not required at runtime for narwhals container backend
- [Phase 01]: from __future__ import annotations enables lazy annotation evaluation so TYPE_CHECKING guard works correctly with PEP 563
- [Phase 02]: Appended Narwhals-backend caveat as continuation of existing sentence — minimal diff, reads naturally in RST; text wraps at 88 chars per project convention
- [Phase 02]: Prose 'Narwhals' always capitalized; code identifiers (imports, variable names, module paths) remain lowercase — established via DOCS-02 sweep
- [Phase 02]: api/narwhals/types.py and utils.py capitalized in Task 3 sweep — plan did not enumerate them but D-04 rule applies to all pandera/*.py prose
- [Phase 03]: Used eager_only=True for pl.DataFrame wrapping in narwhals (matches API requirement for eager frames)
- [Phase 03]: Strategy C class-level docstring annotation preferred for intentionally type-specific tests in test_e2e.py — per-line comments would be too noisy
- [Phase 03]: hasattr guard on cache_clear for register_ibis_backends — function not lru_cache-decorated in main branch; guard prevents AttributeError while remaining correct when lru_cache is present
- [Phase 03]: TEST-01 import-line check uses line.lstrip().startswith() — catches top-of-line imports only; string pandera.backends.narwhals in comments/docstrings not flagged as violation
- [Phase 03]: ubuntu-only CI for unit-tests-narwhals (Python 3.11/3.12) — narwhals backend is experimental, full OS matrix premature
- [Phase 03]: polars+ibis co-installed in narwhals nox session via _testing_requirements augmentation — narwhals extra alone does not list them
- [Phase 03]: test_dir for narwhals is backends/narwhals (not narwhals) — reflects non-flat location under tests/backends/

### Pending Todos

None.

### Blockers/Concerns

- coerce for Ibis is xfail(strict=True) — intentional v2 feature gate; do not address in v1.2
- Custom checks (CHECKS-01) root cause unknown — Phase 1 must investigate before fixing

## Session Continuity

Last session: 2026-04-11T13:47:50.154Z
Stopped at: Completed 03-03-PLAN.md
Resume file: None
