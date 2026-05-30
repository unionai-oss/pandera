---
gsd_state_version: 1.0
milestone: v1.3
milestone_name: Narwhals Backend for PySpark
status: executing
last_updated: "2026-05-30T17:59:45.189Z"
progress:
  total_phases: 15
  completed_phases: 13
  total_plans: 37
  completed_plans: 34
  percent: 87
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-24 after extending v1.3 with PR review phases)

**Core value:** Users can validate any Narwhals-supported dataframe library through a single, consistent backend — reducing maintenance burden and unlocking lazy validation and future library support for free.
**Current focus:** Phase 11 — round-4-pr-review-fixes

## Current Position

Phase: 11 (round-4-pr-review-fixes) — EXECUTING
Plan: 1 of 3
Milestone: v1.3 Narwhals Backend for PySpark — IN PROGRESS (Phases 1-3 complete, 4-6 remaining before merge)
Status: Executing Phase 11

Progress: [█████░░░░░] 50%

## Performance Metrics

**Velocity:**

*No plans completed yet in v1.4*

## Accumulated Context

### Roadmap Evolution

- Phase 7 added: CI Fixes and Post-Review Quick Fixes — revert narwhals container message, fix _spark_env_vars yield, remove redundant assert, noxfile comment, casing
- Phase 8 added: Test Quality Improvements — _cmp_errors in test_pyspark_error, _concat_failure_cases pl_items polars branch, replace source-inspection tests, nw.DataFrame registration comment
- Phase 9 added: Round 2 PR Review Fixes
- Phase 10 added: Round-3 PR Review Fixes — documentation and code-comment fixes from PR review 4393093479 (M1/M2/M3 opt-in note expansions, L1/L2 code comments)
- Phase 11 added: Round-4 PR Review Fixes: SchemaErrors alignment, dead _materialize branch removal, docs/nits

### Decisions

Key decisions from v1.1/v1.2/v1.3 still relevant:

- `polars` imported lazily in `base.py` — polars is optional dep; ibis-only users should not need it
- `nw.from_native(failure_cases, eager_only=False)` unified pattern for failure case counting
- Single materialization point for scalar bool pass/fail; failure_cases stay lazy through check loop
- [Phase 03 v1.2]: ubuntu-only CI for unit-tests-narwhals (Python 3.11/3.12) — narwhals backend is experimental, full OS matrix premature
- [Phase 03 v1.2]: polars+ibis co-installed in narwhals nox session via _testing_requirements augmentation — narwhals extra alone does not list them
- [v1.3]: `_SQL_LAZY_IMPLEMENTATIONS` centralization — PySpark added to set; `_is_sql_lazy()` works uniformly without per-backend probing
- [v1.3]: PySpark excluded from CI on Python 3.12/3.13 — PySpark's maximum supported Python version constraint

### Pending Todos

None.

### Blockers/Concerns

- coerce for Ibis is xfail(strict=True) — intentional feature gate; deferred beyond v1.3
- PR #2339 must not be merged until Phases 4-6 are complete (critical and major review issues outstanding)

## Session Continuity

Last session: 2026-05-26T05:44:13.751Z
Stopped at: Phase 08 context gathered
Resume file: .planning/phases/08-test-quality-improvements/08-CONTEXT.md
