---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_plan: complete
status: complete
stopped_at: Completed 01-pr-review-architecture-fixes / Plan 01-03 complete (phase complete)
last_updated: "2026-03-22T01:49:32.298Z"
last_activity: 2026-03-22 — Plan 01-03 complete (phase complete)
progress:
  total_phases: 1
  completed_phases: 1
  total_plans: 3
  completed_plans: 3
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-15 after v1.0 milestone)

**Core value:** Users can validate any Narwhals-supported dataframe library through a single, consistent backend — reducing maintenance burden and unlocking lazy validation and future library support for free.
**Current focus:** PR Review Architecture Fixes (Phase 01) — cleaning up class hierarchy and architectural issues found during PR review.

## Current Position

Phase: 01-pr-review-architecture-fixes
Current Plan: complete (3 of 3)
Status: Phase 01 complete — all 3 plans executed
Last activity: 2026-03-22 — Plan 01-03 complete (phase complete)

Progress: [██████████] 100%

## Accumulated Context

### Decisions

Previous milestone decisions (v1.0):
- Auto-detection via try/except in `register_polars_backends()` / `register_ibis_backends()` (not config flags)
- Direct BACKEND_REGISTRY writes required to override existing entries
- `group_by().agg(nw.len())` for SQL-lazy uniqueness checks
- Dual ibis.Table / pyarrow.Table detection in `failure_cases_metadata`
- try/except ImportError guard for optional ibis in shared code

Phase 01 decisions:
- NarwhalsErrorHandler uses guarded try/except ImportError for ibis — ibis remains optional dependency
- Fallback to _ErrorHandler._count_failure_cases() in NarwhalsErrorHandler avoids duplicating len()/None logic
- Base ErrorHandler must have zero knowledge of ibis — all backend-specific logic lives in subclasses
- hasattr(return_type, "collect") on the class (not instance) correctly distinguishes lazy (pl.LazyFrame) from eager (pl.DataFrame/ibis.Table) return types without importing polars
- Dynamic Column import via schema.__class__.__module__ check avoids hardcoding polars in a backend-agnostic method
- [Phase 01-pr-review-architecture-fixes]: subsample() receives nw.LazyFrame directly; _to_frame_kind_nw deferred to return statements only — no native round-trips before checks
- [Phase 01-pr-review-architecture-fixes]: drop_invalid_rows branch creates check_obj_parsed locally via _to_frame_kind_nw and returns immediately

### Roadmap Evolution

- Phase 01 added: PR Review Architecture Fixes (4 plans)
- v1.0 milestone complete (5 phases, 18 plans)

### Pending Todos

None.

### Blockers/Concerns

- coerce for Ibis is xfail(strict=True) — intentional v2 feature gate
- `drop_invalid_rows` for Ibis uses IbisSchemaBackend delegation — no narwhals abstraction
- 95 pre-existing ibis test failures unrelated to ErrorHandler changes (ibis backend integration issues)

## Session Continuity

Last session: 2026-03-22T01:49:32.296Z
Stopped at: Completed 01-pr-review-architecture-fixes / Plan 01-03 complete (phase complete)
Resume file: None
