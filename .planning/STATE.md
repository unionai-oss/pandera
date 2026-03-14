---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: planning
stopped_at: Completed 04-container-backend-and-polars-registration-04-PLAN.md
last_updated: "2026-03-14T13:39:54.681Z"
last_activity: 2026-03-09 — Roadmap created
progress:
  total_phases: 5
  completed_phases: 4
  total_plans: 11
  completed_plans: 11
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-08)

**Core value:** Users can validate any Narwhals-supported dataframe library through a single, consistent backend — reducing maintenance burden and unlocking lazy validation and future library support for free.
**Current focus:** Phase 1 — Foundation

## Current Position

Phase: 1 of 5 (Foundation)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-03-09 — Roadmap created

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: -
- Trend: -

*Updated after each plan completion*
| Phase 01-foundation P01 | 2 | 2 tasks | 7 files |
| Phase 01-foundation P02 | 5 | 1 tasks | 2 files |
| Phase 02-check-backend P01 | 8min | 1 tasks | 3 files |
| Phase 02-check-backend P02 | 2min | 1 tasks | 2 files |
| Phase 02-check-backend P03 | 6min | 1 tasks | 3 files |
| Phase 03-column-backend P01 | 5min | 1 tasks | 1 files |
| Phase 03-column-backend P02 | 3min | 2 tasks | 3 files |
| Phase 04-container-backend-and-polars-registration P01 | 5min | 1 tasks | 1 files |
| Phase 04-container-backend-and-polars-registration P02 | 4min | 2 tasks | 2 files |
| Phase 04-container-backend-and-polars-registration P03 | 8min | 1 tasks | 3 files |
| Phase 04-container-backend-and-polars-registration P04 | 5min | 1 tasks | 1 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Pre-Phase 1]: Use `narwhals.stable.v1` (not bare `narwhals`) to insulate from breaking API changes
- [Pre-Phase 1]: Dtype engine (`narwhals_engine.py`) is the Phase 1 priority — it blocks all coercion and dtype-check code
- [Pre-Phase 1]: Narwhals backend must be strictly opt-in; importing backend code must never side-effect native backend registrations
- [Phase 01-foundation]: Use narwhals.stable.v1 imports in all narwhals API code
- [Phase 01-foundation]: NarwhalsData uses field name frame (not lazyframe) to distinguish from Polars PolarsData
- [Phase 01-foundation]: Lazy imports of NarwhalsData and _to_native inside coerce/try_coerce prevent circular imports
- [Phase 01-foundation]: from_parametrized_dtype return type is Any (not forward ref string) to avoid NameError in get_type_hints
- [Phase 01-foundation]: narwhals_engine.py NOT imported from any __init__.py — maintained strict opt-in isolation
- [Phase 02-check-backend]: Both NarwhalsCheckBackend and builtin_checks imports in conftest fixture are guarded with try/except ImportError so autouse=True fixture does not break dtype tests before checks.py exists
- [Phase 02-check-backend]: inspect.signature() on partial correctly resolves free params for builtin vs user-defined routing
- [Phase 02-check-backend]: test_builtin_check_routing xfail changed to strict=False — depends on builtin_checks.py from Plan 02-03
- [Phase 02-check-backend]: narwhals Expr uses Python comparison operators (==, !=, >, >=, <, <=) not .eq/.ne/.gt/.ge/.lt/.le methods
- [Phase 02-check-backend]: NarwhalsCheckBackend.apply() detects builtins via Dispatcher._function_registry[NarwhalsData] lookup
- [Phase 02-check-backend]: ibis narwhals backend returns nw.DataFrame (not LazyFrame); materialization uses nw.to_native().execute()
- [Phase 03-column-backend]: SimpleNamespace schema stub used for tests — decouples test scaffold from Column API before it is stable
- [Phase 03-column-backend]: NarwhalsCheckBackend registered for both nw.LazyFrame (Polars) and nw.DataFrame (Ibis) so run_checks dispatches correctly for both backends
- [Phase 03-column-backend]: failure_cases in check_dtype is str(nw_dtype) scalar — consistent with Polars backend convention
- [Phase 04-container-backend-and-polars-registration]: Lazy imports inside test bodies prevent collection failure before modules exist — used in test_container.py for narwhals container/register imports
- [Phase 04-container-backend-and-polars-registration]: All xfail stubs use strict=False so XPASS stubs don't break CI as implementation lands incrementally
- [Phase 04-container-backend-and-polars-registration]: failure_cases_metadata handles None check_output gracefully with null-filled index Series
- [Phase 04-container-backend-and-polars-registration]: use_narwhals_backend is additive to PanderaConfig with opt-in env var PANDERA_USE_NARWHALS_BACKEND=True
- [Phase 04-container-backend-and-polars-registration]: Native pl.LazyFrame passed to schema_component.validate() via _to_native() — Column backend registered for pl.LazyFrame, not nw.LazyFrame
- [Phase 04-container-backend-and-polars-registration]: Check registered for nw.LazyFrame and nw.DataFrame in register_narwhals_backends() — ColumnBackend wraps to narwhals internally requiring Check dispatch for nw types
- [Phase 04-container-backend-and-polars-registration]: check_dtype two-pass strategy: narwhals_engine.Engine.dtype() first, then native polars dtype fallback — handles polars_engine Column schemas (e.g. Column(pl.Int64))
- [Phase 04-container-backend-and-polars-registration]: Direct BACKEND_REGISTRY writes required in register_narwhals_backends() — register_backend() guard silently no-ops if key already registered
- [Phase 04-container-backend-and-polars-registration]: All imports inside try/except ImportError in register_narwhals_backends() for partial registration safety and opt-in isolation

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 1]: `nw.Datetime` parameterized dtypes (`time_unit`, `time_zone`) need careful handling — verify during Phase 1
- [Phase 2]: `unique_values_eq` forces materialization on lazy frames; confirm this does not conflict with `lazy=True` error handler
- [Phase 5]: `drop_invalid_rows` for Ibis requires positional alignment strategy; no narwhals abstraction exists — needs explicit design

## Session Continuity

Last session: 2026-03-14T13:39:54.676Z
Stopped at: Completed 04-container-backend-and-polars-registration-04-PLAN.md
Resume file: None
