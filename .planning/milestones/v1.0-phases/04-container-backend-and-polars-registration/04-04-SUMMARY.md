---
phase: 04-container-backend-and-polars-registration
plan: "04"
subsystem: backend
tags: [narwhals, polars, registration, lru_cache, BACKEND_REGISTRY]

# Dependency graph
requires:
  - phase: 04-container-backend-and-polars-registration-03
    provides: DataFrameSchemaBackend and ColumnBackend in narwhals container/components; register.py stub with register_backend() calls

provides:
  - register_narwhals_backends() using direct BACKEND_REGISTRY writes to override polars backends
  - lru_cache idempotency for repeated validate() calls
  - try/except ImportError guard for partial registration safety
  - check_cls_fqn parameter matching polars register_polars_backends() signature
  - Complete opt-in narwhals backend via config_context(use_narwhals_backend=True)

affects:
  - Phase 5 (Ibis backend may extend register_narwhals_backends() for ibis frame types)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Direct BACKEND_REGISTRY write to override already-registered backends (bypasses register_backend() guard)
    - try/except ImportError wrapping all library imports in registration functions
    - lru_cache on registration function for idempotency across repeated calls

key-files:
  created: []
  modified:
    - pandera/backends/narwhals/register.py

key-decisions:
  - "Direct BACKEND_REGISTRY writes required — register_backend() guard silently no-ops if key already registered (polars backends register on first use)"
  - "All imports inside try/except ImportError — preserves opt-in isolation, allows partial registration"
  - "check_cls_fqn parameter added to match polars register_polars_backends() signature for API consistency"
  - "Check.BACKEND_REGISTRY overridden for pl.LazyFrame — polars registers PolarsCheckBackend during validate, narwhals must override"

patterns-established:
  - "Backend override pattern: write directly to BACKEND_REGISTRY[(Class, type)] = Backend to override already-registered entries"
  - "Registration isolation: all imports inside try/except block, no module-level side effects"

requirements-completed:
  - REGISTER-01
  - REGISTER-02

# Metrics
duration: 5min
completed: 2026-03-14
---

# Phase 4 Plan 04: register_narwhals_backends() with Direct BACKEND_REGISTRY Writes Summary

**register_narwhals_backends() updated to use direct BACKEND_REGISTRY writes, overriding existing polars backends for pl.DataFrame/pl.LazyFrame opt-in narwhals routing**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-14T13:36:00Z
- **Completed:** 2026-03-14T13:41:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- register_narwhals_backends() now correctly overrides polars backends via direct BACKEND_REGISTRY writes
- lru_cache ensures idempotency — calling twice is a no-op (second call never executes body)
- All imports wrapped in try/except ImportError for partial registration safety
- config_context(use_narwhals_backend=True) + schema.validate(pl.DataFrame(...)) routes through narwhals DataFrameSchemaBackend
- Check.BACKEND_REGISTRY overridden for pl.LazyFrame to ensure PolarsCheckBackend doesn't interfere

## Task Commits

Each task was committed atomically:

1. **Task 1: Create register_narwhals_backends() in register.py** - `110856b` (feat)

**Plan metadata:** (docs commit to follow)

## Files Created/Modified
- `pandera/backends/narwhals/register.py` - Updated: replaced register_backend() calls with direct BACKEND_REGISTRY writes; added check_cls_fqn param; wrapped all imports in try/except ImportError; added Check override for pl.LazyFrame

## Decisions Made
- Direct BACKEND_REGISTRY writes required because register_backend() has a guard (`if (cls, type_) not in cls.BACKEND_REGISTRY`) that silently skips registration if polars backends are already registered. Polars backends register on first use (when polars schema is imported or validated), so narwhals registration must unconditionally override.
- check_cls_fqn parameter added to match the polars register_polars_backends() signature for consistency.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added Check.BACKEND_REGISTRY override for pl.LazyFrame**
- **Found during:** Task 1 (analyzing existing register.py implementation)
- **Issue:** Original implementation used Check.register_backend(pl.LazyFrame, NarwhalsCheckBackend) which would silently no-op after polars validate() registers PolarsCheckBackend for pl.LazyFrame. Direct write required for consistency with DataFrameSchema overrides.
- **Fix:** Changed to Check.BACKEND_REGISTRY[(Check, pl.LazyFrame)] = NarwhalsCheckBackend direct write
- **Files modified:** pandera/backends/narwhals/register.py
- **Verification:** All 103 narwhals tests pass; end-to-end smoke test OK
- **Committed in:** 110856b (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 missing critical override consistency)
**Impact on plan:** Auto-fix ensures Check backend override is reliable regardless of test/import ordering. No scope creep.

## Issues Encountered
- `test_narwhals_not_registered_by_default` shows as XFAIL when run with other tests (lru_cache state persists across tests in same session after other tests call register_narwhals_backends()). When run in isolation it XPASS. This is expected behavior given the test design — the xfail+strict=False marks it as acceptable.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Full narwhals backend opt-in path is complete: PANDERA_USE_NARWHALS_BACKEND=True or config_context(use_narwhals_backend=True) routes pl.DataFrame/pl.LazyFrame through narwhals DataFrameSchemaBackend
- Phase 5 (Ibis) will need to extend register_narwhals_backends() or create a separate register_ibis_backends() for ibis frame types
- All 103 narwhals tests pass; no regressions

## Self-Check: PASSED

- FOUND: pandera/backends/narwhals/register.py (modified)
- FOUND commit: 110856b (Task 1)
- End-to-end smoke test: OK

---
*Phase: 04-container-backend-and-polars-registration*
*Completed: 2026-03-14*
