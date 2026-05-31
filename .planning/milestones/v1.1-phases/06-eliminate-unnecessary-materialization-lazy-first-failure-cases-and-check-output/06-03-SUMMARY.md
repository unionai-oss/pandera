---
phase: 06-eliminate-unnecessary-materialization-lazy-first-failure-cases-and-check-output
plan: "03"
subsystem: narwhals
tags: [narwhals, polars, ibis, lazy-frame, failure-cases, materialization, failure_cases_metadata]

# Dependency graph
requires:
  - phase: 06-eliminate-unnecessary-materialization-lazy-first-failure-cases-and-check-output
    plan: "02"
    provides: Unified run_check() + lazy-first subsample() + SchemaError.failure_cases native conversion in components.py
provides:
  - Backend-agnostic failure_cases_metadata() using narwhals ops (nw.lit, nw.concat_str) for lazy/SQL path — no Arrow roundtrip
  - failure_cases_metadata() returns native ibis.Table for ibis inputs and pl.DataFrame for polars-eager inputs
  - _is_lazy_or_sql() helper for detecting polars-lazy or SQL-lazy (ibis) narwhals frames
  - _to_native() boundary unwrap at SchemaError construction in container.py validate()
  - All Phase 6 RED assertions are now GREEN: ibis SchemaErrors.failure_cases is ibis.Table
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lazy/SQL path in failure_cases_metadata: nw.lit(value).alias(name) via .with_columns() — narwhals literal attach, works for polars AND ibis without forcing materialization"
    - "Multi-column lazy failure_cases: nw.concat_str() for 'col=value, col=value' format — cross-backend, stays lazy"
    - "Backend-aware concat: ibis uses functools.reduce + a.union(b), polars uses pl.concat()"
    - "Native conversion at SchemaError boundary in container.py: same pattern as components.py — ibis LazyFrame -> ibis.Table directly, polars LazyFrame -> collect + to_native -> pl.DataFrame"

key-files:
  created: []
  modified:
    - pandera/backends/narwhals/base.py
    - pandera/backends/narwhals/container.py

key-decisions:
  - "_is_lazy_or_sql() helper: isinstance(fc, nw.LazyFrame) OR (isinstance(fc, nw.DataFrame) AND hasattr(nw.to_native(fc), 'execute')) — detects both polars-lazy and ibis SQL-lazy paths"
  - "Lazy/SQL path uses nw.lit() + .with_columns() for metadata columns — narwhals ops only, no polars import required in that branch"
  - "Multi-column lazy failure_cases produce col=value format via nw.concat_str() — intentionally different from polars-eager JSON struct format; no shared contract requires JSON"
  - "nw.to_native(enriched) unwraps ibis path to ibis.Table, polars-lazy path to pl.LazyFrame — backend-aware concat handles the difference"
  - "container.py boundary unwrap mirrors components.py: manual detection of ibis vs polars LazyFrame — _to_native() alone would return pl.LazyFrame for polars which is not collected"
  - "nw.DataFrame.lazy() works for ibis-backed frames (verified) — subsample normalization block in container.py kept unchanged"

patterns-established:
  - "failure_cases_metadata() three-path design: lazy/SQL (narwhals ops only) | eager polars (pl.lit/pl.from_arrow) | scalar (pl.DataFrame)"
  - "Backend-aware concat: hasattr(first, 'union') identifies ibis.Table; else pl.concat()"

requirements-completed:
  - LAZY-FIRST-01

# Metrics
duration: 30min
completed: 2026-03-23
---

# Phase 6 Plan 03: failure_cases_metadata Redesign + Container Boundary Unwrap Summary

**Backend-agnostic failure_cases_metadata() using narwhals ops (nw.lit, nw.concat_str) for lazy/SQL path — ibis.Table returned for ibis inputs with no Arrow roundtrip; SchemaError.failure_cases now native at all construction boundaries**

## Performance

- **Duration:** 30 min
- **Started:** 2026-03-23T22:20:00Z
- **Completed:** 2026-03-23T22:25:54Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- `failure_cases_metadata()` in base.py redesigned with three-path logic: lazy/SQL (narwhals ops only), eager polars (existing pl.lit/pl.from_arrow), scalar (pl.DataFrame) — no Arrow roundtrip in the lazy/SQL path
- Added `_is_lazy_or_sql()` module-level helper that correctly identifies polars-lazy (`nw.LazyFrame`) and SQL-lazy (`nw.DataFrame` wrapping `ibis.Table` via `hasattr(native, "execute")`)
- Multi-column lazy failure_cases produce `"col=value, col=value"` strings via `nw.concat_str()` — cross-backend, stays lazy
- Backend-aware concat: `functools.reduce(a.union(b))` for ibis, `pl.concat()` for polars
- `container.py validate()` now unwraps narwhals failure_cases to native at SchemaError construction — mirrors the existing components.py pattern
- All Phase 6 RED assertions now GREEN: `test_failure_cases_metadata_ibis_returns_ibis_table` (test_components.py), `test_ibis_lazy_failure_cases_is_ibis_table` (test_e2e.py)

## Task Commits

Each task was committed atomically:

1. **Task 1: Redesign failure_cases_metadata() in base.py** - `66c7f6e` (feat)
2. **Task 2: Add _to_native() boundary unwrap in container.py** - `724e9db` (feat)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified

- `pandera/backends/narwhals/base.py` — Added `_is_lazy_or_sql()` helper; redesigned `failure_cases_metadata()` with three-path logic; backend-aware concat at the end
- `pandera/backends/narwhals/container.py` — Added `_to_native` import; SchemaError construction now converts narwhals failure_cases to native using manual detection pattern

## Decisions Made

- **_is_lazy_or_sql() helper placement:** Module-level function above `NarwhalsSchemaBackend` class — makes it reusable without class instantiation and keeps the class method clean
- **Lazy path format difference accepted:** Multi-column lazy path uses `col=value` format (not JSON struct like eager polars). No shared contract requires JSON — both are string representations of failing rows
- **container.py uses manual detection pattern (not plain _to_native()):** `_to_native(nw.LazyFrame)` returns `pl.LazyFrame` (native but uncollected) which would fall to the scalar branch in `failure_cases_metadata()`. Must collect polars-lazy first, then `to_native`. Mirrors components.py exactly.
- **Subsample normalization block unchanged:** `nw.DataFrame.lazy()` verified to work for ibis-backed frames (returns `nw.LazyFrame`) — existing isinstance normalization is correct after Plan 02

## Deviations from Plan

None — plan executed exactly as written. The plan's suggestion of using `_to_native(result.failure_cases)` in container.py was correctly interpreted as needing the manual detection pattern (same as components.py) since plain `_to_native` on polars `nw.LazyFrame` gives uncollected `pl.LazyFrame` which breaks `failure_cases_metadata()`.

## Issues Encountered

- **`_to_native()` insufficient for polars-lazy:** The plan suggests using `_to_native(result.failure_cases)` at the boundary but `nw.to_native(nw.LazyFrame backed by polars)` returns `pl.LazyFrame` — NOT collected. `failure_cases_metadata()` would treat this as a scalar since it's not a narwhals wrapper or ibis.Table. Used the same manual detection pattern as components.py (which was established in Plan 02) to handle this correctly.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 6 complete: all materialization violations from RESEARCH.md are resolved
- The full narwhals backend test suite passes with no new failures (1 pre-existing failure: `test_custom_check_receives_table_and_key` — ibis API changed `DatabaseTable` → `Table`, deferred since Plan 01)
- lazy-first invariant fully enforced: failure_cases_metadata() returns native ibis.Table for ibis inputs, no forced polars conversion anywhere in the pipeline

---
*Phase: 06-eliminate-unnecessary-materialization-lazy-first-failure-cases-and-check-output*
*Completed: 2026-03-23*
