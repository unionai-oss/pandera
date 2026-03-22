# Retrospective: Pandera Narwhals Backend

---

## Milestone: v1.0 — Narwhals Backend

**Shipped:** 2026-03-15
**Phases:** 5 | **Plans:** 18 | **Commits:** 71

### What Was Built

- Phase 1 (Foundation): `narwhals_engine.py` with 18 dtype registrations, `NarwhalsData` NamedTuple, `_to_native` helper, test scaffold
- Phase 2 (Check Backend): `NarwhalsCheckBackend` with 14 builtin checks via Expr API, Dispatcher routing, element_wise SQL-lazy guard
- Phase 3 (Column Backend): `NarwhalsSchemaBackend` base + `ColumnBackend` (nullable, unique, dtype, run_checks) with SQL-lazy safe patterns
- Phase 4 (Container Backend + Polars): Full `DataFrameSchemaBackend` validate() pipeline, auto-detection in `register_polars_backends()`, Polars E2E green
- Phase 5 (Ibis): `register_ibis_backends()` with lru_cache, SQL-lazy check_unique via group_by, ibis check dispatch, failure_cases materialization, `_count_failure_cases` ibis guard

### What Worked

- **TDD with xfail stubs**: Creating xfail test stubs in Wave 0 plans before implementation kept scope honest and gave clear pass/fail signal as each plan landed
- **Phased dependency ordering**: Strict phase ordering (engine → checks → column → container → registration) eliminated rework — each phase had stable dependencies
- **Auto-detection over config flags**: The pivot from `use_narwhals_backend` config flag to try/except auto-detection in `register_polars_backends()` (Plan 04-05) was correct — simpler and more Pythonic
- **Direct BACKEND_REGISTRY writes**: Discovering early that `register_backend()` silently no-ops for existing keys led to a cleaner override strategy

### What Was Inefficient

- **Plan 04-05 rework**: Config flag (`use_narwhals_backend`) was designed in 04-02, only to be removed in 04-05. The auto-detection approach should have been the first design
- **Ibis gap closure required 3 extra plans (04-06)**: Phase 5 originally had 3 plans; ibis dispatch and failure_cases required 3 more gap-closure plans (05-04, 05-05, 05-06). Better ibis depth estimation upfront would have collapsed these
- **`narwhals/register.py` dead file**: Created in Phase 4, deleted in Phase 5. Scope could have been cleaner if ibis registration design was decided before Phase 4

### Patterns Established

- **Lazy import pattern for circular imports**: Lazy imports of `NarwhalsData` and `_to_native` inside coerce/try_coerce prevent circular imports — useful for all future narwhals-adjacent code
- **Try/except guard for optional ibis in shared code**: `_count_failure_cases` ibis guard uses `try/except ImportError` — apply to any future shared code touching optional backends
- **xfail(strict=False) for XPASS flexibility**: All ibis stubs used `strict=False` so partially-working behaviors don't break CI prematurely
- **group_by().agg(nw.len()) for SQL-lazy uniqueness**: Canonical pattern for uniqueness checks on SQL-lazy backends (Ibis, PySpark, DuckDB) — replaces collect()+is_duplicated()
- **Dual detection in failure_cases_metadata**: Both `ibis.Table` (lazy) and `pyarrow.Table` (materialized) must be detected separately — ibis DuckDB materializes to pyarrow via narwhals collect()

### Key Lessons

- Design ibis/SQL-lazy paths before implementation, not as gap closures — the lazy contract for failure_cases (ibis.Table vs pyarrow.Table) was discovered late
- Auto-detection is better than config flags for optional backend activation — avoids user-facing config surface area
- lru_cache idempotency on register_*_backends() is essential — multiple schema.validate() calls must not re-register

### Cost Observations

- Phases executed: 5 | Plans: 18 | Sessions: multiple
- No model mix data recorded
- Notable: Phase 5 required 3 unplanned gap-closure plans — about 60% more work than estimated

---

## Cross-Milestone Trends

| Milestone | Phases | Plans | Timeline | Scope Accuracy |
|-----------|--------|-------|----------|----------------|
| v1.0 Narwhals Backend | 5 | 18 | 6 days | ~70% (5 gap-closure plans unplanned) |
