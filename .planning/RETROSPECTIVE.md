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

## Milestone: v1.1 — Ibis Parity & Lazy-First Architecture

**Shipped:** 2026-03-25
**Phases:** 9 | **Plans:** 22 | **Commits:** 147

### What Was Built

- Phase 1–2 (Architecture): `NarwhalsErrorHandler` subclass, ibis removed from base `ErrorHandler`, polars coupling removed from container, `check_nullable` and `check_dtype` refactored to narwhals-only ops
- Phase 3 (Dispatch): `Check.native` flag, `NarwhalsCheckBackend.apply()` rewritten to 3 explicit branches (element_wise / native=True / native=False), ibis delegation removed from `__call__`
- Phase 4–5 (Lazy + Expr): Always-lazy `failure_cases` in check loop; all 14 builtin checks rewritten to `nw.Expr` protocol; `apply()` reduced from ~100 lines to ~30 via uniform `frame.with_columns(expr)` — no ibis row_number join
- Phase 6 (Materialization): Single materialization point for scalar bool; `failure_cases_metadata` backend-agnostic via narwhals ops; `SchemaError.failure_cases` native at boundary (`ibis.Table` or `pl.DataFrame`)
- Phase 7 (Hygiene): xfail promotions, docstring updates, dead code removed, ROADMAP reconciled
- Phase 8 (Regressions): Two surgical fixes for `lazy=True`: per-row failure_cases content lost (unified `nw.from_native` rewrap), bool scalar `TypeError` crash (`try/except TypeError` in `_count_failure_cases`)
- Phase 9 (drop_invalid_rows): `apply()` returns `nw.Expr` directly; `drop_invalid_rows` uses `nw.all_horizontal` on accumulated exprs — pure narwhals, works for polars and ibis identically

### What Worked

- **Audit-driven gap closure**: Running `gsd:audit-milestone` after Phase 7 caught MISSING-01 and MISSING-02 before shipping — without it, the lazy=True regressions would have gone undetected
- **TDD RED→GREEN discipline**: Phase 8 wrote failing regression tests first, then fixed. Would have been hard to diagnose without the failing test as a precise reproduction
- **Phase 9 nw.Expr accumulation design**: Deferring `failure_cases` reconstruction to after the check loop (storing `nw.Expr`, reconstructing on error) eliminated the wide-table allocation during normal validation
- **Integration checker confirming cross-phase wiring**: Caught the dead code branch in `failure_cases_metadata` and confirmed `err.data` was always `None`, enabling safe removal in tech debt cleanup

### What Was Inefficient

- **Milestone version mislabeled**: This milestone ran under `v1.0` in planning files — should have been `v1.1` from the start. Required renaming at completion
- **Phase 8 was unplanned**: Two regressions discovered post-Phase 7 audit required an entire gap-closure phase. Better end-to-end `lazy=True` testing earlier (Phase 4–6) would have caught these
- **Dead code in `failure_cases_metadata`**: The unreachable `err.data is not None` block survived through Phase 9 despite being obviously dead. Should have been caught during Phase 9 code review

### Patterns Established

- **`nw.Expr` as check return type**: All narwhals builtin checks now return `nw.Expr`; this is the canonical protocol for native=False checks going forward
- **Lazy import for optional polars**: `import polars as pl` inside branches rather than at module level — apply to any narwhals module that conditionally uses polars
- **`err.data = None` is invariant**: `ErrorHandler.collect_error()` unconditionally nulls `err.data` — never try to use it for deferred reconstruction
- **Post-milestone audit before archiving**: Running `gsd:audit-milestone` caught 2 critical regressions; always audit before calling a milestone complete

### Key Lessons

- Design the `lazy=True` end-to-end flow before implementing individual phases — regressions MISSING-01 and MISSING-02 were caused by phase-by-phase changes that each looked correct locally
- Milestone version numbering matters — set it correctly at kickoff, not at completion
- The `nw.Expr` deferral pattern (store expr, reconstruct failure_cases on error raise) is more efficient than wide-table allocation and more robust across backends

### Cost Observations

- Phases: 9 | Plans: 22 | Commits: 147 | Timeline: 10 days
- 2 unplanned phases (Phase 8 gap closure, Phase 9 added mid-milestone)
- Notable: Scope grew ~55% from original 5-phase plan (PR review + arch + lazy + expr + materialization)

---

## Cross-Milestone Trends

| Milestone | Phases | Plans | Timeline | Scope Accuracy |
|-----------|--------|-------|----------|----------------|
| v1.0 Narwhals Backend | 5 | 18 | 6 days | ~70% (5 gap-closure plans unplanned) |
| v1.1 Ibis Parity & Lazy-First Architecture | 9 | 22 | 10 days | ~55% (2 unplanned phases, scope grew from 5→9) |
