# Milestones

## v1.1 Ibis Parity & Lazy-First Architecture (Shipped: 2026-03-25)

**Phases completed:** 9 phases, 22 plans
**Timeline:** 2026-03-15 → 2026-03-25 (10 days)
**Files changed:** 21 Python files, +2,376 / -626 lines
**Git range:** `18f4709` (Phase 1 plan) → `65a7ca8` (dead code cleanup)
**Tests:** 221 passed, 8 skipped, 1 xfailed

**Key accomplishments:**
- Separated `NarwhalsErrorHandler` from base `ErrorHandler` — no ibis imports in base class; unified expression-based `apply()` replaces ~100-line ibis row_number join path (Phases 1–3)
- Rewrote all 14 builtin checks to `nw.Expr` protocol with `Check.native` flag dispatch — `NarwhalsCheckBackend.apply()` reduced to 3 clean branches (element_wise / native=True / native=False) (Phases 3–5)
- Lazy-first throughout: `failure_cases` never materialized in the check loop; single materialization point for scalar bool; `SchemaError.failure_cases` is native `ibis.Table` or `pl.DataFrame` at boundary (Phases 4–6)
- `drop_invalid_rows` reimplemented as pure narwhals `nw.all_horizontal` accumulation — no `IbisSchemaBackend` delegation, works identically for Polars and Ibis (Phase 9)
- Closed two `lazy=True` regressions: per-row `failure_cases` content lost for Polars, `TypeError` crash on bool scalar `failure_cases` (Phase 8)

---

## v1.0 Narwhals Backend (Shipped: 2026-03-15)

**Phases completed:** 5 phases, 18 plans
**Timeline:** 2026-03-09 → 2026-03-15 (6 days)
**Files changed:** 60 files, ~10,900 lines added
**Git range:** `feat(02-01)` → `docs(phase-05)`

**Key accomplishments:**
- Built `narwhals_engine.py` with 18 dtype registrations and coerce/try_coerce via Narwhals Expr API (Phase 1)
- Implemented `NarwhalsCheckBackend` with all 14 builtin checks, Dispatcher routing, and SQL-lazy `element_wise` guard (Phase 2)
- Built `ColumnBackend` (nullable/unique/dtype/run_checks) and `NarwhalsSchemaBackend` shared helpers for all backends (Phase 3)
- Delivered full `DataFrameSchemaBackend` validate() pipeline with auto-detection in `register_polars_backends()` — Polars E2E complete (Phase 4)
- Closed all Ibis xfail gaps: `register_ibis_backends()` with ibis-aware check dispatch, `_count_failure_cases` ibis guard, and failure_cases materialization (Phase 5)

---
