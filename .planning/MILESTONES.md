# Milestones

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

