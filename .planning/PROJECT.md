# Pandera Narwhals Backend

## What This Is

A new Narwhals-backed validation engine for pandera that replaces the library-specific backends (pandas, polars, ibis, pyspark) with a single unified implementation powered by [Narwhals](https://github.com/narwhals-dev/narwhals). Narwhals normalizes the dataframe API across libraries, so pandera can validate Polars, Ibis, pandas, and PySpark frames through one shared backend instead of four separate ones. This is NOT a Narwhals-specific user-facing API — users continue to pass their native dataframes (Polars, pandas, etc.) and pandera validates them internally via Narwhals.

The v1.0 milestone shipped a complete Polars + Ibis narwhals backend with auto-detection registration, 18 dtype registrations, 14 builtin checks, full lazy validation support, and closed all known Ibis xfail gaps.

The v1.1 milestone hardened the architecture: unified expression-based check protocol (`nw.Expr` throughout), lazy-first evaluation with a single materialization point for the pass/fail scalar, native `ibis.Table` failure_cases at schema error boundaries, and `drop_invalid_rows` reimplemented as pure narwhals `nw.all_horizontal` accumulation.

The v1.2 milestone addresses PR review 4027330818: unified native type detection, backend isolation, elimination of unnecessary eager execution, cohesive CI test strategy, and documentation polish.

## Core Value

Users can validate any Narwhals-supported dataframe library through a single, consistent backend — reducing maintenance burden and unlocking lazy validation and future library support for free.

## Requirements

### Validated

- ✓ Multi-backend plugin system (`BACKEND_REGISTRY`, `get_backend()`, `register_*.py`) — existing
- ✓ `DataFrameSchema` and `DataFrameModel` APIs (object-based and class-based) — existing
- ✓ Built-in checks registered via `MetaCheck` / `CHECK_FUNCTION_REGISTRY` — existing
- ✓ Engine/DataType abstraction (`pandera/engines/engine.py`) with per-framework implementations — existing
- ✓ Registration pattern (`pandera/backends/polars/register.py`) — existing reference implementation
- ✓ `pandera/backends/narwhals/` — Narwhals-backed dataframe schema backend — v1.0
- ✓ `pandera/backends/narwhals/checks.py` and `builtin_checks.py` — 14 builtin checks via Narwhals Expr API — v1.0
- ✓ `pandera/engines/narwhals_engine.py` — 18 dtype registrations, coerce/try_coerce — v1.0
- ✓ Auto-registration when `narwhals` is installed via `register_polars_backends()` and `register_ibis_backends()` — v1.0
- ✓ Polars and Ibis validation working end-to-end via Narwhals backend — v1.0
- ✓ Unified test suite in `tests/backends/narwhals/` — backend-agnostic tests parameterized per library — v1.0
- ✓ SQL-lazy `element_wise` guard raises `NotImplementedError` for Ibis/PySpark/DuckDB — v1.0
- ✓ v1.2 PR review cleanup: unified `_is_lazy()` utility, backend isolation (TYPE_CHECKING guard), single `collect()` at eager-out boundary, CI matrix (native/narwhals-backend/narwhals jobs), custom checks verified (CHECKS-01) — v1.2
- ✓ `failure_cases` always native frame type (not narwhals wrapper) — v1.0
- ✓ `NarwhalsErrorHandler` subclass with no ibis imports in base `ErrorHandler` — v1.1
- ✓ Expression-based check protocol: all 14 builtin checks return `nw.Expr`; `apply()` uniform across polars and ibis — v1.1
- ✓ Lazy-first evaluation: single materialization for scalar bool; `failure_cases` and `check_output` stay lazy through the check loop — v1.1
- ✓ `SchemaError.failure_cases` is native `ibis.Table` for ibis inputs, `pl.DataFrame` for polars — v1.1
- ✓ `drop_invalid_rows` via `nw.all_horizontal` accumulation — pure narwhals, no `IbisSchemaBackend` delegation — v1.1
- ✓ `lazy=True` regression fixes: per-row failure_cases content preserved, bool scalar `TypeError` crash resolved — v1.1

### Active

**v1.3 Narwhals Backend for PySpark — Phases 4-6 (pre-merge review fixes):**
- [ ] No `is_pyspark`/`is_ibis` dispatch branches in narwhals backend — 4 violations removed or properly abstracted (ARCH-01..04)
- [ ] `_concat_failure_cases` does not silently drop non-PySpark scalar frames (ARCH-02)
- [x] `strict='filter'` applies column filtering for PySpark narwhals success path (CORR-01) — Validated in Phase 05
- [x] `df.pandera.schema` is set after narwhals PySpark validation (CORR-02) — Validated in Phase 05
- [x] `test_pyspark_config.py` band-aid xfails removed (TEST-FIX-01) — Validated in Phase 05
- [ ] PySpark coverage in `tests/narwhals/test_e2e.py` (TEST-E2E-01)
- [ ] Minor nits resolved: CI comment, error message, registration assertions, stacked xfails, `supported_types()` (NITS-01)

**Deferred (future milestones):**
- [ ] pandas validation working via Narwhals backend (including lazy mode via Narwhals lazy graph)
- [ ] `add_missing_columns` parser (FEAT-01)
- [ ] `set_default` for Column fields (FEAT-02)
- [ ] Groupby-based checks via `group_by(...).agg()` pattern (FEAT-04)

### Out of Scope

- Narwhals-specific user-facing API (e.g., `pandera.narwhals.DataFrameSchema`) — Narwhals is internal plumbing, not a user-facing target framework
- Immediate removal of library-native backends — coexist until Narwhals backend is proven
- Strategies (Hypothesis data synthesis) for Narwhals backend — defer to future milestone
- Schema IO (YAML/JSON serialization) for Narwhals backend — defer to future milestone
- GeoDataFrame / modin / dask support via Narwhals — defer based on Narwhals coverage
- `narwhals stable.v2` migration — monitor releases; migrate only when officially stabilized

## Context

**v1.2 milestone phase 03 in progress:** CI infrastructure corrected — `unit-tests-narwhals-backend` job runs tests/polars/ and tests/ibis/ with narwhals active (TEST-03); Python version matrix expanded to 3.10–3.13; broken isolation fixtures removed. xfail markers for narwhals backend gaps are a follow-up.

**v1.1 shipped (2026-03-25):** 9 phases, 22 plans, 21 Python files changed, +2,376 / -626 lines over 10 days. 221 tests passing.

**v1.0 shipped (2026-03-15):** 5 phases, 18 plans, 60 files changed, ~10,900 lines added over 6 days.

Tech stack: `narwhals.stable.v1`, `pandera/backends/narwhals/`, `pandera/engines/narwhals_engine.py`, `tests/backends/narwhals/`

Known remaining items:
- coerce for Ibis is still xfail(strict=True) — intentional feature gate for v2
- `register.py` in narwhals backend is an empty stub (registration is handled by polars/ibis register modules via narwhals auto-detection)

**Reference implementations studied:**
- `pandera/backends/polars/` — model for narwhals container/components/checks
- `pandera/backends/ibis/` — gaps identified and closed via narwhals backend
- `pandera/backends/polars/register.py` — canonical registration pattern replicated

## Constraints

- **Compatibility**: Must not break existing pandera users — library-native backends remain active by default until Narwhals backend is stable
- **Optional dependency**: `narwhals` must be an optional extra (`pandera[narwhals]`); Narwhals backend only activates when installed
- **Priority order**: Polars and Ibis first (done ✓), pandas (next), PySpark (medium priority)
- **Test design**: Tests in `tests/backends/narwhals/` must be backend-agnostic and parameterized with pytest markers to run against each supported library

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use Narwhals as internal engine, not user-facing API | Narwhals unifies implementation, but users pass native frames | ✓ Good — clean separation maintained |
| Register per data type, not globally | Follows existing BACKEND_REGISTRY pattern; enables incremental rollout | ✓ Good — direct BACKEND_REGISTRY writes required (register_backend() silently no-ops for existing keys) |
| `narwhals` as optional extra | Avoids forcing dependency on all users | ✓ Good — auto-detection via try/except in register_polars_backends() |
| Polars/Ibis first, pandas/PySpark after | Higher confidence in non-pandas paths through Narwhals | ✓ Good — both shipped in v1.0 |
| `narwhals_engine.py` with full dtype registration | Research confirmed Narwhals exposes unified dtype system sufficient to replace per-library engines | ✓ Good — 18 dtypes registered, coerce/try_coerce working |
| Use `narwhals.stable.v1` imports everywhere | Insulate from breaking API changes in narwhals | ✓ Good |
| `NarwhalsData` NamedTuple with field `frame` (not `lazyframe`) | Distinguishes from `PolarsData`; consistent with narwhals nomenclature | ✓ Good |
| Auto-detection in `register_polars_backends()` not a separate function | Simplifies activation; avoids extra config flag | ✓ Good — `use_narwhals_backend` config field removed in 04-05 |
| `group_by().agg(nw.len())` for SQL-lazy uniqueness checks | `collect()+is_duplicated()` not possible on SQL-lazy backends | ✓ Good |
| Ibis `drop_invalid_rows` delegates to `IbisSchemaBackend` | No narwhals abstraction for positional row alignment | ⚠️ Revisited — removed in v1.1; replaced with `nw.all_horizontal` accumulation |
| `failure_cases` for ibis validation is `pyarrow.Table` | Ibis DuckDB backend returns pyarrow when narwhals LazyFrame.collect() is called | ⚠️ Revisited — v1.1: `failure_cases` is native `ibis.Table` at boundary; pyarrow path eliminated |
| `ibis.Table` detection before `try/len()` in `_count_failure_cases` | Prevents `ExpressionError` from ibis lazy table len() call | ⚠️ Revisited — v1.1: replaced with unified `try/except TypeError` via `nw.from_native` |
| Expression-based check protocol: all checks return `nw.Expr` | Eliminates ibis row_number join hack; enables uniform `frame.with_columns(expr)` for polars and ibis | ✓ Good — v1.1 |
| Single materialization point for scalar bool pass/fail | Keeps failure_cases lazy through check loop; only the `bool` is evaluated early | ✓ Good — v1.1 |
| `NarwhalsErrorHandler` subclass (not `BaseErrorHandler` directly) | Allows `_count_failure_cases` override without ibis imports in base | ✓ Good — v1.1 |
| `polars` imported lazily in `base.py` (not module-level) | polars is optional dep; ibis-only users shouldn't need it | ✓ Good — v1.1 |

## Current Milestone: v1.3 Narwhals Backend for PySpark

**Goal:** Wire PySpark into the Narwhals backend via registration, add CI coverage, document SQL-lazy limitations, and address all pre-merge review findings before the PR ships (PR #2339).

**Phases 1-3 complete.** Phases 4-6 added after self-review of the open PR:
- Remove/refactor the 4 `is_pyspark` dispatch violations in `base.py`, `components.py`, `container.py`
- Fix `_concat_failure_cases` silent-drop of non-PySpark frames when PySpark items are present
- Fix `strict='filter'` no-op and `df.pandera.schema` regression for PySpark narwhals
- Fix `test_pyspark_config.py` band-aid xfails
- Add PySpark section to `tests/narwhals/test_e2e.py`
- Resolve minor nits (CI comment, error message, registration assertions, stacked xfails, `supported_types()` double-append)

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-25 after Phase 06 complete — PySpark E2E tests added to tests/narwhals/test_e2e.py, all 5 PR #2339 review nits resolved*
