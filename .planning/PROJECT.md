# Pandera Narwhals Backend

## What This Is

A new Narwhals-backed validation engine for pandera that replaces the library-specific backends (pandas, polars, ibis, pyspark) with a single unified implementation powered by [Narwhals](https://github.com/narwhals-dev/narwhals). Narwhals normalizes the dataframe API across libraries, so pandera can validate Polars, Ibis, pandas, and PySpark frames through one shared backend instead of four separate ones. This is NOT a Narwhals-specific user-facing API — users continue to pass their native dataframes (Polars, pandas, etc.) and pandera validates them internally via Narwhals.

The v1.0 milestone shipped a complete Polars + Ibis narwhals backend with auto-detection registration, 18 dtype registrations, 14 builtin checks, full lazy validation support, and closed all known Ibis xfail gaps.

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
- ✓ `failure_cases` always native frame type (not narwhals wrapper) — v1.0

### Active

- [ ] pandas validation working via Narwhals backend (including lazy mode via Narwhals lazy graph)
- [ ] PySpark validation working via Narwhals backend (if feasible)
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

**v1.0 shipped (2026-03-15):** 5 phases, 18 plans, 60 files changed, ~10,900 lines added over 6 days.

Tech stack: `narwhals.stable.v1`, `pandera/backends/narwhals/`, `pandera/engines/narwhals_engine.py`, `tests/backends/narwhals/`

Polars and Ibis backends are now fully registered via narwhals. Known remaining issues from planning:
- `nw.Datetime` parameterized dtypes (`time_unit`, `time_zone`) — resolved during Phase 1
- `drop_invalid_rows` for Ibis uses IbisSchemaBackend delegation (positional-join logic)
- coerce for Ibis is still xfail(strict=True) — intentional feature gate

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
| Ibis `drop_invalid_rows` delegates to `IbisSchemaBackend` | No narwhals abstraction for positional row alignment | ✓ Good |
| `failure_cases` for ibis validation is `pyarrow.Table` | Ibis DuckDB backend returns pyarrow when narwhals LazyFrame.collect() is called | ✓ Good — dual detection in failure_cases_metadata |
| `ibis.Table` detection before `try/len()` in `_count_failure_cases` | Prevents `ExpressionError` from ibis lazy table len() call | ✓ Good — try/except ImportError guard for optional ibis |

---
*Last updated: 2026-03-15 after v1.0 milestone*
