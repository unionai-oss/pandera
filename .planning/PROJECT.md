# Pandera Narwhals Backend

## What This Is

A new Narwhals-backed validation engine for pandera that replaces the library-specific backends (pandas, polars, ibis, pyspark) with a single unified implementation powered by [Narwhals](https://github.com/narwhals-dev/narwhals). Narwhals normalizes the dataframe API across libraries, so pandera can validate Polars, Ibis, pandas, and PySpark frames through one shared backend instead of four separate ones. This is NOT a Narwhals-specific user-facing API — users continue to pass their native dataframes (Polars, pandas, etc.) and pandera validates them internally via Narwhals.

## Core Value

Users can validate any Narwhals-supported dataframe library through a single, consistent backend — reducing maintenance burden and unlocking lazy validation and future library support for free.

## Requirements

### Validated

- ✓ Multi-backend plugin system (`BACKEND_REGISTRY`, `get_backend()`, `register_*.py`) — existing
- ✓ `DataFrameSchema` and `DataFrameModel` APIs (object-based and class-based) — existing
- ✓ Built-in checks registered via `MetaCheck` / `CHECK_FUNCTION_REGISTRY` — existing
- ✓ Engine/DataType abstraction (`pandera/engines/engine.py`) with per-framework implementations — existing
- ✓ Registration pattern (`pandera/backends/polars/register.py`) — existing reference implementation

### Active

- [ ] `pandera/backends/narwhals/` — Narwhals-backed dataframe schema backend
- [ ] `pandera/backends/narwhals/checks.py` and `builtin_checks.py` — check execution via Narwhals
- [ ] Dtype handling for the Narwhals backend — approach TBD (see Key Decisions)
- [ ] `pandera/backends/narwhals/register.py` — registration of Narwhals backend for Polars, Ibis, pandas, PySpark data types
- [ ] Auto-registration when `narwhals` is installed; opt-in via `pandera.use_backend("narwhals")`
- [ ] Unified test suite in `tests/backends/narwhals/` — backend-agnostic tests parameterized per library via pytest markers
- [ ] Polars and Ibis validation working end-to-end via Narwhals backend
- [ ] pandas validation working via Narwhals backend (including lazy mode via Narwhals lazy graph)
- [ ] PySpark validation working via Narwhals backend (if feasible)

### Out of Scope

- Narwhals-specific user-facing API (e.g., `pandera.narwhals.DataFrameSchema`) — Narwhals is internal plumbing, not a user-facing target framework
- Immediate removal of library-native backends — coexist until Narwhals backend is proven
- Strategies (Hypothesis data synthesis) for Narwhals backend — defer to future milestone
- Schema IO (YAML/JSON serialization) for Narwhals backend — defer to future milestone
- GeoDataFrame / modin / dask support via Narwhals — defer based on Narwhals coverage

## Context

Pandera currently has four separate library-specific backends (pandas, polars, ibis, pyspark), each implementing the same validation logic in library-specific terms. This creates significant duplication and maintenance burden. The Ibis backend has ~15+ `NotImplementedError` stubs. The Narwhals backend is proposed by the project creator (Niels) to incrementally unify these.

**Reference implementations to study:**
- `pandera/backends/polars/` — most complete non-pandas backend; closest model for Narwhals
- `pandera/backends/ibis/` — has known gaps (`NotImplementedError` for coerce, unique, sampling); shows where Narwhals can help
- `pandera/backends/polars/register.py` — canonical registration pattern to replicate
- `pandera/engines/polars_engine.py` — reference for dtype engine implementation

**Narwhals integration pattern (as proposed by Niels):**
- Introduce `pandera.backends.narwhals` modules
- Register the backend for each supported library type (e.g., `pl.DataFrame`, `pd.DataFrame`, ibis `Table`) — but only if `narwhals` is installed
- Allow explicit opt-in via `pandera.use_backend("narwhals")`
- Eventually phase out library-native backends once confidence is established

**Lazy validation:** Narwhals supports lazy evaluation (`.lazy()` / `.collect()`), enabling deferred validation of pandas DataFrames — currently not possible with the pandas backend.

## Constraints

- **Compatibility**: Must not break existing pandera users — library-native backends remain active by default until Narwhals backend is stable
- **Optional dependency**: `narwhals` must be an optional extra (`pandera[narwhals]`); Narwhals backend only activates when installed
- **Priority order**: Polars and Ibis first (high priority), pandas (high, but fallback to native backend if gaps exist), PySpark (medium priority, skip if not working)
- **Test design**: Tests in `tests/backends/narwhals/` must be backend-agnostic and parameterized with pytest markers to run against each supported library

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use Narwhals as internal engine, not user-facing API | Narwhals unifies implementation, but users pass native frames | — Pending |
| Register per data type, not globally | Follows existing BACKEND_REGISTRY pattern; enables incremental rollout | — Pending |
| `narwhals` as optional extra | Avoids forcing dependency on all users | — Pending |
| Polars/Ibis first, pandas/PySpark after | Higher confidence in non-pandas paths through Narwhals | — Pending |
| Whether a `narwhals_engine.py` is needed | The existing Engine/DataType pattern maps native dtypes to pandera DataTypes per library. It's unclear if Narwhals exposes a unified dtype system sufficient to replace per-library engines, or if we should delegate dtype handling to the existing engines within the Narwhals backend. Needs research. | — Pending |

---
*Last updated: 2026-03-08 after initialization*
