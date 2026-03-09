# Requirements: Pandera Narwhals Backend

**Defined:** 2026-03-09
**Core Value:** Users can validate any Narwhals-supported dataframe library through a single, consistent backend — reducing maintenance burden and unlocking lazy validation and future library support for free.

## v1 Requirements

### Infrastructure

- [x] **INFRA-01**: `narwhals>=2.15.0` added as optional extra in `pyproject.toml` (`pandera[narwhals]`); all imports use `narwhals.stable.v1`
- [x] **INFRA-02**: `pandera/api/narwhals/types.py` exists with `NarwhalsData` named tuple (analogous to `PolarsData`) enabling `Dispatcher` routing to narwhals builtin checks
- [x] **INFRA-03**: `pandera/api/narwhals/utils.py` exists with `_to_native()` helper used at every `SchemaError` construction site to prevent narwhals wrappers leaking into error messages

### Dtype Engine

- [x] **ENGINE-01**: `pandera/engines/narwhals_engine.py` exists with `Engine` metaclass following the existing `engine.Engine` pattern from `polars_engine.py`
- [x] **ENGINE-02**: Narwhals dtype objects (`nw.Int8/16/32/64`, `nw.UInt*`, `nw.Float32/64`, `nw.String`, `nw.Boolean`, `nw.Date`, `nw.Datetime`, `nw.Duration`, `nw.Categorical`, `nw.List`, `nw.Struct`) are registered via `@Engine.register_dtype` and map to pandera `DataType` subclasses
- [x] **ENGINE-03**: `coerce()` and `try_coerce()` are implemented via `nw.col(name).cast(nw_dtype)` and return native frames

### Check Backend

- [ ] **CHECKS-01**: `pandera/backends/narwhals/checks.py` exists with `NarwhalsCheckBackend` that routes builtin checks to `NarwhalsData` containers and user-defined checks to native containers
- [ ] **CHECKS-02**: `pandera/backends/narwhals/builtin_checks.py` exists with all 14 builtin checks implemented via narwhals Expr API: `equal_to`, `not_equal_to`, `greater_than`, `greater_than_or_equal_to`, `less_than`, `less_than_or_equal_to`, `in_range`, `isin`, `notin`, `str_matches`, `str_contains`, `str_startswith`, `str_endswith`, `str_length`
- [ ] **CHECKS-03**: `element_wise=True` checks on SQL-lazy backends (Ibis, PySpark, DuckDB) raise `NotImplementedError` with a clear explanation rather than silently failing

### Column Backend

- [ ] **COLUMN-01**: `pandera/backends/narwhals/components.py` exists with `ColumnBackend` implementing `check_nullable` (handling float NaN via `is_nan()` in addition to `is_null()`), `check_unique`, `check_dtype` (via narwhals engine), and `run_checks`
- [ ] **COLUMN-02**: `check_unique` forces collection via `.collect()` before calling `is_duplicated()`; the collect-first pattern is documented to prevent lazy evaluation bugs

### Container Backend

- [ ] **CONTAINER-01**: `pandera/backends/narwhals/base.py` exists with `NarwhalsSchemaBackend` providing shared helpers: `subsample()`, `run_check()`, `failure_cases_metadata()`, `drop_invalid_rows()`
- [ ] **CONTAINER-02**: `pandera/backends/narwhals/container.py` exists with `DataFrameSchemaBackend` implementing the full validation pipeline: wraps native frame with `nw.from_native()` as first step, runs column presence, dtype, unique, and check validation, unwraps with `nw.to_native()` as last step
- [ ] **CONTAINER-03**: `DataFrameSchemaBackend` supports `strict` and `filter` column modes via `collect_schema().names()` and `frame.drop()`
- [ ] **CONTAINER-04**: Lazy validation mode (`lazy=True`) collects all errors via `ErrorHandler` before raising `SchemaErrors`

### Registration

- [ ] **REGISTER-01**: `pandera/backends/narwhals/register.py` exists with `register_narwhals_backends()` decorated with `lru_cache`, guarded by per-library `try/except ImportError`, writing directly into `BACKEND_REGISTRY` (not via `register_backend()`) to override existing entries when opt-in is active
- [ ] **REGISTER-02**: Narwhals backend registers for `pl.DataFrame` and `pl.LazyFrame` (Polars) — end-to-end `schema.validate(df)` works for Polars frames
- [ ] **REGISTER-03**: Narwhals backend registers for `ibis.Table` — end-to-end `schema.validate(table)` works for Ibis frames, closing known xfail gaps (`coerce_dtype`, column `unique`)
- [ ] **REGISTER-04**: Opt-in activation mechanism exists — narwhals backend is never registered by default; requires explicit `pandera.use_backend("narwhals")` or `import pandera.narwhals`

### Testing

- [ ] **TEST-01**: `tests/backends/narwhals/` directory exists with backend-agnostic test suite parameterized via pytest markers to run against each registered backend (Polars, Ibis at minimum)
- [ ] **TEST-02**: Tests cover schema validation (column presence, dtype check, nullable, unique), all 14 builtin checks, lazy validation mode, dtype coercion, and error message correctness (native frame types in `failure_cases`)
- [ ] **TEST-03**: Tests assert that `SchemaError.failure_cases` is always a native frame type (not a narwhals wrapper)

## v2 Requirements

### Extended Library Support

- **PANDAS-01**: pandas `pd.DataFrame` registered as a narwhals backend target — enables lazy-mode pandas validation
- **PYSPARK-01**: PySpark `DataFrame` registered (PySpark >=3.5.0 required by narwhals)
- **DUCKDB-01**: DuckDB `PyRelation` registered (zero additional work once core is stable)
- **PYARROW-01**: PyArrow `Table` registered

### Extended Features

- **FEAT-01**: `add_missing_columns` parser implemented (depends on narwhals_engine.py coerce path)
- **FEAT-02**: `set_default` for Column fields implemented
- **FEAT-03**: `drop_invalid_rows` with library-specific positional alignment strategy
- **FEAT-04**: Groupby-based checks via `group_by(...).agg()` pattern

### Phase-Out

- **PHASEOUT-01**: Polars native backend deprecated with migration notice pointing to narwhals backend
- **PHASEOUT-02**: Ibis native backend deprecated with migration notice

## Out of Scope

| Feature | Reason |
|---------|--------|
| `pandera.narwhals` user-facing API | Narwhals is internal plumbing only; users keep existing `pandera.polars`, `pandera.pandas`, etc. APIs |
| Series / Index / MultiIndex validation | Narwhals has no index concept; narwhals backend is DataFrame + Column only |
| Hypothesis strategies | Explicitly deferred in PROJECT.md |
| Schema IO (YAML/JSON) for narwhals | Explicitly deferred in PROJECT.md |
| `map_batches` / element-wise checks on Ibis/PySpark | Not implemented in narwhals for SQL-lazy backends; raise `NotImplementedError` |
| pyspark.pandas registration | Route to native pandas backend; do not mix |
| GeoDataFrame / modin / dask via narwhals | Defer based on narwhals coverage and demand |
| Narwhals stable.v2 migration | Monitor narwhals releases; migrate only when stable.v2 is officially stabilized |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | Phase 1 | Complete |
| INFRA-02 | Phase 1 | Complete |
| INFRA-03 | Phase 1 | Complete |
| ENGINE-01 | Phase 1 | Complete |
| ENGINE-02 | Phase 1 | Complete |
| ENGINE-03 | Phase 1 | Complete |
| CHECKS-01 | Phase 2 | Pending |
| CHECKS-02 | Phase 2 | Pending |
| CHECKS-03 | Phase 2 | Pending |
| TEST-01 | Phase 2 | Pending |
| COLUMN-01 | Phase 3 | Pending |
| COLUMN-02 | Phase 3 | Pending |
| CONTAINER-01 | Phase 4 | Pending |
| CONTAINER-02 | Phase 4 | Pending |
| CONTAINER-03 | Phase 4 | Pending |
| CONTAINER-04 | Phase 4 | Pending |
| REGISTER-01 | Phase 4 | Pending |
| REGISTER-02 | Phase 4 | Pending |
| REGISTER-04 | Phase 4 | Pending |
| TEST-03 | Phase 4 | Pending |
| REGISTER-03 | Phase 5 | Pending |
| TEST-02 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 21 total
- Mapped to phases: 21
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-09*
*Last updated: 2026-03-09 after roadmap creation*
