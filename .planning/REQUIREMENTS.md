# Requirements: Pandera Narwhals Backend

**Defined:** 2026-05-10
**Core Value:** Users can validate any Narwhals-supported dataframe library through a single, consistent backend — reducing maintenance burden and unlocking lazy validation and future library support for free.

## v1.3 Requirements

Requirements for the Narwhals Backend for PySpark milestone. Each maps to roadmap phases.

### Registration

- [ ] **REG-01**: `register_pyspark_backends()` conditionally registers `NarwhalsCheckBackend`, `ColumnBackend`, and `DataFrameSchemaBackend` for `pyspark_sql.DataFrame` (and `pyspark_connect.DataFrame` if available) when `PANDERA_USE_NARWHALS_BACKEND=True`; the `else` branch keeps existing native registrations untouched

### Testing

- [ ] **TEST-01**: The existing PySpark test suite runs under `PANDERA_USE_NARWHALS_BACKEND=True` with all failures either passing or `xfail`-marked with a justifying comment
- [ ] **TEST-02**: Expected PySpark+Narwhals limitations are `xfail`-marked: element-wise checks (no `map_batches` on SQL-lazy), `sample=`/`tail=` params, row-index in `failure_cases`
- [ ] **TEST-03**: Unexpected failures (true bugs in the narwhals backend or error-reporting layer) are investigated and fixed

### CI

- [ ] **CI-01**: A nox session (or parametrized entry) runs the PySpark test suite under `PANDERA_USE_NARWHALS_BACKEND=True` with pyspark + narwhals deps installed

### Documentation

- [ ] **DOCS-01**: Narwhals backend documentation lists PySpark as a supported SQL-lazy backend alongside Ibis/DuckDB, with a note on SQL-lazy limitations (no element-wise checks, no row sampling)

### Architecture (Phase 4 — Pre-Merge Review Fixes)

- [x] **ARCH-01**: `run_check` in `pandera/backends/narwhals/base.py` has no PySpark-specific implementation branch (`Implementation in (PYSPARK, PYSPARK_CONNECT)` check removed or eliminated via `_materialize()` fix)
- [x] **ARCH-02**: `_concat_failure_cases` in `pandera/backends/narwhals/base.py` uses narwhals-native dispatch (`nw.Implementation`) instead of module-string sniffing; scalar polars frames are not silently dropped when PySpark frames are present
- [x] **ARCH-03**: `check_dtype` in `pandera/backends/narwhals/components.py` uses schema-driven detection (`isinstance(schema.dtype, pyspark_engine.DataType)`) instead of frame-implementation probe
- [x] **ARCH-04**: PySpark error-setting logic in `pandera/backends/narwhals/container.py` is extracted to a `_handle_pyspark_validation_result()` method rather than an inline `is_pyspark` block

### Correctness (Phase 5 — Pre-Merge Review Fixes)

- [x] **CORR-01**: `strict='filter'` returns filtered columns for PySpark narwhals in the success path
- [x] **CORR-02**: `df.pandera.schema` is set after narwhals PySpark validation (behavioral parity with native backend)
- [x] **TEST-FIX-01**: `test_pyspark_config.py` band-aid xfails removed (hardcoded `use_narwhals_backend: False` in expected dicts replaced with dynamic or key-removed assertions)

### Test Coverage (Phase 6 — Pre-Merge Review Fixes)

- [x] **TEST-E2E-01**: `tests/narwhals/test_e2e.py` includes a PySpark section with backend registration, return-type preservation, passing/failing check with failure cases, and nullable/unique behavior
- [x] **NITS-01**: Minor pre-merge nits resolved: CI Python version exclusion comment, "not in dataframe" message, registration test completeness, stacked xfail marks, `supported_types()` double-append

## Future Requirements

Deferred to future milestones and not included in the current roadmap.

### Additional Backend Support

- pandas validation working via Narwhals backend (including lazy mode)

### Features

- `add_missing_columns` parser (FEAT-01)
- `set_default` for Column fields (FEAT-02)
- Groupby-based checks via `group_by(...).agg()` pattern (FEAT-04)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Narwhals-specific user-facing API | Narwhals is internal plumbing; users pass native frames |
| Removal of native PySpark backend | Coexist until Narwhals backend is proven stable |
| Strategies (Hypothesis) for PySpark via Narwhals | Defer to future milestone |
| Schema IO (YAML/JSON) for PySpark via Narwhals | Defer to future milestone |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| REG-01 | Phase 1 | Pending |
| TEST-01 | Phase 2 | Pending |
| TEST-02 | Phase 2 | Pending |
| TEST-03 | Phase 2 | Pending |
| CI-01 | Phase 2 | Pending |
| DOCS-01 | Phase 3 | Pending |
| ARCH-01 | Phase 4 | Complete |
| ARCH-02 | Phase 4 | Complete |
| ARCH-03 | Phase 4 | Complete |
| ARCH-04 | Phase 4 | Complete |
| CORR-01 | Phase 5 | Complete |
| CORR-02 | Phase 5 | Complete |
| TEST-FIX-01 | Phase 5 | Complete |
| TEST-E2E-01 | Phase 6 | Complete |
| NITS-01 | Phase 6 | Complete |

**Coverage:**
- v1.3 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0 ✓

---
*Requirements defined: 2026-05-10*
*Last updated: 2026-05-25 after Phase 4 completion*
