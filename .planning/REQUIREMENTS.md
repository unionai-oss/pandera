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

**Coverage:**
- v1.3 requirements: 6 total
- Mapped to phases: 6
- Unmapped: 0 ✓

---
*Requirements defined: 2026-05-10*
*Last updated: 2026-05-10 after roadmap creation*
