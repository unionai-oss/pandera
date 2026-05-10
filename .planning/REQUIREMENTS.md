# Requirements: Pandera Narwhals Backend

**Defined:** 2026-03-29
**Core Value:** Users can validate any Narwhals-supported dataframe library through a single, consistent backend — reducing maintenance burden and unlocking lazy validation and future library support for free.

## v1.2 Requirements

Requirements for the PR Review Cleanup & Test Strategy milestone. Each maps to roadmap phases.

### Native Type Detection

- [ ] **TYPES-01**: Unified constants (`EAGER_NATIVE_TYPES`, `LAZY_NATIVE_TYPES`, or equivalent) define which native frame types are eager vs lazy — used everywhere in place of ad-hoc `isinstance`/`hasattr` checks
- [ ] **TYPES-02**: `_is_lazy(frame)` (or equivalent) utility replaces all scattered `hasattr(native, "execute")` and `isinstance(fc, nw.LazyFrame)` checks
- [x] **TYPES-03**: Failure case result handling in `base.py`, `container.py`, and `components.py` uses a clean dispatch pattern backed by TYPES-01/02 rather than complex if/elif/else blocks

### Backend Isolation

- [ ] **CLEAN-01**: `narwhals/checks.py` contains no Polars-specific imports or code paths
- [ ] **CLEAN-02**: `narwhals/container.py` does not import from `pandera.api.polars.components`
- [x] **CLEAN-03**: `narwhals/base.py` does not produce code paths that require Polars installed when validating Ibis frames (base.py:294 branch)
- [x] **CLEAN-04**: All inner imports moved to top-level (stdlib in `container.py:449`; narwhals engine imports in `narwhals_engine.py:34,56`; any others)

### Eager Execution

- [ ] **EAGER-01**: `narwhals_engine.py` does not call `.collect()` on entire frames; coerce/try_coerce do not materialize 100B-row datasets
- [ ] **EAGER-02**: `container.py` does not call `.collect()` on `pl.DataFrame` inputs unnecessarily; `components.py` does not use `_materialize` solely to perform a lazy concat

### Custom Checks

- [ ] **CHECKS-01**: User-defined (custom) checks work through the Narwhals backend; root cause of current failure identified, fixed, and covered by a test

### Documentation

- [x] **DOCS-01**: `pandera/api/checks.py` `native` param docstring clarifies it only applies when using the Narwhals backend (not all backends)
- [x] **DOCS-02**: "Narwhals" is consistently capitalized in all comments, docstrings, and `register.py` files

### Testing Strategy

- [ ] **TEST-01**: Existing Polars and Ibis backend tests pass (or are `xfail`-marked with justification) when Narwhals is installed in the same environment — CI infrastructure in place; xfails TBD in follow-up
- [x] **TEST-02**: Narwhals backend tests parametrize across `pl.DataFrame`, `pl.LazyFrame`, and `ibis.Table`; all parametrized cases pass
- [x] **TEST-03**: CI matrix covers: (a) existing backends without Narwhals installed (`unit-tests-dataframe-extras`), (b) existing backends with Narwhals installed (`unit-tests-narwhals-backend`), (c) narwhals-specific tests with all supported frame types (`unit-tests-narwhals`)

## Future Requirements

These are deferred to future milestones and not included in the current roadmap.

### Additional Backend Support

- pandas validation working via Narwhals backend (including lazy mode)
- PySpark validation working via Narwhals backend (if feasible)

### Features

- `add_missing_columns` parser
- `set_default` for Column fields
- Groupby-based checks via `group_by(...).agg()` pattern

## Out of Scope

| Feature | Reason |
|---------|--------|
| Narwhals-specific user-facing API | Narwhals is internal plumbing, not a user-facing target |
| Immediate removal of library-native backends | Coexist until Narwhals backend is proven |
| Strategies (Hypothesis) for Narwhals backend | Deferred to future milestone |
| Schema IO (YAML/JSON) for Narwhals backend | Deferred to future milestone |
| `narwhals stable.v2` migration | Monitor releases; migrate only when officially stabilized |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| TYPES-01 | Phase 1 | Pending |
| TYPES-02 | Phase 1 | Pending |
| TYPES-03 | Phase 1 | Complete |
| CLEAN-01 | Phase 1 | Pending |
| CLEAN-02 | Phase 1 | Pending |
| CLEAN-03 | Phase 1 | Complete |
| CLEAN-04 | Phase 1 | Complete |
| EAGER-01 | Phase 1 | Pending |
| EAGER-02 | Phase 1 | Pending |
| CHECKS-01 | Phase 1 | Pending |
| DOCS-01 | Phase 2 | Complete |
| DOCS-02 | Phase 2 | Complete |
| TEST-01 | Phase 3 | Complete |
| TEST-02 | Phase 3 | Complete |
| TEST-03 | Phase 3 | Complete |

**Coverage:**
- v1.2 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-29*
*Last updated: 2026-03-29 after roadmap creation*
