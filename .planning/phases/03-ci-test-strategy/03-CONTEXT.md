# Phase 3: CI Test Strategy - Context

**Gathered:** 2026-04-11
**Status:** Ready for planning
**Source:** Requirements-driven (no discuss-phase session)

<domain>
## Phase Boundary

Ensure existing Polars and Ibis backend tests run cleanly without Narwhals installed. Parametrize Narwhals backend tests across all supported frame types (pl.DataFrame, pl.LazyFrame, ibis.Table). Document a CI matrix covering both isolation scenarios.

</domain>

<decisions>
## Implementation Decisions

### TEST-01: Existing Backend Isolation
- Existing tests in `tests/polars/` and `tests/ibis/` must pass when narwhals is installed in the same environment
- Add session-scoped autouse fixtures to re-register native backends, defending against narwhals backend shadowing
- Add architecture regression test to `tests/backends/narwhals/test_phase01_arch.py` enforcing conftest isolation

### TEST-02: Narwhals Test Parametrization
- `make_narwhals_frame` fixture must parametrize across all three native types: `pl.DataFrame`, `pl.LazyFrame`, `ibis.Table`
- Each parametrization ID must be descriptive: `polars_eager`, `polars_lazy`, `ibis_table`
- No test may silently skip any frame type — all three must execute or be explicitly xfail/skip

### TEST-03: CI Matrix Documentation
- Document CI matrix in conftest docstring at top of `tests/backends/narwhals/conftest.py`
- Add narrative comment block in `.github/workflows/ci-tests.yml` explaining the isolation split
- Add narwhals CI job or matrix entry that installs narwhals + polars + ibis and runs `tests/backends/narwhals/`
- Wire narwhals nox session in `noxfile.py`

### Claude's Discretion
- Exact CI job name and structure (new job vs matrix entry)
- Nox session parametrization approach (new mapping vs special-case in `test_dir` assignment)
- Comment placement and wording in CI matrix documentation

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` — TEST-01, TEST-02, TEST-03 requirement definitions
- `.planning/ROADMAP.md` — Phase 3 success criteria and goal

### Test Infrastructure
- `tests/backends/narwhals/conftest.py` — existing narwhals test fixtures and registration
- `tests/polars/conftest.py` — existing polars test configuration
- `tests/ibis/conftest.py` — existing ibis test configuration (may not exist)

### CI and Build
- `.github/workflows/ci-tests.yml` — current CI matrix structure
- `noxfile.py` — nox session definitions and `EXTRA_PYTHON_PYDANTIC` parametrization
- `pyproject.toml` — extras declarations

</canonical_refs>

<deferred>
## Deferred Ideas

None — requirements fully captured in decisions above.

</deferred>

---

*Phase: 03-ci-test-strategy*
*Context gathered: 2026-04-11 via requirements-driven (no discuss-phase)*
