# Phase 2: Test Coverage and CI - Context

**Gathered:** 2026-05-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Run the existing `tests/pyspark/` suite under `PANDERA_USE_NARWHALS_BACKEND=True`, triage every failure as either an expected SQL-lazy limitation (`xfail`) or a true narwhals backend bug (investigate and fix), then wire a nox session into the CI matrix. Scope is the vanilla `pyspark_sql.DataFrame` path only; Spark Connect coverage is excluded.

</domain>

<decisions>
## Implementation Decisions

### Nox Session Shape

- **D-01:** Extend the existing `tests_narwhals_backend` session by adding `"pyspark"` to its `@nox.parametrize("extra", [...])` list. Each parametrize value is an independent nox session with its own virtualenv, so PySpark deps (Java, pyspark, numpy<2 on Python 3.10) are fully isolated. No separate session needed.
- **D-02:** The test path resolves automatically from `f"tests/{extra}/"` → `tests/pyspark/`. Any pyspark-specific dep constraints (numpy<2 on Python 3.10) must be handled inside the function body with an `if extra == "pyspark":` guard, mirroring how the base `_testing_requirements` handles it.

### xfail Marker Strategy

- **D-03:** SQL-lazy limitation xfails use `condition=CONFIG.use_narwhals_backend` — the same convention as `tests/ibis/test_ibis_check.py` and `tests/polars/test_polars_config.py`. Import: `from pandera.config import CONFIG`. Do NOT use `os.getenv(...)` directly.
- **D-04:** All SQL-lazy limitation xfails use `strict=True` — if narwhals unexpectedly fixes a limitation, CI must catch it and the marker must be removed.
- **D-05:** Known SQL-lazy limitations to xfail (from REQUIREMENTS.md TEST-02): element-wise checks, `sample=`/`tail=` params, row-index in `failure_cases`.

### Config Test Handling

- **D-06:** `test_pyspark_config.py` is the PySpark analog of `test_polars_config.py` (no `test_ibis_config.py` exists). Its tests hardcode `"use_narwhals_backend": False` in expected config dicts, which will fail under narwhals mode. No pre-treatment — the triage run surfaces these failures, and the executor applies `condition=CONFIG.use_narwhals_backend, strict=True` xfail markers exactly as was done in `test_polars_config.py`.

### PySpark Connect Scope

- **D-07:** Vanilla `pyspark_sql.DataFrame` only. Spark Connect (`pyspark_connect.DataFrame`) requires a live remote server (`sc://localhost`) — not practical in standard CI. Registration of `pyspark_connect` was already tested in Phase 1. The existing `spark_connect` fixture in `tests/pyspark/conftest.py` is left as-is.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` — TEST-01, TEST-02, TEST-03, CI-01 define what "done" means for this phase
- `.planning/ROADMAP.md` §Phase 2 — success criteria and phase goal

### Existing Narwhals Backend Session (direct model)
- `noxfile.py` lines 328–376 — `tests_narwhals_backend` session: extend its `@nox.parametrize` list and follow its install/env pattern
- `noxfile.py` lines 166–170 — numpy<2 constraint for pyspark+Python 3.10 (must replicate inside `tests_narwhals_backend` for `extra="pyspark"`)

### xfail Convention (must match exactly)
- `tests/ibis/test_ibis_check.py` lines 43–47 — canonical `condition=CONFIG.use_narwhals_backend, strict=True` pattern
- `tests/polars/test_polars_config.py` lines 82–86 — same pattern in a config test file (direct analog for `test_pyspark_config.py`)

### PySpark Test Suite
- `tests/pyspark/` — all 13 test files are in scope for triage
- `tests/pyspark/conftest.py` — Spark session setup; `spark` fixture is session-scoped; `spark_connect` fixture excluded from narwhals coverage

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tests_narwhals_backend` nox session (noxfile:328–376): extend parametrize list, add `if extra == "pyspark":` guards for numpy constraint and any pyspark-specific install logic
- `from pandera.config import CONFIG` import: already present in `test_ibis_check.py` and `test_polars_config.py` — add to any pyspark test file that gains xfail markers

### Established Patterns
- `@pytest.mark.xfail(condition=CONFIG.use_narwhals_backend, reason="...", strict=True)` — established in polars and ibis test files; must be followed uniformly for pyspark
- `f"tests/{extra}/"` path resolution in `tests_narwhals_backend` — pyspark falls out naturally
- `env = {"PANDERA_USE_NARWHALS_BACKEND": "True"}` passed to `session.run("pytest", ...)` — already in the session, no change needed

### Integration Points
- `noxfile.py` `@nox.parametrize("extra", ["polars", "ibis"])` → extend to `["polars", "ibis", "pyspark"]`
- CI matrix (`.github/workflows/`) — the `tests_narwhals_backend` session is already in the matrix; adding "pyspark" to the parametrize list makes CI pick it up automatically

</code_context>

<specifics>
## Specific Ideas

- The numpy<2 constraint for pyspark on Python 3.10 is currently handled in `_testing_requirements` (noxfile:167–170) but not in `tests_narwhals_backend`. The planner should add an `if extra == "pyspark" and session.python in ("3.10",):` guard inside `tests_narwhals_backend`.
- `test_pyspark_config.py` failure pattern is predictable: any test that asserts `asdict(get_config_context()) == expected` where `expected["use_narwhals_backend"] == False` will fail under narwhals mode.

</specifics>

<deferred>
## Deferred Ideas

- **Synthetic column construction refactor** (`pandera/backends/narwhals/container.py:318-323`): abstraction leak where narwhals backend imports framework-specific Column classes. Reviewed as a potential triage item but left as a standalone todo — out of scope for Phase 2 unless it surfaces as a blocking failure during TEST-03 triage.
- **Spark Connect test coverage**: excluded from this phase; can be revisited in a future milestone once vanilla PySpark narwhals coverage is stable.
- **`test_ibis_config.py` and cross-backend config test parity**: Ibis currently has no config test file equivalent to `test_polars_config.py` and `test_pyspark_config.py`. A future cleanup could add `test_ibis_config.py` and potentially parametrize config tests across backends to avoid per-backend divergence.

</deferred>

---

*Phase: 2-test-coverage-and-ci*
*Context gathered: 2026-05-10*
