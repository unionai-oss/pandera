---
phase: 2
slug: test-coverage-and-ci
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-10
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (pyproject.toml `[tool.pytest.ini_options]`) |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/ -k "spark and not spark_connect" -x --no-header -q` |
| **Full suite command** | `nox -s "tests_narwhals_backend-3.11(extra='pyspark')"` |
| **Estimated runtime** | ~3–5 minutes (PySpark JVM startup + test suite) |

---

## Sampling Rate

- **After every task commit:** Run `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/test_pyspark_config.py -x -q`
- **After every plan wave:** Run `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/ -k "spark and not spark_connect" -q`
- **Before `/gsd-verify-work`:** Full nox session must be green — `nox -s "tests_narwhals_backend-3.11(extra='pyspark')"`
- **Max feedback latency:** ~60 seconds (per-task quick run)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Automated Command | File Exists | Status |
|---------|------|------|-------------|-------------------|-------------|--------|
| 02-01-01 | 02-01 | 1 | TEST-02 | `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/test_pyspark_config.py -x -q` | ✅ | ⬜ pending |
| 02-01-02 | 02-01 | 1 | TEST-02 | `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/test_pyspark_check.py tests/pyspark/test_pyspark_container.py tests/pyspark/test_pyspark_decorators.py -x -q` | ✅ | ⬜ pending |
| 02-02-01 | 02-02 | 1 | CI-01 | `python -c "import ast, sys; ast.parse(open('noxfile.py').read()); print('parse OK')"` | ✅ | ⬜ pending |
| 02-02-02 | 02-02 | 1 | CI-01 | `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci-tests.yml')); print('parse OK')"` | ✅ | ⬜ pending |
| 02-03-01 | 02-03 | 2 | TEST-01, TEST-03 | `test -f .planning/phases/02-test-coverage-and-ci/02-03-TRIAGE.md && grep -c "Category" .planning/phases/02-test-coverage-and-ci/02-03-TRIAGE.md` | ⬜ W2 | ⬜ pending |
| 02-03-02 | 02-03 | 2 | TEST-01 | `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/ -k "spark and not spark_connect" -q` | ✅ | ⬜ pending |
| 02-03-03 | 02-03 | 2 | TEST-03 | `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/ -k "spark and not spark_connect" -q` | ✅ | ⬜ pending |
| 02-03-04 | 02-03 | 2 | TEST-01 | Human verify: run full suite, confirm no unexpected failures | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing test infrastructure covers all phase requirements. No new test files needed; changes are xfail markers on existing tests plus noxfile/CI wiring.

*Wave 0 complete — no stubs required.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| CI matrix runs pyspark nox session | CI-01 | Requires GitHub Actions environment | Push branch; confirm `unit-tests-narwhals-backend / tests_narwhals_backend-3.11(extra='pyspark')` job appears in CI run |
| No unexpected failures after triage | TEST-01 | Requires full PySpark JVM + narwhals runtime | Run `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/ -k "spark and not spark_connect"` locally; confirm all failures are xfail |
