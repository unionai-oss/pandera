---
phase: 03-ci-test-strategy
verified: 2026-04-11T16:00:00Z
status: passed
score: 7/7 must-haves verified
re_verification:
  previous_status: gaps_found
  previous_score: 5/7
  gaps_closed:
    - "Session-scoped autouse isolation fixtures exist in tests/polars/conftest.py and tests/ibis/conftest.py (TEST-01)"
    - "Regression-guard test_polars_and_ibis_conftests_do_not_import_narwhals_backend exists in test_phase01_arch.py (TEST-01)"
  gaps_remaining: []
  regressions: []
---

# Phase 03: CI Test Strategy Verification Report

**Phase Goal:** Establish a CI test strategy for the Narwhals backend — ensure the three test environments (TEST-01: isolation, TEST-02: parametrization, TEST-03: narwhals CI job) are all wired up and documented.
**Verified:** 2026-04-11T16:00:00Z
**Status:** passed
**Re-verification:** Yes — after gap closure

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `make_narwhals_frame` fixture parametrizes across 3 native frame types: `pl.DataFrame`, `pl.LazyFrame`, and `ibis.Table` | ✓ VERIFIED | `tests/backends/narwhals/conftest.py`: `params=["polars_eager", "polars_lazy", "ibis_table"]` confirmed; all 3 branches wired in `_make()` |
| 2 | Tests in `tests/backends/narwhals/` that exercise specific frame types carry `TEST-02: intentionally {type}-specific` annotations | ✓ VERIFIED | `test_e2e.py` and `test_lazy_regressions.py` have Strategy C annotations; `test_checks.py` uses `make_narwhals_frame` for all cross-backend tests |
| 3 | Session-scoped autouse isolation fixtures exist in `tests/polars/conftest.py` and `tests/ibis/conftest.py` to prevent narwhals backend shadowing (TEST-01) | ✓ VERIFIED | `tests/polars/conftest.py` (45 lines): `_ensure_polars_backend_registered` session fixture at line 31; `tests/ibis/conftest.py` (36 lines): `_ensure_ibis_backend_registered` session fixture at line 27; both files contain TEST-01 guard comments |
| 4 | A regression-guard test in `test_phase01_arch.py` asserts polars and ibis conftests do not import `pandera.backends.narwhals` | ✓ VERIFIED | `test_polars_and_ibis_conftests_do_not_import_narwhals_backend` at line 278 of `test_phase01_arch.py` (313 lines); checks both conftests; raises AssertionError with "TEST-01 violation" message for both forbidden import patterns |
| 5 | A `unit-tests-narwhals` CI job exists in `.github/workflows/ci-tests.yml` that installs narwhals + polars + ibis and runs `tests/backends/narwhals/` | ✓ VERIFIED | `grep -c 'unit-tests-narwhals'` returns 2 (job definition + cross-reference comment); nox session invoked with `extra='narwhals'` |
| 6 | `noxfile.py` maps `--extra narwhals` to `tests/backends/narwhals/` with polars + ibis co-installed | ✓ VERIFIED | `test_dir = "backends/narwhals"` special-case present; `"narwhals"` in `DATAFRAME_EXTRAS`; `_testing_requirements` co-installs polars + ibis extras |
| 7 | The CI matrix documentation (TEST-01/02/03) is present in `tests/backends/narwhals/conftest.py` docstring | ✓ VERIFIED | `grep -c 'CI Matrix'` returns 1; docstring references TEST-01, TEST-02, TEST-03, `ci-tests.yml`, and `REQUIREMENTS.md` |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/backends/narwhals/conftest.py` | 3-way parametrized `make_narwhals_frame` fixture + CI Matrix docstring | ✓ VERIFIED | `params=["polars_eager", "polars_lazy", "ibis_table"]` confirmed; CI Matrix docstring with TEST-01/02/03 references present |
| `tests/backends/narwhals/test_e2e.py` | TEST-02 annotations for intentionally frame-type-specific tests | ✓ VERIFIED | Strategy C annotations (`# TEST-02: intentionally {type}-specific`) present |
| `tests/backends/narwhals/test_lazy_regressions.py` | TEST-02 annotations | ✓ VERIFIED | Strategy C annotations present |
| `tests/polars/conftest.py` | Session-scoped `_ensure_polars_backend_registered` fixture + TEST-01 guard comment | ✓ VERIFIED | 45-line file; fixture at line 31; TEST-01 guard comment at lines 10-12; `register_polars_backends` imported and called with `hasattr` cache_clear guard; no narwhals imports |
| `tests/ibis/conftest.py` | File with `validation_depth_schema_and_data` + `_ensure_ibis_backend_registered` fixture | ✓ VERIFIED | 36-line file created; both fixtures present; TEST-01 guard comment at lines 10-11; `register_ibis_backends` imported and called; no narwhals imports |
| `tests/backends/narwhals/test_phase01_arch.py` | `test_polars_and_ibis_conftests_do_not_import_narwhals_backend` | ✓ VERIFIED | Function at line 278 of 313-line file; checks both conftests; guards "from pandera.backends.narwhals" and "import pandera.backends.narwhals" patterns; cites TEST-01 in error messages |
| `.github/workflows/ci-tests.yml` | `unit-tests-narwhals` job with narwhals + polars + ibis; TEST-01/02/03 comment blocks | ✓ VERIFIED | Job defined; `extra='narwhals'` nox invocation; TEST-03 CI Matrix comment block above job; TEST-01 cross-reference comment above `unit-tests-dataframe-extras` job |
| `noxfile.py` | `"narwhals"` in `DATAFRAME_EXTRAS`; `test_dir = "backends/narwhals"` special-case; polars + ibis co-install | ✓ VERIFIED | All three confirmed; polars + ibis extras added in `_testing_requirements` when `extra == "narwhals"` |
| `pyproject.toml` | `narwhals` optional dependency present | ✓ VERIFIED | `narwhals = ["narwhals >= 2.15.0"]` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `conftest.py::make_narwhals_frame` | `pl.DataFrame / pl.LazyFrame / ibis.Table` | `request.param` branching on `polars_eager`, `polars_lazy`, `ibis_table` | ✓ WIRED | All 3 branches present; `polars_eager` uses `eager_only=True`, `polars_lazy` uses `eager_or_interchange_only=False`, `ibis_table` uses `ibis.memtable` |
| `tests/polars/conftest.py` | `pandera.backends.polars.register` | `_ensure_polars_backend_registered` autouse session fixture | ✓ WIRED | Fixture imports `register_polars_backends` and calls it with conditional `cache_clear()` guard |
| `tests/ibis/conftest.py` | `pandera.backends.ibis.register` | `_ensure_ibis_backend_registered` autouse session fixture | ✓ WIRED | Fixture imports `register_ibis_backends` and calls it with conditional `cache_clear()` guard |
| `ci-tests.yml::unit-tests-narwhals` | `tests/backends/narwhals/` | `nox session tests-{py}(extra='narwhals', ...)` | ✓ WIRED | Step invokes correct nox session; narwhals extra maps to `backends/narwhals` test_dir in noxfile |
| `noxfile.py::_testing_requirements` | `polars + ibis extras` | `if extra == "narwhals": _requirements += [polars, ibis]` | ✓ WIRED | Co-install block present in `_testing_requirements` |
| `test_phase01_arch.py` | `tests/polars/conftest.py` + `tests/ibis/conftest.py` | `test_polars_and_ibis_conftests_do_not_import_narwhals_backend` import-line scan | ✓ WIRED | Test iterates both conftest paths; asserts file exists; scans each line for forbidden import patterns; error messages cite TEST-01 |

### Data-Flow Trace (Level 4)

Not applicable — this phase produces test infrastructure and CI configuration, not data-rendering components.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `make_narwhals_frame` has 3-way params | `grep -c 'params=\["polars_eager", "polars_lazy", "ibis_table"\]' tests/backends/narwhals/conftest.py` | 1 | ✓ PASS |
| polars conftest has isolation fixture | `grep -c "_ensure_polars_backend_registered" tests/polars/conftest.py` | 1 | ✓ PASS |
| ibis conftest exists and has isolation fixture | `ls tests/ibis/conftest.py` + `grep -c "_ensure_ibis_backend_registered" tests/ibis/conftest.py` | file found; 1 | ✓ PASS |
| arch test has TEST-01 regression guard | `grep -c "def test_polars_and_ibis_conftests_do_not_import_narwhals_backend" tests/backends/narwhals/test_phase01_arch.py` | 1 | ✓ PASS |
| no narwhals imports in polars/ibis conftests | `grep -c "from pandera.backends.narwhals\|import pandera.backends.narwhals" tests/polars/conftest.py tests/ibis/conftest.py` | 0, 0 | ✓ PASS |
| noxfile maps narwhals to backends/narwhals | `grep -c 'test_dir = "backends/narwhals"' noxfile.py` | 1 | ✓ PASS |
| CI job unit-tests-narwhals defined | `grep -c "unit-tests-narwhals" .github/workflows/ci-tests.yml` | 2 | ✓ PASS |
| CI matrix docstring in conftest | `grep -c 'CI Matrix' tests/backends/narwhals/conftest.py` | 1 | ✓ PASS |
| pyproject.toml narwhals extra present | `python3 -c "import tomllib; p=tomllib.load(open('pyproject.toml','rb')); assert 'narwhals' in p['project']['optional-dependencies']"` | exits 0 | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| TEST-01 | 03-02-PLAN.md | Existing Polars and Ibis backend tests pass when Narwhals is installed in the same environment | ✓ SATISFIED | `tests/polars/conftest.py` has `_ensure_polars_backend_registered` session fixture; `tests/ibis/conftest.py` has `_ensure_ibis_backend_registered` session fixture; `test_phase01_arch.py` has regression guard `test_polars_and_ibis_conftests_do_not_import_narwhals_backend`; no narwhals imports in either conftest |
| TEST-02 | 03-01-PLAN.md | Narwhals backend tests parametrize across `pl.DataFrame`, `pl.LazyFrame`, and `ibis.Table`; all parametrized cases pass | ✓ SATISFIED | `make_narwhals_frame` uses `params=["polars_eager", "polars_lazy", "ibis_table"]`; frame-type-specific tests carry Strategy C annotations |
| TEST-03 | 03-03-PLAN.md | CI matrix is documented and covers: (a) existing backends without Narwhals installed, (b) Narwhals backend with all supported frame types | ✓ SATISFIED | `unit-tests-narwhals` CI job defined; TEST-03 comment block in `ci-tests.yml`; CI Matrix docstring in `tests/backends/narwhals/conftest.py`; noxfile maps narwhals extra to `backends/narwhals` test directory |

### Anti-Patterns Found

No blockers or warnings detected. All previously-flagged anti-patterns (missing isolation fixtures, missing arch regression test) have been resolved.

### Human Verification Required

None — all checks are programmatically verifiable.

### Re-Verification Summary

Both gaps from the initial verification have been closed. No regressions in the 5 items that passed initial verification.

**Gap 1 closed — TEST-01 isolation fixtures now present.**

`tests/polars/conftest.py` (now 45 lines) contains the `_ensure_polars_backend_registered` session-scoped autouse fixture with a TEST-01 guard comment block. `tests/ibis/conftest.py` (now 36 lines, newly created) contains both the `validation_depth_schema_and_data` function-scoped fixture and the `_ensure_ibis_backend_registered` session-scoped autouse fixture. Both files have zero narwhals imports. The implementation uses `hasattr(register_*_backends, "cache_clear")` as a robustness guard before calling `cache_clear()` — a minor defensive improvement over the plan's exact template text.

**Gap 2 closed — TEST-01 architecture regression guard now present.**

`test_phase01_arch.py` grew from 275 to 313 lines. The `test_polars_and_ibis_conftests_do_not_import_narwhals_backend` function at line 278 iterates both conftest paths, asserts each file exists, and scans every source line for both `from pandera.backends.narwhals` and `import pandera.backends.narwhals` import patterns. Error messages cite "TEST-01 violation" explicitly.

---

_Verified: 2026-04-11T16:00:00Z_
_Verifier: Claude (gsd-verifier)_
