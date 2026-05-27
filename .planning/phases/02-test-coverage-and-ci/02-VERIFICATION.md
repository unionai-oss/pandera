---
phase: 02-test-coverage-and-ci
verified: 2026-05-11T20:10:54Z
status: gaps_found
score: 5/8 must-haves verified
overrides_applied: 0
gaps:
  - truth: "Running `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/` produces no unexpected failures — every failure is either a passing test or an `xfail` with a justifying comment"
    status: failed
    reason: "300 tests fail under PANDERA_USE_NARWHALS_BACKEND=True; all 300 are Category C backend bugs (not xfail-marked) deferred to follow-on Plan 02-04. Root causes: (1) validate() raises SchemaErrors instead of setting df.pandera.errors (~290 tests), (2) _concat_failure_cases() crashes with PySparkAttributeError, (3) dtype string format mismatch (Int64 vs IntegerType()), (4) unnecessary PySpark materialization causing STRUCT_ARRAY_LENGTH_MISMATCH."
    artifacts:
      - path: "tests/pyspark/test_pyspark_check.py"
        issue: "~250 parametrized tests fail due to SchemaErrors being raised instead of set on df.pandera.errors"
      - path: "tests/pyspark/test_pyspark_dtypes.py"
        issue: "All 58 tests fail due to narwhals materializing PySpark DataFrames + dtype string mismatch"
      - path: "tests/pyspark/test_pyspark_error.py"
        issue: "4/5 tests fail (SchemaErrors raised + _concat_failure_cases crash)"
      - path: "tests/pyspark/test_pyspark_container.py"
        issue: "5/11 tests fail due to SchemaErrors + dtype mismatch"
      - path: "tests/pyspark/test_pyspark_model.py"
        issue: "9/17 tests fail due to SchemaErrors + dtype mismatch"
    missing:
      - "Plan 02-04 must be created to fix 4 Category C root-cause bugs in pandera/backends/narwhals/ before SC1 is met"
      - "After fixes: re-run suite to confirm 0 FAILED outcomes (only PASSED + XFAIL)"

  - truth: "Any test failure that is not an expected SQL-lazy limitation (i.e., a true narwhals backend bug) is diagnosed and fixed before this phase closes"
    status: failed
    reason: "4 distinct root-cause backend bugs were identified (Category C) but ALL were deferred under the per-run cap with a PHASE SPLIT RECOMMENDED signal. Zero Category C entries have been fixed. Plan 02-04 has not yet been created."
    artifacts:
      - path: "pandera/backends/narwhals/container.py"
        issue: "validate() raises SchemaErrors instead of setting df.pandera.errors for PySpark frames (affects ~290 tests)"
      - path: "pandera/backends/narwhals/base.py"
        issue: "_concat_failure_cases() uses pl.concat() which crashes on PySpark DataFrames (AttributeError on ._df)"
      - path: "pandera/backends/narwhals/column.py or components.py"
        issue: "dtype string reports narwhals format (Int64) instead of PySpark format (IntegerType())"
      - path: "pandera/backends/narwhals/base.py"
        issue: "_materialize() called on PySpark DataFrames during validation, triggering STRUCT_ARRAY_LENGTH_MISMATCH"
    missing:
      - "Create Plan 02-04 to rearchitect narwhals DataFrameSchemaBackend.validate() for PySpark"
      - "Fix _concat_failure_cases to use PySpark .union() instead of pl.concat() for PySpark frames"
      - "Fix dtype string to use PySpark-native format in ColumnBackend.check_dtype()"
      - "Fix _materialize() to avoid collecting PySpark DataFrames during validation"

  - truth: "D-05: element-wise checks, sample=/tail= params, and row-index failures are each covered by at least one xfail-marked test"
    status: failed
    reason: "row-index in failure_cases is listed in ROADMAP SC2 and PLAN 02-01 D-05 as a required xfail coverage area, but no xfail-marked test covering row-index in failure_cases exists in tests/pyspark/. The native pyspark test suite does not have a row-index-in-failure_cases test because PySpark lacks pandas-style row indices. This means either: (a) the requirement is inapplicable to pyspark (needs human clarification), or (b) it is a gap."
    artifacts: []
    missing:
      - "Human decision: is 'row-index in failure_cases' applicable to pyspark tests? If yes, create a test that validates the absence of row-index in narwhals failure_cases for pyspark. If no, update ROADMAP SC2 and PLAN must-have to exclude this clause for pyspark."
---

# Phase 2: Test Coverage and CI — Verification Report

**Phase Goal:** The existing PySpark test suite runs cleanly under the Narwhals backend, with expected SQL-lazy limitations marked `xfail`, unexpected bugs fixed, and a CI nox session added

**Verified:** 2026-05-11T20:10:54Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC1 | Running `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/` produces no unexpected failures — every failure is either a passing test or an `xfail` with a justifying comment | FAILED | 300 FAILED outcomes observed; all Category C backend bugs, all deferred. TRIAGE.md line 5: "300 failed (Category C — PHASE SPLIT RECOMMENDED)" |
| SC2a | Element-wise checks are covered by at least one `xfail`-marked test | VERIFIED | TestCustomCheck.test_extension and test_extension_dataframe_model xfail-marked (API incompatibility = element-wise limitation). test_pyspark_check.py lines 1845–1848, 1872–1875. |
| SC2b | `sample=`/`tail=` params are covered by at least one `xfail`-marked test | VERIFIED | test_pyspark_sample xfail-marked in test_pyspark_container.py line 139: "sample= is not supported in the Narwhals backend" |
| SC2c | row-index in `failure_cases` is covered by at least one `xfail`-marked test | FAILED | No xfail-marked test covering row-index in failure_cases exists in any pyspark test file. Native PySpark tests don't include row-index assertions. The clause may be inapplicable — needs human decision. |
| SC3 | Any test failure that is not an expected SQL-lazy limitation (true backend bug) is diagnosed and fixed before this phase closes | FAILED | 4 root-cause Category C bugs diagnosed (TRIAGE.md lines 143–227) but NONE fixed. All deferred to follow-on Plan 02-04 (not yet created). |
| SC4 | A nox session (or parametrized entry) runs `tests/pyspark/` under `PANDERA_USE_NARWHALS_BACKEND=True` with pyspark and narwhals dependencies installed, and that session is listed in the CI matrix | VERIFIED | noxfile.py: `@nox.parametrize("extra", ["polars", "ibis", "pyspark"])` + pyspark[connect]>=3.2.0 guard + numpy<2 on py3.10 guard. ci-tests.yml: matrix.extra=[polars, ibis, pyspark], conditional Java 17, pyspark excludes on 3.12+3.13. All files parse cleanly. |

**Score:** 3/6 roadmap success criteria verified (SC1 and SC3 failed outright; SC2c uncertain)

### Plan-level Must-Have Truths

| Plan | # | Truth | Status | Evidence |
|------|---|-------|--------|----------|
| 02-01 | 1 | All 5 TestPanderaConfig methods in test_pyspark_config.py are xfail-marked when narwhals backend is active | VERIFIED | `grep -c "condition=CONFIG.use_narwhals_backend" test_pyspark_config.py` = 5 |
| 02-01 | 2 | test_pyspark_sample in test_pyspark_container.py is xfail-marked for the sample= SQL-lazy limitation | VERIFIED | Line 139: "sample= is not supported in the Narwhals backend; use head= instead", strict=True |
| 02-01 | 3 | TestCustomCheck.test_extension and test_extension_dataframe_model in test_pyspark_check.py are xfail-marked | VERIFIED | Lines 1845–1849 and 1872–1876; strict=True confirmed by direct file read |
| 02-01 | 4 | TestUniqueValuesEqCheck.test_unique_values_eq_check and test_failed_unaccepted_datatypes are xfail-marked | VERIFIED | test_unique_values_eq_check: class-level xfail (line 2034); test_failed_unaccepted_datatypes: per-parametrization xfail for ArrayType+MapType via _xfail_unique_values_eq (line 1978), BooleanType correctly excluded after XPASS fix |
| 02-01 | 5 | TestPanderaDecorators.test_cache_dataframe_settings in test_pyspark_decorators.py is xfail-marked | VERIFIED | Lines 1845-1848 (confirmed by grep: condition=CONFIG.use_narwhals_backend, strict=True) |
| 02-01 | 6 | Every xfail marker uses condition=CONFIG.use_narwhals_backend, strict=True, and a specific reason string | VERIFIED | All 4 files: 11 `condition=CONFIG.use_narwhals_backend` occurrences total, all have `strict=True` (verified by direct file reads at exact lines — regex false positive due to `)` in reason string was investigated and ruled out) |
| 02-01 | 7 | Each modified test file imports CONFIG from pandera.config | VERIFIED | All 4 files pass `grep "from pandera.config import.*CONFIG"` |
| 02-01 | 8 | D-05: element-wise checks, sample=/tail= params, and row-index failures each covered by at least one xfail-marked test | FAILED | element-wise and sample= covered; row-index not covered (see SC2c above) |
| 02-02 | 1 | The tests_narwhals_backend nox session parametrizes over polars, ibis, AND pyspark | VERIFIED | noxfile.py: `@nox.parametrize("extra", ["polars", "ibis", "pyspark"])` confirmed |
| 02-02 | 2 | When extra='pyspark', the session installs pyspark[connect] >= 3.2.0 and (on Python 3.10) constrains numpy < 2 | VERIFIED | noxfile.py: pyspark guard and numpy<2 on `session.python in ("3.10",)` both present |
| 02-02 | 3 | When extra='pyspark', the tests/common/ pytest run is skipped | VERIFIED | noxfile.py: `if extra in ("polars", "ibis"):` guard on tests/common/ run |
| 02-02 | 4 | The CI unit-tests-narwhals-backend job matrix includes pyspark | VERIFIED | ci-tests.yml: `extra: [polars, ibis, pyspark]` |
| 02-02 | 5 | The CI job installs Java 17 only when matrix.extra == 'pyspark' | VERIFIED | ci-tests.yml: `if: matrix.extra == 'pyspark'` on actions/setup-java@v4 step |
| 02-02 | 6 | The CI matrix excludes pyspark on Python 3.12 and 3.13 | VERIFIED | ci-tests.yml: exclude entries for pyspark+3.12 and pyspark+3.13 confirmed via YAML parse |
| 02-03 | 1 | TRIAGE.md documents every test in tests/pyspark/ | VERIFIED | 109 table rows; covers all filtered tests (minus ignored pyspark-pandas file and spark_connect); all 7 RESEARCH.md assumptions resolved |
| 02-03 | 2 | Running tests under narwhals produces no unexpected failures (after triage) | FAILED | 300 FAILED, all Category C backend bugs, all [DEFERRED — exceeded per-run cap] |
| 02-03 | 3 | No xfail uses os.getenv() | VERIFIED | `grep -rE "condition=os\.getenv" tests/pyspark/` produces no output |
| 02-03 | 4 | Every newly-added xfail uses strict=True | VERIFIED | All 11 condition=CONFIG.use_narwhals_backend occurrences confirmed to have strict=True by direct file reads |

**Overall Plan Score:** 5/8 key must-have areas verified; SC1 (clean run) and SC3 (bugs fixed) are the primary blockers.

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/pyspark/test_pyspark_config.py` | 5 xfail markers + CONFIG import | VERIFIED | 5 occurrences of `condition=CONFIG.use_narwhals_backend`; CONFIG imported; file parses OK |
| `tests/pyspark/test_pyspark_check.py` | 4 xfail sites (markers + per-param) + CONFIG import | VERIFIED | 4 `condition=CONFIG.use_narwhals_backend` occurrences; XPASS fix applied (per-param marks for ArrayType/MapType); file parses OK |
| `tests/pyspark/test_pyspark_container.py` | 1 xfail marker + CONFIG import | VERIFIED | 1 occurrence; CONFIG imported; file parses OK |
| `tests/pyspark/test_pyspark_decorators.py` | 1 xfail marker + CONFIG import | VERIFIED | 1 occurrence; CONFIG imported; file parses OK |
| `noxfile.py` | pyspark in parametrize + dep guard + common/ guard | VERIFIED | All 3 changes present; noxfile parses OK |
| `.github/workflows/ci-tests.yml` | pyspark matrix + conditional Java + excludes | VERIFIED | All 3 changes present; YAML parses OK |
| `.planning/phases/02-test-coverage-and-ci/02-03-TRIAGE.md` | Per-test triage record ≥30 rows; all 7 assumptions resolved | VERIFIED | 109 table rows; 7 assumption lines (`^- A[1-7] (`); Category B/C/D sections; 4 verbatim Failure output blocks; PHASE SPLIT RECOMMENDED signal |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| test_pyspark_config.py xfail markers | pandera.config.CONFIG | `condition=CONFIG.use_narwhals_backend` | WIRED | 5 occurrences confirmed |
| test_pyspark_check.py xfail markers | pandera.config.CONFIG | `condition=CONFIG.use_narwhals_backend` | WIRED | 4 occurrences (regex false positive on reason-string `)` investigated; actual file has strict=True on all 4) |
| test_pyspark_container.py xfail marker | pandera.config.CONFIG | `condition=CONFIG.use_narwhals_backend` | WIRED | 1 occurrence confirmed |
| test_pyspark_decorators.py xfail marker | pandera.config.CONFIG | `condition=CONFIG.use_narwhals_backend` | WIRED | 1 occurrence confirmed |
| noxfile.py @nox.parametrize extra list | tests/pyspark/ path | `f"tests/{extra}/"` path resolution | WIRED | `["polars", "ibis", "pyspark"]` confirmed; f-string path unchanged |
| ci-tests.yml unit-tests-narwhals-backend matrix | nox session tests_narwhals_backend pyspark | matrix.extra interpolation | WIRED | `extra: [polars, ibis, pyspark]` confirmed |
| ci-tests.yml unit-tests-narwhals-backend steps | actions/setup-java@v4 | `if: matrix.extra == 'pyspark'` | WIRED | Conditional Java step present; verified via YAML parse |

---

### Data-Flow Trace (Level 4)

Not applicable — this phase produces test markers, configuration, and documentation artifacts only. No dynamic-data rendering components introduced.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| noxfile.py parses as valid Python | `python3 -c "import ast; ast.parse(open('noxfile.py').read())"` | exit 0 | PASS |
| ci-tests.yml parses as valid YAML | `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci-tests.yml'))"` | exit 0 | PASS |
| pyspark in nox parametrize | `grep "@nox.parametrize.*pyspark" noxfile.py` | found | PASS |
| pyspark install guard present | `grep 'if extra == "pyspark":' noxfile.py` | found | PASS |
| CI matrix includes pyspark | YAML parse confirms `extra: [polars, ibis, pyspark]` | found | PASS |
| CI excludes pyspark on 3.12+3.13 | YAML parse confirms both exclude entries | found | PASS |
| Conditional Java 17 setup | YAML parse: Java step `if` = `matrix.extra == 'pyspark'` | found | PASS |
| TRIAGE.md exists with ≥30 rows | `grep -c "^| " 02-03-TRIAGE.md` | 109 | PASS |
| All 7 RESEARCH.md assumptions resolved | `grep -cE "^- A[1-7] \(" 02-03-TRIAGE.md` | 7 | PASS |
| No os.getenv in xfail conditions | `grep -rE "condition=os\.getenv" tests/pyspark/` | no output | PASS |
| XPASS fix applied | test_failed_unaccepted_datatypes restructured to per-parametrization marks (BooleanType excluded) | confirmed | PASS |

---

### Probe Execution

No probes declared in plan frontmatter. Step 7c: SKIPPED (no probe scripts declared; running the nox pyspark session requires Java + PySpark which is not available in this verification environment).

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| TEST-02 | 02-01 | Expected PySpark+Narwhals limitations are xfail-marked: element-wise checks, sample=/tail= params, row-index in failure_cases | PARTIAL | element-wise (TestCustomCheck xfails) and sample= (test_pyspark_sample xfail) covered; row-index in failure_cases NOT covered by any xfail-marked test |
| CI-01 | 02-02 | A nox session runs PySpark test suite under PANDERA_USE_NARWHALS_BACKEND=True with pyspark + narwhals deps | SATISFIED | tests_narwhals_backend parametrized with pyspark; pyspark[connect]>=3.2.0 + numpy<2 guards; CI matrix entry with conditional Java 17 |
| TEST-01 | 02-03 | The existing PySpark test suite runs under PANDERA_USE_NARWHALS_BACKEND=True with all failures either passing or xfail-marked | BLOCKED | 300 FAILED tests remain (all Category C backend bugs); TRIAGE.md documents all; Plan 02-04 required |
| TEST-03 | 02-03 | Unexpected failures (true bugs in narwhals backend) are investigated and fixed | BLOCKED | 4 root-cause bugs diagnosed in TRIAGE.md but 0 fixed; all deferred with [DEFERRED — exceeded per-run cap]; Plan 02-04 required |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `.planning/phases/02-test-coverage-and-ci/02-03-TRIAGE.md` (in memory, not code) | N/A | PHASE SPLIT RECOMMENDED signal present but Plan 02-04 not yet created | Warning | Plan 02-03 required the follow-on plan to exist before phase 02 closes; the SUMMARY documents this but the plan file is absent |

No `TBD`, `FIXME`, `XXX`, or `PLACEHOLDER` markers found in any file modified by this phase. No empty implementations. The regex-based strict=True check produced 2 false positives (reason strings containing `)` truncate the `[^)]*` match group) — manually verified at lines 1845–1849 and 1872–1876 of test_pyspark_check.py; both have `strict=True`.

---

### Human Verification Required

#### 1. Row-Index in failure_cases Applicability

**Test:** Determine whether the ROADMAP SC2 clause "row-index in `failure_cases`" applies to PySpark tests.

**Expected:** Either (a) confirm that native PySpark failure_cases intentionally have no row-index (distributed system — no integer row index exists), and update ROADMAP SC2 and PLAN 02-01 must-have D-05 to drop this clause for pyspark; or (b) identify a specific test that should be xfail-marked for this limitation and create it.

**Why human:** The PySpark test suite has no test that validates row-index in failure_cases (pyspark failure_cases use column-level not row-level error reporting). The requirement may be inherited from ibis/polars limitations that don't apply to pyspark. Resolving this requires a product decision, not a code check.

#### 2. Plan 02-04 Creation Gate

**Test:** Confirm Plan 02-04 has been created (or is being created) to address the 4 Category C backend bugs before closing Phase 02.

**Expected:** `.planning/phases/02-test-coverage-and-ci/02-04-PLAN.md` exists and addresses the 4 root-cause bugs in `pandera/backends/narwhals/`.

**Why human:** Plan 02-03 emitted a PHASE SPLIT RECOMMENDED signal, which means Phase 02 should not be marked complete until Plan 02-04 ships. This verifier confirms the plan file does not yet exist (`ls .planning/phases/02-test-coverage-and-ci/02-04*` returns no matches).

---

### Gaps Summary

Phase 02 is **not yet complete**. Two requirements are unambiguously blocked:

**BLOCKER 1 — TEST-01 (SC1):** 300 unexpected failures remain under `PANDERA_USE_NARWHALS_BACKEND=True`. All are Category C backend bugs. The triage run in Plan 02-03 correctly identified them but all were deferred under the per-run cap (exceeded because root cause 1 alone requires 30-50 lines of core backend changes). Plan 02-04 must be created to fix these before SC1 can be satisfied.

**BLOCKER 2 — TEST-03 (SC3):** Zero backend bugs fixed. Four root causes documented in TRIAGE.md: (1) validate() raises SchemaErrors instead of setting df.pandera.errors (~290 tests), (2) _concat_failure_cases() crashes with _df AttributeError on PySpark DataFrames, (3) dtype string format mismatch (narwhals reports Int64 vs PySpark's IntegerType()), (4) unnecessary materialization triggering STRUCT_ARRAY_LENGTH_MISMATCH in test_pyspark_dtypes.py. These require targeted changes to `pandera/backends/narwhals/container.py`, `base.py`, and `column.py`.

**WARNING — TEST-02 (SC2c):** The "row-index in failure_cases" xfail coverage requirement is not met. Whether this is a gap or an inapplicable requirement for PySpark needs human decision.

**What IS complete:**
- Plan 02-01 (TEST-02): 11 xfail markers applied correctly across 4 test files; XPASS regression fixed
- Plan 02-02 (CI-01): nox session extended with pyspark extra; CI matrix updated with conditional Java + excludes
- Plan 02-03 Task 1 (TRIAGE.md): Complete, self-contained, 109-row triage report; all 7 assumptions resolved; Category B XPASS correction applied

**Recommended next action:** Run `/gsd-plan-phase --gaps` with the gaps structured above to generate Plan 02-04 targeting the 4 Category C bugs. After Plan 02-04 ships, re-verify SC1 and SC3.

---

_Verified: 2026-05-11T20:10:54Z_
_Verifier: Claude (gsd-verifier)_
