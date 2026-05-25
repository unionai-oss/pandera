---
phase: 05-correctness-and-behavioral-parity
verified: 2026-05-25T18:00:00Z
status: human_needed
score: 3/3 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Run test_pyspark_model.py::test_dataframe_schema_strict under PANDERA_USE_NARWHALS_BACKEND=True"
    expected: "Test passes (not xfail/xpass); schema.validate(df) with strict='filter' returns a DataFrame with only columns ['a', 'b']"
    why_human: "Requires Java and a running Spark session; cannot be verified with grep or Python import checks alone"
  - test: "Run test_pyspark_accessor.py::test_dataframe_add_schema under PANDERA_USE_NARWHALS_BACKEND=True"
    expected: "Test passes (not xfail/xpass); data.pandera.schema == schema1 holds after validation"
    why_human: "Requires Java and a running Spark session; cannot be verified with grep or Python import checks alone"
  - test: "Run tests/pyspark/test_pyspark_config.py::TestPanderaConfig under PANDERA_USE_NARWHALS_BACKEND=True"
    expected: "All five TestPanderaConfig tests pass without xfail; asdict(get_config_context()) assertion passes with use_narwhals_backend=True"
    why_human: "Requires Java and a running Spark session; executor confirmed Java unavailable in execution environment"
  - test: "Run PANDERA_USE_NARWHALS_BACKEND=False tests/pyspark/test_pyspark_config.py::TestPanderaConfig"
    expected: "All five tests still pass; no regression to native PySpark path"
    why_human: "Requires Java and a running Spark session"
  - test: "Run full tests/pyspark/ suite under PANDERA_USE_NARWHALS_BACKEND=True"
    expected: "No unexpected new failures; only the known pre-existing xfails remain (group_by limitation, coerce_dtype limitation, stacked xfail at line 551)"
    why_human: "Requires Java and a running Spark session for full integration validation"
---

# Phase 5: Correctness and Behavioral Parity Verification Report

**Phase Goal:** Achieve behavioral parity between the narwhals PySpark validation path and the native PySpark backend for two pre-merge defects (CORR-01, CORR-02) and restore meaningful config test coverage (TEST-FIX-01).
**Verified:** 2026-05-25T18:00:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `strict='filter'` returns filtered columns for PySpark narwhals in the success path — `_to_frame_kind_nw(check_lf, return_type)` is returned with `errors = {}` attached; corresponding xfail removed | VERIFIED | `container.py` lines 242-245 pass `_to_frame_kind_nw(check_lf, return_type)` as first arg; lines 231-234 do same for error path. No occurrence of `"narwhals PySpark backend always returns original frame"` in `test_pyspark_model.py`. Method signature unchanged. |
| 2 | `check_obj.pandera.add_schema(schema)` is called before returning from narwhals PySpark validation; xfail in `test_pyspark_accessor.py` removed | VERIFIED | `container.py` line 280 has `check_obj.pandera.add_schema(schema)` as first statement of `_handle_pyspark_validation_result`, before the `if has_errors:` branch. No occurrence of `"narwhals backend does not call add_schema"` in `test_pyspark_accessor.py`. Behavioral spot-check ran and passed. |
| 3 | The five `test_pyspark_config.py` tests that xfailed due to hardcoded `"use_narwhals_backend": False` are fixed — updated to use `CONFIG.use_narwhals_backend` dynamically | VERIFIED | `test_pyspark_config.py` contains exactly 5 occurrences of `"use_narwhals_backend": CONFIG.use_narwhals_backend`, 0 occurrences of `"use_narwhals_backend": False`, 0 xfail reason strings, 0 `@pytest.mark.xfail` decorators. `pytestmark` and `CONFIG` import unchanged. |

**Score:** 3/3 truths verified (code evidence; integration test execution requires human with Java/Spark)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pandera/backends/narwhals/container.py` | Updated validate() PySpark return sites pass `_to_frame_kind_nw(check_lf, return_type)`; `_handle_pyspark_validation_result` calls `check_obj.pandera.add_schema(schema)` | VERIFIED | Commit 9b01cd16. Lines 232-234 (error path) and 243-245 (success path) confirmed. Line 280 has the `add_schema` call. Method signature unchanged: `def _handle_pyspark_validation_result(self, check_obj, error_handler, schema, has_errors: bool)`. Docstring updated (line 278: "pandera.schema and pandera.errors set"). |
| `tests/pyspark/test_pyspark_model.py` | `test_dataframe_schema_strict` with CORR-01 xfail decorator removed | VERIFIED | Commit 1f2a4f9b. `grep -c "narwhals PySpark backend always returns original frame" → 0`. Remaining 3 xfails are unrelated (group_by limitation, ValueError, coerce_dtype). `from pandera.config import CONFIG` import preserved. |
| `tests/pyspark/test_pyspark_accessor.py` | `test_dataframe_add_schema` with CORR-02 xfail decorator removed | VERIFIED | Commit 1f2a4f9b. `grep -c "narwhals backend does not call add_schema" → 0`. 0 xfail decorators remain. `from pandera.config import CONFIG` import preserved. |
| `tests/narwhals/test_phase01_arch.py` | ARCH-04 success/error path tests extended with `add_schema` assertions | VERIFIED | Commit ee31ea60. Exactly 2 occurrences of `check_obj.pandera.add_schema.assert_called_once_with(schema)`. Placement verified: after `summarize.assert_*` and before `assert result is check_obj` in both tests. Still exactly 4 `test_handle_pyspark_validation_result` functions. |
| `tests/pyspark/test_pyspark_config.py` | Five TestPanderaConfig tests with xfail decorators removed and dynamic `CONFIG.use_narwhals_backend` assertions | VERIFIED | Commit 3180a5fb. `dynamic_count=5`, `hardcoded_false_count=0`, `xfail_reason_count=0`, `xfail_count=0`. `CONFIG` import and `pytestmark` unchanged. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `validate()` error path (line 232) | `_handle_pyspark_validation_result` | `_to_frame_kind_nw(check_lf, return_type)` as first arg | WIRED | Line 233: `_to_frame_kind_nw(check_lf, return_type), error_handler, schema, has_errors=True` confirmed |
| `validate()` success path (line 243) | `_handle_pyspark_validation_result` | `_to_frame_kind_nw(check_lf, return_type)` as first arg | WIRED | Line 244: `_to_frame_kind_nw(check_lf, return_type), error_handler, schema, has_errors=False` confirmed |
| `_handle_pyspark_validation_result` | `PanderaPySparkAccessor.add_schema` | `check_obj.pandera.add_schema(schema)` called before `if has_errors:` | WIRED | Line 280 confirmed; behavioral spot-check ran the method via MagicMock and asserted `add_schema.assert_called_once_with(schema)` on both paths |
| `test_pyspark_config.py` expected dicts | `CONFIG.use_narwhals_backend` | Dynamic reference in 5 expected dicts | WIRED | 5 occurrences of `"use_narwhals_backend": CONFIG.use_narwhals_backend` confirmed |

### Data-Flow Trace (Level 4)

Not applicable. This phase modifies validation logic (return value and accessor mutation) and test files — no new dynamic data rendering components.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `_handle_pyspark_validation_result` sets `add_schema` on error path | `python3 -c "...MagicMock error path..."` | `check_obj.pandera.add_schema.assert_called_once_with(schema)` passed | PASS |
| `_handle_pyspark_validation_result` sets `add_schema` on success path | `python3 -c "...MagicMock success path..."` | `check_obj.pandera.add_schema.assert_called_once_with(schema)` passed | PASS |
| `add_schema` placed before `if has_errors:` branch | `python3 -c "...index comparison..."` | `add_schema_pos (pos 280) < has_errors_pos` | PASS |
| Both call sites pass `_to_frame_kind_nw(check_lf, return_type)` | `python3 -c "...validate_src.count(...)..."` | count = 2 | PASS |
| No old `check_obj` call sites remain in `validate()` | `python3 -c "...check_obj call pattern..."` | False (no old sites) | PASS |
| `test_pyspark_config.py` dynamic count = 5 | `python3 -c "...content.count(...)..."` | 5 | PASS |
| `test_pyspark_config.py` hardcoded False count = 0 | `python3 -c "...content.count(...)..."` | 0 | PASS |
| Narwhals module imports clean | `python3 -c "from pandera.backends.narwhals.container import DataFrameSchemaBackend; print('OK')"` | OK | PASS |

### Probe Execution

Step 7c: SKIPPED — No `probe-*.sh` scripts declared or found in `scripts/` for this phase.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| CORR-01 | 05-01-PLAN.md | `strict='filter'` returns filtered columns for PySpark narwhals in the success path | SATISFIED | Both PySpark return sites in `validate()` pass `_to_frame_kind_nw(check_lf, return_type)`; xfail removed from `test_pyspark_model.py` |
| CORR-02 | 05-01-PLAN.md | `df.pandera.schema` is set after narwhals PySpark validation (behavioral parity with native backend) | SATISFIED | `check_obj.pandera.add_schema(schema)` added as first statement in `_handle_pyspark_validation_result`; xfail removed from `test_pyspark_accessor.py` |
| TEST-FIX-01 | 05-02-PLAN.md | `test_pyspark_config.py` band-aid xfails removed; hardcoded `use_narwhals_backend: False` replaced with dynamic or key-removed assertions | SATISFIED | 5 dynamic `CONFIG.use_narwhals_backend` references, 0 hardcoded `False`, 0 xfail decorators remain in `TestPanderaConfig` |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | None found | — | — |

No TBD/FIXME/XXX markers, no stub implementations, no empty return values, no placeholder patterns detected in the four modified files.

### Human Verification Required

### 1. Integration Test: CORR-01 strict='filter' (PySpark)

**Test:** Run `PANDERA_USE_NARWHALS_BACKEND=True pixi run pyspark-test tests/pyspark/test_pyspark_model.py::test_dataframe_schema_strict -x` in an environment with Java installed
**Expected:** Test passes without xfail/xpass; `schema.validate(df).columns == ['a', 'b']` when `strict='filter'`
**Why human:** Requires Java Runtime and a running Spark session. The executor environment has no Java, so the test cannot run. The code fix is structurally correct (verified via source inspection and MagicMock behavioral checks), but only an integration run can confirm end-to-end behavior.

### 2. Integration Test: CORR-02 pandera.schema accessor (PySpark)

**Test:** Run `PANDERA_USE_NARWHALS_BACKEND=True pixi run pyspark-test tests/pyspark/test_pyspark_accessor.py::test_dataframe_add_schema -x` in an environment with Java installed
**Expected:** Test passes without xfail/xpass; `data.pandera.schema == schema1` after `schema1(data)` call
**Why human:** Same as above — Java runtime required.

### 3. Integration Test: TEST-FIX-01 config dict assertions (narwhals mode)

**Test:** Run `PANDERA_USE_NARWHALS_BACKEND=True pixi run pyspark-test tests/pyspark/test_pyspark_config.py::TestPanderaConfig -x` in an environment with Java installed
**Expected:** All 5 TestPanderaConfig tests pass without xfail; `asdict(get_config_context())` equals expected dict with `use_narwhals_backend=True`
**Why human:** Java runtime required.

### 4. Integration Test: TEST-FIX-01 config dict assertions (native mode)

**Test:** Run `PANDERA_USE_NARWHALS_BACKEND=False pixi run pyspark-test tests/pyspark/test_pyspark_config.py::TestPanderaConfig -x`
**Expected:** All 5 tests still pass; `use_narwhals_backend=False` in both expected and actual config dicts
**Why human:** Java runtime required; validates no regression to native PySpark path.

### 5. Full PySpark Suite Regression Check

**Test:** Run `PANDERA_USE_NARWHALS_BACKEND=True pixi run pyspark-test tests/pyspark/` in an environment with Java installed
**Expected:** No unexpected new failures; only the 3 known pre-existing xfails in `test_pyspark_model.py` remain (group_by limitation at line 363, ValueError at line 550, coerce_dtype at line 551)
**Why human:** Java runtime required for full suite.

### Gaps Summary

No code gaps found. All three requirements (CORR-01, CORR-02, TEST-FIX-01) have complete, substantive, and wired implementations. The `human_needed` status reflects the pre-existing infrastructure constraint: PySpark integration tests require Java, which is not available in the verification environment. This is not a new limitation introduced by this phase — it was already noted in the 05-02-SUMMARY.md. The code logic is sound and was verified via:

- Source inspection (correct call sites, correct method body, correct test assertions)
- MagicMock behavioral spot-checks exercising the actual method contract
- Pattern-match verification of all acceptance criteria from both plans

---

_Verified: 2026-05-25T18:00:00Z_
_Verifier: Claude (gsd-verifier)_
