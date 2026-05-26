---
phase: 08-test-quality-improvements
verified: 2026-05-26T15:00:00Z
status: passed
score: 4/4 must-haves verified
overrides_applied: 0
---

# Phase 8: Test Quality Improvements Verification Report

**Phase Goal:** Replace test anti-patterns identified in the updated PR review with idiomatic, maintainable alternatives — no production code changes except the `_concat_failure_cases` polars-branch fix, only test improvements
**Verified:** 2026-05-26T15:00:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| #  | Truth | Status | Evidence |
|----|-------|--------|---------|
| 1  | `test_pyspark_error.py` contains no inline `if CONFIG.use_narwhals_backend else` ternaries | VERIFIED | `grep -c 'if CONFIG.use_narwhals_backend' tests/pyspark/test_pyspark_error.py` returns 0; `from pandera.config import CONFIG` import is absent; 3 DATA assertions use `_cmp_errors(...)` at lines 68, 164, 237 |
| 2  | `_concat_failure_cases` polars branch merges `pl_items` (not silently drops them) | VERIFIED | `pandera/backends/narwhals/base.py` lines 119-122: `lazy_result = nw.to_native(nw.concat(nw_items))` + `if pl_items: return pl.concat([lazy_result.collect()] + pl_items)` + `return lazy_result`; regression test file exists and passes |
| 3  | `test_arch03_schema_driven_dispatch.py` source-inspection tests replaced with behavioral equivalents | VERIFIED | `grep -c 'inspect.getsource'` returns 0; `import inspect` removed; 4 old source-inspection tests are gone; 2 new behavioral PySpark-gated tests exist; existing narwhals-engine test preserved |
| 4  | PySpark registration either adds `nw.DataFrame` or documents why it is omitted | VERIFIED | Comment block at lines 62-65 of `pandera/backends/pyspark/register.py` explains intentional omission citing ibis precedent; no `Check.register_backend(nw.DataFrame, ...)` call added |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/pyspark/conftest.py` | Module-level `_cmp_errors(actual, expected)` helper | VERIFIED | Function exists at line 194; inner helper named `drop_error` (no underscore per naming convention); correct docstring and body |
| `tests/pyspark/test_pyspark_config.py` | `TestPanderaConfig._cmp_errors` delegates to conftest | VERIFIED | Lines 33-36: `@staticmethod`, one-line delegation body `_cmp_errors(actual, expected)`; explicit import via `from tests.pyspark.conftest import _cmp_errors, spark_df` |
| `tests/pyspark/test_pyspark_error.py` | Zero CONFIG ternaries; 3 DATA assertions use `_cmp_errors` | VERIFIED | `grep -c 'if CONFIG.use_narwhals_backend'` returns 0; `grep -n '_cmp_errors'` shows exactly 3 call sites (lines 68, 164, 237); CONFIG import absent |
| `pandera/backends/narwhals/base.py` | Polars branch merges `pl_items` via `pl.concat([lazy_result.collect()] + pl_items)` | VERIFIED | Lines 119-122 contain exact fix; no `SchemaWarning` in polars branch; PySpark branch warning unchanged |
| `tests/narwhals/test_concat_failure_cases.py` | Regression test suite for `_concat_failure_cases` polars branch | VERIFIED | File exists; 3 tests: `test_concat_failure_cases_polars_merges_pl_items_and_nw_items`, `test_concat_failure_cases_polars_only_nw_items_returns_lazy_native`, `test_concat_failure_cases_polars_emits_no_warning`; correct imports |
| `tests/narwhals/test_arch03_schema_driven_dispatch.py` | Behavioral tests for ARCH-03; no source-inspection tests | VERIFIED | File contains 3 behavioral tests: narwhals-engine path (kept), PySpark-pass (new), PySpark-fail (new); `import inspect` removed; `inspect.getsource` absent; `HAS_PYSPARK` guard + `pyspark_only` marker + `spark` module-scoped fixture + `_spark_env_vars` autouse fixture |
| `pandera/backends/pyspark/register.py` | Comment documenting intentional `nw.DataFrame` omission citing ibis | VERIFIED | Lines 62-65: 4-line comment before `Check.register_backend(nw.LazyFrame, ...)` explains omission, names ibis as precedent, contrasts with polars |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `test_pyspark_error.py` | `tests/pyspark/conftest.py::_cmp_errors` | `from tests.pyspark.conftest import _cmp_errors, spark_df` (line 11) | WIRED | Explicit import; 3 call sites confirmed |
| `test_pyspark_config.py::TestPanderaConfig._cmp_errors` | `tests/pyspark/conftest.py::_cmp_errors` | delegation call `_cmp_errors(actual, expected)` (line 36) | WIRED | Explicit import at line 21; single delegation body |
| `tests/narwhals/test_concat_failure_cases.py` | `pandera.backends.narwhals.base._concat_failure_cases` | `from pandera.backends.narwhals.base import _concat_failure_cases` | WIRED | Direct unit test of the function |
| `test_arch03_schema_driven_dispatch.py::test_check_dtype_pyspark_schema_pass` | `pandera.backends.narwhals.components.ColumnBackend.check_dtype` | direct method call `ColumnBackend().check_dtype(frame, schema)` | WIRED | Local import inside test; real PySpark frame + pyspark_engine dtype |
| `pandera/backends/pyspark/register.py` | ibis precedent (documentation) | comment text names `pandera/backends/ibis/register.py` explicitly | WIRED | Comment references ibis by name |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| TQ-01 | 08-01-PLAN.md | Eliminate 6 `if CONFIG.use_narwhals_backend else` ternaries from `test_pyspark_error.py` DATA assertions; extract shared `_cmp_errors` helper | SATISFIED | Zero ternaries remain; `_cmp_errors` in conftest.py; delegation from `TestPanderaConfig` |
| TQ-02 | 08-02-PLAN.md | Fix silent `pl_items` drop in `_concat_failure_cases` polars branch; add regression tests | SATISFIED | Polars branch merges via `pl.concat([lazy_result.collect()] + pl_items)`; 3-test regression file exists |
| TQ-03 | 08-03-PLAN.md | Replace 4 source-inspection tests in `test_arch03_schema_driven_dispatch.py` with behavioral equivalents | SATISFIED | 4 brittle tests deleted; 2 new PySpark-gated behavioral tests added; `import inspect` removed |
| TQ-04 | 08-03-PLAN.md | Document intentional `nw.DataFrame` omission in `pandera/backends/pyspark/register.py` | SATISFIED | 4-line comment block with ibis precedent reference added |

**Gap — TQ-01 through TQ-04 not in REQUIREMENTS.md traceability table:** These requirement IDs appear in `ROADMAP.md` Phase 8 `Requirements:` field and in all three PLAN `requirements:` frontmatter fields, but are absent from `.planning/REQUIREMENTS.md` (which does not define them or include them in the traceability table). The implementation is complete and correct; this is a documentation bookkeeping gap only — the requirements have no formal definition entry or traceability row. This does not block the phase goal, which is verifiably achieved.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `pandera/backends/narwhals/base.py` | 100-103 | `TODO(ARCH-02 follow-up):` in the PySpark branch of `_concat_failure_cases` — pre-existing; references `ARCH-02 follow-up` as a follow-up item but without a formal issue/PR number | Warning | Pre-existing marker not introduced by Phase 8; Phase 8 modified the polars branch only. The PySpark branch with this TODO is unchanged. Not a Phase 8 artifact. |

No blocking anti-patterns introduced by Phase 8.

### Human Verification Required

None. All must-haves are verifiable programmatically. The behavioral PySpark tests (`test_check_dtype_pyspark_schema_pass`, `test_check_dtype_pyspark_schema_fail`) require a PySpark environment to execute but their structure is correct and consistent with established patterns from `test_e2e.py`. No human review needed to confirm phase goal achievement.

### Gaps Summary

No gaps blocking goal achievement. All four ROADMAP.md success criteria are met:

1. **SC1 (TQ-01):** Zero `if CONFIG.use_narwhals_backend else` ternaries in `test_pyspark_error.py`; shared `_cmp_errors` helper extracted to `conftest.py` with delegation from `TestPanderaConfig`. SCHEMA assertions unchanged with direct equality.

2. **SC2 (TQ-02):** `_concat_failure_cases` polars branch now merges `pl_items` via `pl.concat([lazy_result.collect()] + pl_items)` without a `SchemaWarning`. Regression test file with 3 behavioral tests confirms the fix. PySpark branch is unchanged.

3. **SC3 (TQ-03):** All 4 `inspect.getsource` source-inspection tests deleted. `import inspect` removed. Existing 5th behavioral test preserved verbatim. Two new PySpark-gated behavioral tests (`test_check_dtype_pyspark_schema_pass`, `test_check_dtype_pyspark_schema_fail`) exercise the dispatch contract with real frames and dtypes.

4. **SC4 (TQ-04):** `pandera/backends/pyspark/register.py` has a 4-line comment documenting the intentional omission of `nw.DataFrame` registration, naming ibis as the precedent and contrasting with polars.

**Documentation note:** TQ-01 through TQ-04 are not defined in `.planning/REQUIREMENTS.md`. They exist only as a `Requirements:` line in ROADMAP.md and as `requirements:` frontmatter in the three PLAN files. The traceability table in REQUIREMENTS.md should be extended to include these IDs, but this is a bookkeeping issue and does not affect the implementation correctness.

---

_Verified: 2026-05-26T15:00:00Z_
_Verifier: Claude (gsd-verifier)_
