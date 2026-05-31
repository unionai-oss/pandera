---
phase: 11-round-4-pr-review-fixes
verified: 2026-05-30T00:00:00Z
status: passed
score: 6/6 must-haves verified
overrides_applied: 0
---

# Phase 11: Round-4 PR Review Fixes Verification Report

**Phase Goal:** Apply round-4 PR review fixes: unify PySpark Narwhals validation contract to raise SchemaErrors (remove is_pyspark dispatch), delete dead _handle_pyspark_validation_result method, remove dead PySpark branch from _materialize(), update PySpark tests to use backend-aware helper, and apply documentation/capitalization nits.
**Verified:** 2026-05-30
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | `DataFrameSchemaBackend.validate()` has no `is_pyspark` branch; PySpark Narwhals raises `SchemaErrors` (SE-01) | VERIFIED | `grep -v '^#' container.py | grep -c "is_pyspark"` returns 0; `raise SchemaErrors(` at line 226 of container.py |
| 2 | `_handle_pyspark_validation_result` method removed from `DataFrameSchemaBackend` (SE-01) | VERIFIED | `grep -v '^#' container.py | grep -c "_handle_pyspark_validation_result"` returns 0; method absent from container.py; regression test `test_validate_no_handle_pyspark_method_after_se01` added at line 310 of test_phase01_arch.py |
| 3 | `_materialize()` in utils.py has no `nw.Implementation.PYSPARK` check — dead branch removed (DC-01) | VERIFIED | `inspect.getsource(_materialize)` contains no PYSPARK, pyarrow, or .first(); PYSPARK entries remain only in module-level `_SQL_LAZY_IMPLEMENTATIONS` frozenset (lines 10-18); regression test `test_materialize_has_no_pyspark_branch_after_dc01` at line 330 of test_phase01_arch.py |
| 4 | `tests/pyspark/` tests use `validate_collecting_errors` helper instead of `df.pandera.errors` (SE-02) | VERIFIED | `validate_collecting_errors` defined in tests/pyspark/conftest.py at line 196; imported and used in all 7 pyspark test files and tests/narwhals/test_e2e.py; `grep -v '^#' tests/pyspark/test_pyspark_check.py | grep -c '.pandera.errors\b'` returns 0 (the 2 grep hits are commented-out lines beginning with whitespace+#) |
| 5 | `pyspark_sql.md` narwhals note updated to SchemaErrors behavior; install command simplified (SE-03, NIT-03) | VERIFIED | `grep -Fc "df.pandera.errors" pyspark_sql.md` returns 0; `grep -c "SchemaErrors" pyspark_sql.md` returns 2; `grep -Fc "pip install 'pandera[pyspark,narwhals]' pyspark" pyspark_sql.md` returns 0; install command now `pip install 'pandera[pyspark,narwhals]'` |
| 6 | `supported_libraries.md` updated to 0.32.0; proper nouns capitalized in touched files; ibis container unnecessary comment removed (NIT-01, NIT-02, NIT-04, NIT-05) | VERIFIED | `grep -c "0.26.0" supported_libraries.md` returns 0; `grep -c "0.32.0" supported_libraries.md` returns 2; all NIT-01 capitalization patterns verified (Polars LazyFrame, Ibis, PySpark in docstring; Ibis-backed Narwhals, Polars-backed Narwhals, Scalar Polars items, SQL-lazy path (Ibis, DuckDB, etc.), All-Polars path); NIT-04 comment absent from ibis/container.py; NIT-05 inner `import inspect` removed from test_phase01_arch.py function bodies (module-level import at line 10 only) |

**Score:** 6/6 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|---------|---------|--------|---------|
| `pandera/backends/narwhals/container.py` | No is_pyspark dispatch; raises SchemaErrors | VERIFIED | No is_pyspark token; no _handle_pyspark_validation_result method; `raise SchemaErrors(` at line 226 |
| `pandera/api/narwhals/utils.py` | _materialize() with no PySpark-specific code | VERIFIED | _materialize() contains no PYSPARK, pyarrow, or .first() references; _SQL_LAZY_IMPLEMENTATIONS frozenset preserves PYSPARK entries |
| `tests/narwhals/test_phase01_arch.py` | Updated with SE-01 and DC-01 regression tests; ARCH-04 tests removed | VERIFIED | Old ARCH-04 tests absent; new tests at lines 290, 310, 330 for SE-01 and DC-01 |
| `tests/pyspark/conftest.py` | `validate_collecting_errors()` helper | VERIFIED | Defined at line 196 with complete docstring; handles both native (pandera.errors attr) and narwhals (SchemaErrors raise) paths |
| `docs/source/pyspark_sql.md` | SchemaErrors contract documented; simplified install | VERIFIED | df.pandera.errors bullet removed; "Unified SchemaErrors contract" bullet added; install command simplified |
| `docs/source/supported_libraries.md` | Version reference 0.32.0 | VERIFIED | Both occurrences of 0.26.0 updated to 0.32.0 |
| `pandera/backends/narwhals/base.py` | Proper-noun capitalization in _concat_failure_cases | VERIFIED | All NIT-01 patterns capitalized: Polars LazyFrame/Ibis/PySpark, Ibis-backed Narwhals, Polars-backed Narwhals, Scalar Polars items, SQL-lazy path (Ibis, DuckDB), All-Polars path, Separate Narwhals-wrapped items |
| `pandera/backends/ibis/container.py` | No "component validate() not raising" comment | VERIFIED | Comment absent from run_schema_component_checks |
| `tests/ibis/test_ibis_container.py` | "Narwhals backend" and "native Ibis backend" capitalized | VERIFIED | `grep -Fc "Narwhals backend) vs 'not found' (native Ibis backend)"` returns 1 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `pandera/backends/narwhals/container.py` | `pandera.errors.SchemaErrors` | `raise SchemaErrors(...)` at line 226 | WIRED | Single unified raise path; no is_pyspark branch |
| `tests/pyspark/test_pyspark_container.py` | `tests/pyspark/conftest.py:validate_collecting_errors` | `from tests.pyspark.conftest import validate_collecting_errors` at line 16 | WIRED | Used at multiple call sites |
| `tests/pyspark/test_pyspark_model.py` | `tests/pyspark/conftest.py:validate_collecting_errors` | `from tests.pyspark.conftest import validate_collecting_errors` at line 17 | WIRED | Used at multiple call sites |
| `tests/pyspark/test_pyspark_check.py` | `tests/pyspark/conftest.py:validate_collecting_errors` | `from tests.pyspark.conftest import validate_collecting_errors` at line 33 | WIRED | Used at multiple call sites |
| `tests/pyspark/test_pyspark_error.py` | `tests/pyspark/conftest.py:validate_collecting_errors` | `from tests.pyspark.conftest import validate_collecting_errors` at line 12 | WIRED | Used at multiple call sites |
| `tests/pyspark/test_pyspark_config.py` | `tests/pyspark/conftest.py:validate_collecting_errors` | `from tests.pyspark.conftest import validate_collecting_errors` at line 21 | WIRED | Used at multiple call sites |
| `tests/pyspark/test_pyspark_dtypes.py` | `tests/pyspark/conftest.py:validate_collecting_errors` | `from tests.pyspark.conftest import validate_collecting_errors` at line 14 | WIRED | Used at call sites |
| `tests/pyspark/test_pyspark_accessor.py` | `tests/pyspark/conftest.py:validate_collecting_errors` | `from tests.pyspark.conftest import validate_collecting_errors` at line 13 | WIRED | Used at call site line 51 |
| `tests/narwhals/test_e2e.py` | `tests/pyspark/conftest.py:validate_collecting_errors` | `from tests.pyspark.conftest import validate_collecting_errors` at line 64 | WIRED | Used in all 4 PySpark test sections |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| SE-01 | 11-01 | Remove is_pyspark branch + _handle_pyspark_validation_result; unify to SchemaErrors | SATISFIED | Both tokens absent from container.py; `raise SchemaErrors` present; regression tests at lines 290 and 310 of test_phase01_arch.py |
| DC-01 | 11-01 | Remove dead PySpark branch from _materialize() | SATISFIED | _materialize() body has no PYSPARK, pyarrow, or .first(); regression test at line 330 of test_phase01_arch.py |
| NIT-05 | 11-01 | Hoist inner imports in narwhals test files | SATISFIED | `import inspect` appears only at module level (line 10) in test_phase01_arch.py; no inner import inspect in function bodies |
| SE-02 | 11-02 | PySpark tests use validate_collecting_errors helper instead of df.pandera.errors | SATISFIED | Helper in conftest.py; used in all 7 pyspark test files + test_e2e.py; no inline .pandera.errors access (confirmed via stricter grep excluding commented-out code) |
| SE-03 | 11-03 | pyspark_sql.md narwhals note updated for SchemaErrors contract | SATISFIED | df.pandera.errors bullet removed; SchemaErrors contract documented; test_e2e.py PySpark sections use validate_collecting_errors |
| NIT-01 | 11-03 | Proper nouns capitalized (Narwhals, Ibis, Polars) in touched files | SATISFIED | All specific capitalization patterns verified in base.py, ibis test, and pyspark conftest |
| NIT-02 | 11-03 | supported_libraries.md version 0.26.0 → 0.32.0 | SATISFIED | 0 occurrences of 0.26.0; 2 occurrences of 0.32.0 |
| NIT-03 | 11-03 | Simplified install command in pyspark_sql.md | SATISFIED | Install command is `pip install 'pandera[pyspark,narwhals]'` (no trailing `pyspark`) |
| NIT-04 | 11-03 | Remove unnecessary comment from ibis/container.py | SATISFIED | Comment absent from ibis/container.py run_schema_component_checks |

**Note:** Requirements SE-01 through NIT-05 are defined in ROADMAP.md (Phase 11 success criteria) and do not appear in REQUIREMENTS.md. REQUIREMENTS.md was last updated after Phase 6 and does not track Phase 11 requirements. The ROADMAP.md is the canonical authority for this milestone and the phase 11 requirements are fully mapped there. REQUIREMENTS.md not being updated is a documentation gap but does not block the phase goal — the ROADMAP contains all success criteria verified here.

**Note on ARCH-04 regression:** REQUIREMENTS.md marks ARCH-04 as "Complete" (Phase 4 added `_handle_pyspark_validation_result`). Phase 11 intentionally removes that method (SE-01). The ARCH-04 checkbox in REQUIREMENTS.md is now stale, but this is an intentional architectural decision (unified SchemaErrors contract) documented in the Phase 11 plan objective. The REQUIREMENTS.md description of ARCH-04 describes the extraction pattern that Phase 11 supersedes.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---------|---------|--------|--------|
| No is_pyspark token in container.py | `grep -v '^#' container.py | grep -c "is_pyspark"` | 0 | PASS |
| No _handle_pyspark_validation_result in container.py | `grep -v '^#' container.py | grep -c "_handle_pyspark_validation_result"` | 0 | PASS |
| raise SchemaErrors present in container.py | `grep -c "raise SchemaErrors" container.py` | 1 | PASS |
| No pyarrow in utils.py | `grep -c "pyarrow" utils.py` | 0 | PASS |
| No .first() in utils.py | `grep -c ".first()" utils.py` | 0 | PASS |
| PYSPARK in _materialize() function body | `python -c "import inspect; from pandera.api.narwhals.utils import _materialize; src=inspect.getsource(_materialize); assert 'PYSPARK' not in src"` | exit 0 | PASS |
| No df.pandera.errors in pyspark_sql.md | `grep -Fc "df.pandera.errors" pyspark_sql.md` | 0 | PASS |
| No 0.26.0 in supported_libraries.md | `grep -c "0.26.0" supported_libraries.md` | 0 | PASS |
| NIT-04 ibis comment removed | `grep -Fc "The component validate() not raising..." ibis/container.py` | 0 | PASS |
| validate_collecting_errors in conftest.py | `grep -c "def validate_collecting_errors" conftest.py` | 1 | PASS |
| No inline .pandera.errors in test_pyspark_check.py (excl. comments) | `grep -v '^\s*#' test_pyspark_check.py | grep -c '.pandera.errors\b'` | 0 | PASS |
| New SE-01 regression tests in test_phase01_arch.py | `grep -n "test_validate_has_no_is_pyspark_branch_after_se01"` | line 290 | PASS |
| New DC-01 regression test in test_phase01_arch.py | `grep -n "test_materialize_has_no_pyspark_branch_after_dc01"` | line 330 | PASS |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|---------|--------|
| `pandera/backends/narwhals/base.py` | 100 | `# TODO(ARCH-02 follow-up):` | Info | Pre-existing from Phase 8 (commit 4d820d66); formal requirement ID reference satisfies debt-marker gate; not introduced by Phase 11 |
| `pandera/backends/narwhals/container.py` | 288 | `# The component validate() not raising is the success signal.` | Info | This comment remains in `narwhals/container.py`; NIT-04 only targeted `ibis/container.py` per the plan; this is the narwhals backend's own instance of the same comment pattern and is out of scope for NIT-04 |

No TBD, FIXME, or XXX markers found in Phase 11 modified files.

### Human Verification Required

None — all acceptance criteria are mechanically verifiable from the codebase.

### Gaps Summary

No gaps identified. All 6 roadmap success criteria and all 9 requirement IDs (SE-01, SE-02, SE-03, DC-01, NIT-01, NIT-02, NIT-03, NIT-04, NIT-05) are satisfied by the codebase evidence.

---

_Verified: 2026-05-30_
_Verifier: Claude (gsd-verifier)_
