---
phase: 07-ci-fixes-and-post-review-quick-fixes
verified: 2026-05-25T00:00:00Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
---

# Phase 7: CI Fixes and Post-Review Quick Fixes — Verification Report

**Phase Goal:** Fix the two CI failures introduced by Phase 6 (CI-FIX-01, CI-FIX-02) and resolve three post-review nits (NITS-02) so that CI goes green and the PR is clean for review.
**Verified:** 2026-05-25
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

All five roadmap success criteria are verified against actual codebase state.

| #   | Truth                                                                                                          | Status     | Evidence                                                                                                           |
|-----|----------------------------------------------------------------------------------------------------------------|------------|--------------------------------------------------------------------------------------------------------------------|
| 1   | Narwhals container `check_column_in_dataframe` emits `"not in dataframe"` message (CI-FIX-01)                 | VERIFIED   | `container.py:598` — `f"column '{colname}' not in dataframe"` confirmed via grep; no "not found" present          |
| 2   | `_spark_env_vars` fixture yields (not returns) on `HAS_PYSPARK=False` branch (CI-FIX-02)                      | VERIFIED   | `test_e2e.py:712-714` — `yield` + `return` (generator-exit); AST check: 2 yields, 1 return (generator-safe)       |
| 3   | Redundant `assert native_pyspark_schema is not None` removed from `check_dtype` (NITS-02)                     | VERIFIED   | `components.py` — grep and AST walk both return 0 assert nodes targeting `native_pyspark_schema`; module imports clean |
| 4   | `noxfile.py` `tests_narwhals_backend` (and `unit_tests`) session `tests/common/` gates carry inline comment   | VERIFIED   | Both occurrences confirmed: `# tests/common/ has no pyspark marker — pytest -m pyspark would deselect every test there` appears directly above each `if extra in ("polars", "ibis"):` gate |
| 5   | "Pyspark SQL" misspellings corrected to "PySpark SQL" in `docs/source/supported_libraries.md` (NITS-02)       | VERIFIED   | `grep -c "Pyspark SQL"` → 0; `grep -c "PySpark SQL"` → 10 (7 pre-existing + 3 corrected); Sphinx refs unchanged (`<native-pyspark>` → 2, `<pyspark_sql>` → 1) |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact                                            | Expected                                                       | Status    | Details                                                                                     |
|-----------------------------------------------------|----------------------------------------------------------------|-----------|---------------------------------------------------------------------------------------------|
| `pandera/backends/narwhals/container.py`            | COLUMN_NOT_IN_DATAFRAME message reads "not in dataframe"       | VERIFIED  | Line 598: `f"column '{colname}' not in dataframe"` — exact match                           |
| `tests/narwhals/test_e2e.py`                        | `_spark_env_vars` fixture yields on every code path            | VERIFIED  | Lines 712-714: `yield` + `return` (generator-exit) on HAS_PYSPARK=False branch; HAS_PYSPARK=True branch also yields |
| `tests/ibis/test_ibis_container.py`                 | `test_column_absent_error` has xfail decorator for narwhals    | VERIFIED  | Lines 198-202: `@pytest.mark.xfail(condition=CONFIG.use_narwhals_backend, strict=True, reason="Error message format differs: 'not in dataframe' (narwhals backend) vs 'not found' (native ibis backend)")` |
| `pandera/backends/narwhals/components.py`           | `check_dtype` without redundant non-None assert                | VERIFIED  | No `assert native_pyspark_schema is not None` in file; `if uses_pyspark_dtype:` guard at line 289 preserved; subscript `native_pyspark_schema[column]` at line 297 preserved |
| `noxfile.py`                                        | Two `tests/common/` exclusion gates carry inline comment       | VERIFIED  | Lines 323 and 390: both `if extra in ("polars", "ibis"):` gates have the exact comment directly above them |
| `docs/source/supported_libraries.md`               | "PySpark SQL" correct capitalisation throughout                | VERIFIED  | 0 occurrences of "Pyspark SQL"; 10 occurrences of "PySpark SQL"; Sphinx reference targets unchanged |

### Key Link Verification

| From                                                    | To                                               | Via                                                    | Status   | Details                                                                 |
|---------------------------------------------------------|--------------------------------------------------|--------------------------------------------------------|----------|-------------------------------------------------------------------------|
| `tests/polars/test_polars_container.py::test_column_absent_error` | `pandera/backends/narwhals/container.py`  | SchemaError message "column .* not in dataframe"       | VERIFIED | container.py:598 emits `"not in dataframe"`; polars test unchanged      |
| `tests/narwhals/test_e2e.py::_spark_env_vars`           | pytest fixture protocol                          | Generator yields on every branch (no bare return)      | VERIFIED | `yield` + `return` pattern confirmed; 2 yields, 1 return (generator-exit, not non-generator return) |
| `pandera/backends/narwhals/components.py::check_dtype`  | `pyspark_engine.DataType isinstance probe`       | `if uses_pyspark_dtype:` guard proves non-None binding | VERIFIED | Guard at line 289, subscript at line 297, assert absent                 |
| `noxfile.py::tests_narwhals_backend`                    | `tests/common/` pytest invocation                | Inline comment above extra-conditional dispatch        | VERIFIED | Comment present at both occurrences                                     |

### Data-Flow Trace (Level 4)

Not applicable — this phase modifies error message text, a fixture generator, a dead assert, a comment, and documentation capitalisation. No components render dynamic data from a data source.

### Behavioral Spot-Checks

| Behavior                                                                 | Command                                                                                                   | Result                                     | Status |
|--------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|--------------------------------------------|--------|
| `_spark_env_vars` generator protocol — HAS_PYSPARK=False yields once then stops | AST: 2 yields, 1 return in `_spark_env_vars` function body                                   | yields=2, returns=1                        | PASS   |
| `pandera.backends.narwhals.components` imports cleanly after assert removal | `python3 -c "import pandera.backends.narwhals.components"` | exit 0                                    | PASS   |
| No "Pyspark SQL" remains in docs                                         | `grep -c "Pyspark SQL" docs/source/supported_libraries.md`                                               | 0                                          | PASS   |
| Two inline comments above noxfile exclusion gates                        | `grep -B1 'extra in ("polars", "ibis"):' noxfile.py` shows comment above each occurrence                | 2 occurrences matched                      | PASS   |

### Probe Execution

No probes declared in plans. No conventional probe scripts found for this phase. Step 7c: SKIPPED (no probe scripts).

### Requirements Coverage

The PLAN frontmatter declares requirement IDs `CI-FIX-01`, `CI-FIX-02`, and `NITS-02`. These IDs are referenced in `ROADMAP.md` under Phase 7 but are **not formally defined as entries in `REQUIREMENTS.md`**. The REQUIREMENTS.md file tracks the v1.3 milestone requirements (REG-01 through NITS-01). CI-FIX-01, CI-FIX-02, and NITS-02 are task-tracking labels used inline in the ROADMAP and PLAN files, not formally registered requirements.

| Requirement | Source Plan  | Description                                                          | Status        | Evidence                                             |
|-------------|--------------|----------------------------------------------------------------------|---------------|------------------------------------------------------|
| CI-FIX-01   | 07-01-PLAN   | Revert narwhals container message to "not in dataframe"              | SATISFIED     | container.py:598 confirmed; ibis xfail restored      |
| CI-FIX-02   | 07-01-PLAN   | Fix `_spark_env_vars` fixture to yield on HAS_PYSPARK=False branch  | SATISFIED     | test_e2e.py:712-714 confirmed; generator-safe        |
| NITS-02     | 07-02-PLAN   | Remove assert, add noxfile comment, fix docs capitalisation          | SATISFIED     | All three sub-items verified above                   |

**Orphaned requirements check:** `REQUIREMENTS.md` has no entries mapping to Phase 7 — CI-FIX-* and NITS-02 are ROADMAP-level labels, not REQUIREMENTS.md-level entries. No orphans; this is an intentional documentation pattern for hotfix/polish phases.

### Anti-Patterns Found

| File                                          | Line | Pattern                               | Severity | Impact                                                                                     |
|-----------------------------------------------|------|---------------------------------------|----------|--------------------------------------------------------------------------------------------|
| `tests/narwhals/test_e2e.py`                  | 327  | `# TODO: root fix is in schema construction` | WARNING | Pre-existing TODO with context but no issue reference; not introduced by this phase — in `components.py`, not this phase's file |

Note: The `# noqa: return-after-yield needed to prevent fall-through` comment on line 714 of `test_e2e.py` is correct usage documentation, not a debt marker — it explains an intentional generator-exit pattern.

No TBD, FIXME, or XXX markers introduced in files modified by this phase.

### Human Verification Required

None. All must-haves are mechanically verifiable from source code.

### Gaps Summary

No gaps. All five roadmap success criteria are met. The three NITS-02 sub-items (assert removal, noxfile comment, docs capitalisation) are each confirmed in the actual files. The two CI-FIX items (message revert, fixture yield fix) are confirmed in the actual files with the correct implementation patterns.

The SUMMARY noted a deviation from the PLAN's AST assertion (`len(returns) == 0`) — the actual fix uses `yield` + `return` (generator-exit pattern) which has `returns=1`. This is NOT a code defect: a `return` following a `yield` in a Python generator is a generator-exit statement (`StopIteration`), not a non-generator return. The behavioral requirement (fixture does not raise `ValueError: fixture did not yield a value`) is fully satisfied. The PLAN's AST check was over-specified.

---

_Verified: 2026-05-25_
_Verifier: Claude (gsd-verifier)_
