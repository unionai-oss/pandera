---
phase: 05-expression-based-check-protocol-eliminate-framework-specific-apply-branching
verified: 2026-03-23T06:23:47Z
status: passed
score: 10/10 must-haves verified
gaps: []
---

# Phase 5: Expression-Based Check Protocol Verification Report

**Phase Goal:** Redesign check function protocol so checks return declarative narwhals expressions, enabling `apply()` to use `frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))` uniformly for polars and ibis — eliminating the ibis row_number join hack entirely.
**Verified:** 2026-03-23T06:23:47Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | apply() has no isinstance/hasattr branching between polars and ibis in the native=False path | VERIFIED | apply() native=False path is a single `if key and key != "*": expr = self.check_fn(nw.col(key))` else branch — no backend type checks |
| 2 | The ibis row_number join block is deleted entirely | VERIFIED | `grep "row_number\|_row_col\|idx_frame\|idx_out" checks.py` returns nothing |
| 3 | apply() native=False path is `frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))` | VERIFIED | Line 74 of checks.py; confirmed by source inspection |
| 4 | apply() native=True path is unchanged | VERIFIED | native=True calls `to_native(frame)` then `check_fn(native_frame, key)` then `_normalize_native_output` — exactly as before |
| 5 | element_wise path returns `frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))` | VERIFIED | Line 52 of checks.py; try/except wraps both `map_batches` and `with_columns` call |
| 6 | All 14 builtin check functions have first arg annotated as `col_expr: nw.Expr` | VERIFIED | `grep -c "col_expr: nw.Expr" builtin_checks.py` returns 14 |
| 7 | No `frame.select()` calls remain in any builtin check function | VERIFIED | `grep "frame.select" builtin_checks.py` returns nothing |
| 8 | Dispatcher routes correctly on nw.Expr key when narwhals builtins are loaded | VERIFIED | When `pandera.backends.narwhals.builtin_checks` is imported, `nw.Expr` is present in `_function_registry`; `get_first_arg_type(equal_to)` returns `(nw.Expr,)` |
| 9 | test_builtin_check_routing and test_native_false_user_check are GREEN | VERIFIED | Both tests pass for polars and ibis backends (4/4 parametrize cases pass) |
| 10 | All 28 builtin pass + 28 builtin fail parametrize cases pass for polars and ibis | VERIFIED | 56 cases pass, 0 failures in test_builtin_checks_pass and test_builtin_checks_fail |

**Score:** 10/10 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pandera/backends/narwhals/checks.py` | Rewritten apply() — no ibis row_number join, no Dispatcher workaround | VERIFIED | 34-line apply() with 3 clean branches; no row_number, no Dispatcher import, no reassembly block |
| `pandera/backends/narwhals/builtin_checks.py` | All 14 builtins with `col_expr: nw.Expr` signature | VERIFIED | 14 functions with `col_expr: nw.Expr` annotation; all return `nw.Expr`; no `frame.select()` calls |
| `tests/backends/narwhals/test_checks.py` | Updated routing tests for expression protocol | VERIFIED | `test_builtin_check_routing` patches `_function_registry[nw.Expr]`; `test_native_false_user_check` defines `user_check(col_expr)` returning `nw.Expr` |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `builtin_checks.py` | `Dispatcher._function_registry[nw.Expr]` | `register_builtin_check` decorator reads first-arg annotation | WIRED | `get_first_arg_type(equal_to)` returns `(nw.Expr,)`; after module import, `nw.Expr` is present in `_function_registry` alongside `typing.Any` |
| `apply() native=False` | `check_fn(nw.col(key))` | expression dispatch | WIRED | Line 71: `expr = self.check_fn(nw.col(key))` — confirmed in source |
| `apply()` branches | `frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))` | uniform expression protocol | WIRED | Lines 52 and 74 of checks.py — 2 occurrences, one per non-native branch |

**Note on Dispatcher key:** The `typing.Any` key is present because the narwhals builtin functions are registered _twice_ — once during base check initialization (from `pandera.api.base.builtin_checks`) with `typing.Any`, and again by `pandera.backends.narwhals.builtin_checks` with `nw.Expr`. The Dispatcher's `__call__` dispatches on `type(args[0])`, so when `apply()` calls `check_fn(nw.col(key))`, the runtime type is `narwhals.stable.v1.Expr` which resolves to the `nw.Expr`-keyed function. This is correct behavior. The test `test_builtin_check_routing` confirms this routing works end-to-end.

---

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| EXPR-01 | 05-01, 05-03 | apply() native=False uses `frame.with_columns(expr.alias(...))` for all backends | SATISFIED | Lines 52 and 74 of checks.py; test_builtin_check_routing PASS |
| EXPR-02 | 05-02 | All 14 builtin checks pass on valid data (polars + ibis) | SATISFIED | 28 parametrize cases PASS in test_builtin_checks_pass |
| EXPR-03 | 05-02 | All 14 builtin checks fail on invalid data (polars + ibis) | SATISFIED | 28 parametrize cases PASS in test_builtin_checks_fail |
| EXPR-04 | 05-03 | element_wise=True still raises NotImplementedError on ibis | SATISFIED | test_element_wise_sql_lazy_raises[ibis] PASS |
| EXPR-05 | 05-03 | apply() returns wide table (data cols + CHECK_OUTPUT_KEY) for ibis | SATISFIED | test_apply_returns_wide_table[ibis] PASS; `_normalize_native_output` uses `native.mutate(**{CHECK_OUTPUT_KEY: out})` |
| EXPR-06 | 05-01, 05-03 | native=False check routing (after protocol change) | SATISFIED | test_native_false_user_check PASS — receives nw.Expr |
| EXPR-07 | 05-03 | native=True checks unchanged | SATISFIED | test_native_true_user_check_polars and test_native_true_user_check_ibis PASS |

All 7 EXPR requirements are satisfied. There is no REQUIREMENTS.md file (EXPR requirements are defined in ROADMAP.md and RESEARCH.md); all are covered by the three plans.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | None | — | No TODOs, FIXMEs, placeholders, or stub implementations found in any modified file |

---

### Human Verification Required

None. All phase goal behaviors are fully verifiable programmatically:

- Expression protocol uniformity: confirmed by source code inspection
- ibis row_number join removal: confirmed by grep
- Test suite outcome: confirmed by running pytest

---

## Full Test Suite Results

### test_checks.py (primary phase test file)

```
68 passed, 8 skipped, 3 xfailed, 1 xpassed
```

- All 68 non-skipped, non-xfail tests PASS
- 1 xpassed: `test_postprocess_lazyframe_no_materialization_ibis[ibis]` — was `xfail(strict=False)` from Phase 4, now unexpectedly passes due to Phase 5 improvements. Not a failure; `strict=False` means xpass is a warning, not an error.
- 3 xfailed: Phase 4 stubs that remain intentionally unfixed (polars materialization, ignore_na lazy, n_failure_cases lazy)

### Full narwhals backend suite

```
4 failed, 175 passed, 8 skipped, 4 xfailed, 5 xpassed
```

4 pre-existing failures documented in SUMMARY.md, all caused by pyarrow vs ibis.Table type disambiguation issues unrelated to Phase 5:

- `TestBuiltinChecksIbis::test_greater_than_fails_failure_cases_type` — pyarrow.Table vs ibis.Table native type assertion
- `TestBuiltinChecksIbis::test_greater_than_fails_failure_cases_values` — same
- `TestCustomChecksIbis::test_custom_check_receives_table_and_key` — DatabaseTable vs Table ibis naming
- `test_parity.py::test_failure_cases_native_ibis` — pyarrow.Table vs ibis.Table type assertion

No new failures introduced by Phase 5.

---

## Documentation Note

The ROADMAP.md still shows "2/3 plans executed" with unchecked plan boxes for Phase 5. This is a documentation-only inconsistency — the ROADMAP was not updated by the plan executor after completing plans 05-01, 05-02, and 05-03. The actual code, tests, git commits (a73672f, 6e70705, 5157df0, e22a4b7, e725b5f), and SUMMARY files all confirm all 3 plans completed successfully. This is not a code gap.

---

## Gaps Summary

No gaps. The phase goal is fully achieved:

- `apply()` is now a clean 34-line method with 3 branches and zero backend-specific code
- All 14 builtin checks accept `col_expr: nw.Expr` and return `nw.Expr` directly
- The ibis row_number join hack is completely eliminated
- Both polars and ibis backends operate uniformly through `frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))`
- All 7 EXPR requirements are satisfied
- No regressions in the full narwhals suite

---

_Verified: 2026-03-23T06:23:47Z_
_Verifier: Claude (gsd-verifier)_
