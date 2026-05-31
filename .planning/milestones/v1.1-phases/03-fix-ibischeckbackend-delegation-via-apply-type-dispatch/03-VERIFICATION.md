---
phase: 03-fix-ibischeckbackend-delegation-via-apply-type-dispatch
verified: 2026-03-22T17:15:00Z
status: passed
score: 12/12 must-haves verified
re_verification: false
---

# Phase 03: Fix IbisCheckBackend Delegation via apply() Type Dispatch — Verification Report

**Phase Goal:** Remove IbisCheckBackend delegation from NarwhalsCheckBackend by introducing a native flag on Check that controls what apply() passes to the check function. Unify the calling convention for all checks to check_fn(frame, key). No new user-facing capabilities — purely architectural clean-up.
**Verified:** 2026-03-22T17:15:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | Check.__init__ accepts `native: bool = True` and stores it as `self.native` | VERIFIED | `pandera/api/checks.py` line 28 (`native: bool = True` param), line 184 (`self.native = native`) |
| 2  | `from_builtin_check_name` passes `native=False` when constructing builtin Check objects | VERIFIED | `pandera/api/base/checks.py` line 143: `native=False` explicit keyword before `**kws` |
| 3  | All 14 builtin narwhals check functions accept `(frame: nw.LazyFrame, key: str, ...)` instead of `(data: NarwhalsData, ...)` | VERIFIED | All 14 functions in `builtin_checks.py` confirmed with `(frame, key, ...)` signature; `def equal_to(frame: nw.LazyFrame, key: str, value: Any)` etc. |
| 4  | Builtin function bodies use `frame` and `key` directly instead of `data.frame` and `data.key` | VERIFIED | No occurrences of `data.frame` or `data.key` in `builtin_checks.py`; `NarwhalsData` import removed |
| 5  | `NarwhalsCheckBackend.apply()` dispatches on `self.check.native` flag — three explicit branches | VERIFIED | `checks.py` lines 43-82: `if self.check.element_wise` / `elif self.check.native` / `else` (native=False) |
| 6  | native=True user-defined checks receive `(native_frame, key)` as two positional args | VERIFIED | Lines 60-61: `native_frame = nw.to_native(check_obj.frame); out = self.check_fn(native_frame, check_obj.key)` |
| 7  | native=False builtin checks receive `(nw.LazyFrame, key)` as two positional args | VERIFIED | Lines 82: `out = check_fn(check_obj.frame, check_obj.key)` in the else branch |
| 8  | `_normalize_native_output` static method exists and handles ibis output types | VERIFIED | Lines 100-121: handles `ir.BooleanScalar`, `ir.BooleanColumn`, `ibis.Table`, and passthrough |
| 9  | The ibis delegation block in `__call__` is removed entirely | VERIFIED | `__call__` is exactly 4 functional lines: preprocess, NarwhalsData, apply, postprocess — no ibis delegation |
| 10 | All 14 builtin checks pass on valid Polars data after refactor | VERIFIED | 99 tests pass, 4 skipped (ibis skipped — ibis not installed in env); all BUILTIN_CHECK_CASES pass |
| 11 | `inspect` import removed from `checks.py` | VERIFIED | No `import inspect` found at top-level; no `inspect.` usage in file |
| 12 | `IbisCheckBackend` no longer delegated to from `NarwhalsCheckBackend` | VERIFIED | No `IbisCheckBackend` string found anywhere in `checks.py` |

**Score:** 12/12 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pandera/api/checks.py` | `native: bool = True` param in `Check.__init__`, `self.native = native` | VERIFIED | Lines 28 and 184 confirmed; docstring documents the param |
| `pandera/api/base/checks.py` | `native=False` propagation in `from_builtin_check_name` | VERIFIED | Line 143: `native=False` keyword in `cls(...)` call before `**kws` |
| `pandera/backends/narwhals/builtin_checks.py` | All 14 functions with `(frame: nw.LazyFrame, key: str, ...)` signature; `NarwhalsData` not imported | VERIFIED | 14 functions confirmed; no `NarwhalsData` import present |
| `pandera/backends/narwhals/checks.py` | `def _normalize_native_output`; `self.check.native` in apply(); no ibis delegation in `__call__` | VERIFIED | All three confirmed at lines 101, 58, and 207-215 |
| `tests/backends/narwhals/test_checks.py` | `test_native_true_user_check_polars` and related new tests | VERIFIED | Tests present at lines 301, 330, 359, 382 |
| `tests/core/test_checks.py` | 33 tests for native flag and builtin signatures | VERIFIED | File exists, 33 test cases across TestNativeFlag, TestBuiltinNativeFalse, TestBuiltinCheckSignatures |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `pandera/api/base/checks.py` `from_builtin_check_name` | `pandera/api/checks.py` `Check.__init__` | `native=False` explicit keyword arg | WIRED | Line 143 of base/checks.py passes `native=False`; Check.__init__ line 184 stores it |
| `pandera/backends/narwhals/checks.py` `apply()` | `self.check.native` | `elif self.check.native` branch | WIRED | Line 58 reads `self.check.native` to dispatch |
| `_normalize_native_output` | `ibis.expr.types` | `try: import ibis; isinstance(out, ir.BooleanScalar)` | WIRED | Lines 106-120: guarded import with fallback for missing ibis |
| `builtin_checks.py` functions | `NarwhalsCheckBackend.apply()` native=False branch | `check_fn(check_obj.frame, check_obj.key)` | WIRED | Line 82 calls check_fn with (frame, key); partial binding provides extra kwargs transparently |

---

### Requirements Coverage

Both plans declare `requirements: []` in their frontmatter. The ROADMAP declares `Requirements: TBD` for Phase 03. No formal requirement IDs exist for this phase — coverage assessment is not applicable.

No REQUIREMENTS.md found in `.planning/` (file does not exist). No orphaned requirements to report.

---

### Notable Deviation (Not a Gap)

**Dispatcher retained in native=False branch:** Plan 02 truth stated "no Dispatcher or inspect.signature usage" but the implementation keeps `Dispatcher` inside the `native=False` else-branch (lines 68-80). This is intentional — ibis frames arrive as `nw.DataFrame` (not `nw.LazyFrame`), causing a `KeyError` in the Dispatcher registry which is keyed on `nw.LazyFrame`. The fix looks up the `nw.LazyFrame` implementation directly. The SUMMARY (03-02) documents this as a Rule 1 bug fix. The phase goal — removing IbisCheckBackend delegation — is fully achieved. The Dispatcher usage is internal plumbing, not a user-visible delegation pattern.

**ROADMAP checkbox for plan 03-02:** The ROADMAP shows `[ ]` (unchecked) for 03-02 despite commits `b43548e` and `0c358d0` implementing the plan. This is a documentation-only gap — no functional impact.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None found | — | — | — | — |

No TODO/FIXME/placeholder comments, empty implementations, or stub returns found in any modified file.

---

### Human Verification Required

#### 1. Ibis backend end-to-end test

**Test:** Install ibis-framework, run `pytest tests/backends/narwhals/test_checks.py -x -q` in an environment with ibis available.
**Expected:** The 4 ibis-skipped tests (`test_native_true_user_check_ibis`, `test_ibis_boolean_scalar_normalization`) should pass; `test_builtin_checks_pass`/`fail` should pass for ibis fixture too.
**Why human:** ibis is not installed in the current environment — the 4 ibis-specific tests are skipped. Automated verification confirmed the code paths are present and correct but cannot execute the ibis paths.

---

### Gaps Summary

No gaps found. All 12 must-have truths verified against the actual codebase. The phase goal is fully achieved:

- `Check.native` flag exists and stores correctly (True by default, False for all builtins)
- `from_builtin_check_name` always creates `Check` with `native=False`
- All 14 builtin functions use `(frame, key, ...)` signature — `NarwhalsData` wrapper eliminated
- `NarwhalsCheckBackend.apply()` dispatches purely on `self.check.native` — no inspect.signature magic
- `IbisCheckBackend` is no longer delegated to from `NarwhalsCheckBackend.__call__`
- `_normalize_native_output` normalizes ibis output types for native=True checks
- Test suite: 99 passing, 4 skipped (ibis not installed)

The one minor deviation (Dispatcher kept inside native=False branch for ibis nw.DataFrame compatibility) is a bug fix acknowledged in the SUMMARY, not a regression. The architectural goal — unified `check_fn(frame, key)` calling convention with no IbisCheckBackend delegation — is realized.

---

_Verified: 2026-03-22T17:15:00Z_
_Verifier: Claude (gsd-verifier)_
