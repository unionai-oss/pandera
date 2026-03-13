---
phase: 02-check-backend
verified: 2026-03-09T00:00:00Z
status: passed
score: 13/13 must-haves verified
re_verification: null
gaps: []
human_verification: []
---

# Phase 2: Check Backend Verification Report

**Phase Goal:** Implement a check backend for the narwhals integration, covering builtin check dispatch, user-defined check routing, and element-wise check handling
**Verified:** 2026-03-09
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | pytest collects `tests/backends/narwhals/test_checks.py` without ImportError | VERIFIED | 85 passed, 1 skipped, 2 xfailed — no collection errors |
| 2  | `make_narwhals_frame` fixture produces `nw.LazyFrame` for polars and ibis backends | VERIFIED | `conftest.py` params=["polars","ibis"] with `nw.from_native(pl.LazyFrame(...))` and `nw.from_native(ibis.memtable(...))` |
| 3  | `NarwhalsCheckBackend` is registered for `nw.LazyFrame` in conftest fixture | VERIFIED | `Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend)` in guarded `_register_narwhals_check_backend` fixture |
| 4  | `builtin_checks` side-effect import is attempted in fixture (guarded) | VERIFIED | `try: from pandera.backends.narwhals import builtin_checks` in conftest |
| 5  | A builtin check (NarwhalsData first-arg annotation) is dispatched to `NarwhalsData` path | VERIFIED | `NarwhalsData in inner_fn._function_registry` check in `apply()` routes correctly; direct python test confirms |
| 6  | A user-defined check (no NarwhalsData annotation) receives the native frame | VERIFIED | `test_user_defined_check_routing[polars]` and `[ibis]` both PASS |
| 7  | `element_wise=True` on Ibis backend raises `NotImplementedError` with documented message | VERIFIED | `test_element_wise_sql_lazy_raises[ibis]` PASSES; message contains "SQL-lazy backends" |
| 8  | All 14 builtin checks pass on valid data for both Polars and Ibis | VERIFIED | 28 `test_builtin_checks_pass` cases (14 checks x 2 backends) all PASS |
| 9  | All 14 builtin checks fail on invalid data for both Polars and Ibis | VERIFIED | 28 `test_builtin_checks_fail` cases all PASS |
| 10 | `in_range` maps `include_min`/`include_max` to narwhals `closed=` parameter | VERIFIED | `_CLOSED_MAP` dict + `is_between(min_value, max_value, closed=closed)` — confirmed by test |
| 11 | `notin` uses `~expr` (tilde) not `.not_()` | VERIFIED | `return data.frame.select(~nw.col(data.key).is_in(forbidden_values))` |
| 12 | `str_contains` has no `literal=` kwarg; `str_matches` anchors with `^` | VERIFIED | `str.contains(pattern)` with no extra kwargs; `if not pattern.startswith("^"): pattern = f"^{pattern}"` |
| 13 | `str_length` handles exact, min-only, max-only, and range cases | VERIFIED | Four-branch `if/elif/elif/else` with `==`, `<=`, `>=`, `.is_between(closed="both")` |

**Score:** 13/13 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pandera/backends/narwhals/__init__.py` | Empty package init with docstring | VERIFIED | Exists, 1 line docstring, importable |
| `tests/backends/narwhals/conftest.py` | `make_narwhals_frame` fixture parameterized over polars/ibis; autouse registration with guarded imports | VERIFIED | 52 lines, both fixtures present, both imports guarded with `try/except ImportError` |
| `tests/backends/narwhals/test_checks.py` | 5 test functions covering all check behaviors | VERIFIED | 282 lines, 5 test functions, `BUILTIN_CHECK_CASES` list with 14 parametrized cases |
| `pandera/backends/narwhals/checks.py` | `NarwhalsCheckBackend` with `preprocess`/`apply`/`postprocess`/`__call__` | VERIFIED | 191 lines, all methods implemented and substantive |
| `pandera/backends/narwhals/builtin_checks.py` | 14 builtin check registrations with `NarwhalsData` first-arg annotation | VERIFIED | 295 lines, all 14 functions present with `@register_builtin_check` and `NarwhalsData` annotations |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `tests/backends/narwhals/conftest.py` | `pandera/backends/narwhals/checks.py` | `Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend)` | WIRED | Guarded import + registration call in `_register_narwhals_check_backend` |
| `tests/backends/narwhals/conftest.py` | `pandera/backends/narwhals/builtin_checks.py` | `try/except ImportError` guarded import | WIRED | `try: from pandera.backends.narwhals import builtin_checks` at line 49 |
| `NarwhalsCheckBackend.apply()` | `inspect.signature(check_fn)` | First-param annotation check as fallback for non-Dispatcher callables | WIRED | `Dispatcher._function_registry[NarwhalsData]` lookup is primary path; signature inspection is fallback |
| `NarwhalsCheckBackend.apply()` | `Dispatcher._function_registry` | `isinstance(inner_fn, Dispatcher) and NarwhalsData in inner_fn._function_registry` | WIRED | Verified: `NarwhalsData` in registry for `equal_to` |
| `NarwhalsCheckBackend.postprocess_lazyframe_output()` | `nw.concat` | `_materialize()` collects both frames before horizontal concat | WIRED | `_materialize(check_output)` and `_materialize(check_obj.frame)` called before `nw.concat([...], how="horizontal")` |
| `pandera/backends/narwhals/builtin_checks.py` | `pandera/api/extensions.py` | `@register_builtin_check` decorator registers into `CHECK_FUNCTION_REGISTRY` Dispatcher | WIRED | 14 `@register_builtin_check` decorators; `NarwhalsData in dispatcher._function_registry` confirmed True at runtime |
| `NarwhalsCheckBackend.apply()` element_wise path | `map_batches` | `try/except NotImplementedError` re-raise with pandera message | WIRED | `selector.map_batches(self.check_fn, return_dtype=nw.Boolean)` wrapped in try/except; ibis raises NotImplementedError |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| TEST-01 | 02-01 | `tests/backends/narwhals/` with backend-agnostic test suite parameterized for Polars and Ibis | SATISFIED | `conftest.py` with `params=["polars","ibis"]`; 85 tests run across both backends |
| CHECKS-01 | 02-02 | `NarwhalsCheckBackend` routes builtin checks to `NarwhalsData` containers and user-defined checks to native containers | SATISFIED | `test_user_defined_check_routing` passes; Dispatcher registry lookup confirmed; direct python verification passes |
| CHECKS-02 | 02-03 | 14 builtin checks in `builtin_checks.py` via narwhals Expr API | SATISFIED | 56 test cases across 14 checks x 2 backends x pass/fail all pass |
| CHECKS-03 | 02-02 | `element_wise=True` on SQL-lazy backends raises `NotImplementedError` with clear explanation | SATISFIED | `test_element_wise_sql_lazy_raises[ibis]` PASSES; message: "element_wise checks are not supported on SQL-lazy backends..." |

No orphaned requirements — all 4 requirement IDs claimed in plans and verified in codebase.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `pandera/backends/narwhals/checks.py` | 169 | `import polars as pl` inside `postprocess_bool_output` | Info | Known Phase 2 limitation — Polars used to wrap bool output in `nw.LazyFrame`. Documented in SUMMARY-02 as acceptable until Phase 4 when narwhals backend is registered for Ibis. |
| `tests/backends/narwhals/test_checks.py` | 136 | `@pytest.mark.xfail(strict=False, ...)` on `test_builtin_check_routing` | Warning | Test has a design flaw: the `capturing_fn` wrapper has no `NarwhalsData` annotation, so the backend correctly routes it as user-defined, then it calls the original builtin check_fn with a native frame instead of `NarwhalsData`, causing `KeyError` in the Dispatcher. The underlying CHECKS-01 behavior (Dispatcher routing) IS correct and verified independently. The test xfail marker is intentional and documented. |
| `pandera/backends/narwhals/checks.py` | 26-36 | `groupby`, `query`, `aggregate` raise `NotImplementedError` | Info | Intentional — same pattern as Polars template; not a stub, these behaviors are genuinely unsupported. |

No blockers found. The Polars import in `postprocess_bool_output` is a documented known limitation. The `test_builtin_check_routing` xfail is due to a test design issue (the capturing wrapper breaks the Dispatcher routing), not a backend bug — user-defined routing and builtin routing both work correctly as verified by `test_user_defined_check_routing` (passing) and direct python verification.

---

### Human Verification Required

None. All behaviors are fully verifiable from the test suite and static analysis:
- Routing dispatch is verified by passing tests and direct python invocation
- Element-wise NotImplementedError is verified by `test_element_wise_sql_lazy_raises[ibis]`
- All 14 builtin checks are verified by 56 parametrized test cases

---

### Gaps Summary

No gaps. All phase goals are achieved:

1. **Builtin check dispatch** (CHECKS-01): `NarwhalsCheckBackend.apply()` uses Dispatcher registry lookup to detect builtin narwhals checks and routes them with `NarwhalsData`. User-defined checks correctly receive the native frame via `nw.to_native()`.

2. **User-defined check routing** (CHECKS-01): `test_user_defined_check_routing` passes for both polars and ibis backends.

3. **Element-wise check handling** (CHECKS-03): SQL-lazy backends raise `NotImplementedError` with the documented message. Polars narwhals frame uses `map_batches` without error (skipped in test since polars supports it without raising).

4. **14 builtin checks** (CHECKS-02): All implemented with correct narwhals Expr API — Python comparison operators instead of `.eq/.ne` (which don't exist on narwhals Expr), `~expr` tilde for `notin`, `is_between(closed=)` for `in_range`, `str.contains()` without `literal=` kwarg.

5. **Test scaffold** (TEST-01): Backend-agnostic fixtures parameterized over polars and ibis; both `NarwhalsCheckBackend` and `builtin_checks` imports are guarded so dtype tests are unaffected.

The only notable residual item is `test_builtin_check_routing` remaining as `xfail(strict=False)` due to a test design limitation (not a backend defect). This is documented and does not block any phase goal.

---

_Verified: 2026-03-09_
_Verifier: Claude (gsd-verifier)_
