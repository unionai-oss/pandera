---
phase: 02-remaining-pr-review-fixes
verified: 2026-03-22T12:00:00Z
status: passed
score: 9/9 must-haves verified
re_verification:
  previous_status: gaps_found
  previous_score: 7/9
  gaps_closed:
    - "IbisCheckBackend delegation in __call__ has inline comment explaining why it exists and a TODO for the future"
    - "check_dtype uses a single narwhals-engine pass with no polars_engine or ibis_engine fallback"
  gaps_remaining: []
  regressions: []
---

# Phase 02: Remaining PR Review Fixes — Verification Report

**Phase Goal:** Address the remaining unresolved PR #2223 review comments — redesign horizontal concat in checks/components, remove Polars-specific code from postprocess_bool_output, investigate custom checks Ibis delegation, and fix backend-specific dtype logic in check_dtype.
**Verified:** 2026-03-22
**Status:** passed
**Re-verification:** Yes — after gap closure (previous score: 7/9, previous status: gaps_found)

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | postprocess_lazyframe_output combines data and check results via with_columns, not horizontal concat | VERIFIED | `combined = data_df.with_columns(results_df[CHECK_OUTPUT_KEY])` at line 149 of checks.py; no `how="horizontal"` anywhere in file |
| 2  | postprocess_bool_output creates a LazyFrame without importing polars directly | VERIFIED | `ns = nw.get_native_namespace(check_obj.frame)` + `nw.from_dict(...).lazy()` at lines 170-173 of checks.py; no `import polars` in method |
| 3  | IbisCheckBackend delegation in __call__ has inline comment explaining why it exists and a TODO for the future | VERIFIED | Comment at lines 215-217 explains backward-compat rationale and delegation mechanics. TODO at lines 218-220: "apply() should unwrap NarwhalsData to the type the check function expects (via type annotation inspection), making this IbisCheckBackend delegation unnecessary." — exact wording required by plan 02-01 must_haves now present. |
| 4  | check_nullable computes null indicator inline via with_columns and filters without materializing a separate LazyFrame | VERIFIED | `combined_lf = check_obj.with_columns(null_expr.alias(CHECK_OUTPUT_KEY))` at line 128; `_materialize(combined_lf)` at line 131 materializes the single combined frame; no `how="horizontal"` in components.py |
| 5  | check_dtype uses a single narwhals-engine pass with no polars_engine or ibis_engine fallback | VERIFIED | Three-pass polars/ibis native fallback ladder is gone; no `native_schema`, `ibis_native_dtype`, or `ibis_schema` variables. Single `narwhals_engine.Engine.dtype()` call at line 234. Two try/except blocks at lines 233-236 and 245-249 handle parametric-type fallback only — a correct and documented deviation. TODO comment at lines 242-244: "root fix is in schema construction — pandera.polars/pandera.ibis should produce narwhals engine dtypes when the Narwhals backend is active." — exact wording required by plan 02-02 must_haves now present. |
| 6  | All 125 narwhals backend tests pass | VERIFIED (prior run) | Test run from initial verification: 125 passed, 1 skipped, 3 xfailed, 4 xpassed. No functional code changes in gap-closure commits — only TODO comments added. |

**Score:** 9/9 must-haves verified (all truths fully verified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pandera/backends/narwhals/checks.py` | NarwhalsCheckBackend with three updated methods | VERIFIED | File exists, substantive, all three methods modified |
| `pandera/backends/narwhals/checks.py` | Contains `with_columns(results_df[CHECK_OUTPUT_KEY])` | VERIFIED | Line 149 confirmed |
| `pandera/backends/narwhals/checks.py` | Contains `nw.get_native_namespace` | VERIFIED | Line 170 confirmed |
| `pandera/backends/narwhals/checks.py` | IbisCheckBackend delegation block has TODO for apply() unwrapping | VERIFIED | Lines 218-220: TODO text matches plan requirement exactly |
| `pandera/backends/narwhals/components.py` | ColumnBackend with refactored check_nullable and check_dtype | VERIFIED | File exists, both methods refactored |
| `pandera/backends/narwhals/components.py` | Contains `with_columns(null_expr.alias(CHECK_OUTPUT_KEY))` | VERIFIED | Line 128 confirmed |
| `pandera/backends/narwhals/components.py` | Contains `"TODO: root fix is in schema construction"` | VERIFIED | Lines 242-244: TODO text present, matches plan requirement |
| `pandera/engines/narwhals_engine.py` | Engine.dtype() extended for cross-engine inputs | VERIFIED | Lines 92-110+ confirm cross-engine dtype extension (deviation from plan — correctly implemented per SUMMARY 02-02) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `postprocess_lazyframe_output` | `data_df.with_columns(results_df[CHECK_OUTPUT_KEY])` | `nw.DataFrame.with_columns(Series)` | VERIFIED | Line 149: exact pattern match |
| `postprocess_bool_output` | `nw.from_dict` | `nw.get_native_namespace(check_obj.frame)` | VERIFIED | Lines 170-173: pattern confirmed |
| `check_nullable` | `combined_df.filter(nw.col(CHECK_OUTPUT_KEY)).select(col)` | `check_obj.with_columns(null_expr.alias(CHECK_OUTPUT_KEY))` | VERIFIED | Line 128: `with_columns(null_expr.alias(...))` confirmed; filter at line 144 on materialized `combined_df` |
| `check_dtype` | `narwhals_engine.Engine.dtype(nw_dtype)` | single try/except, no second/third pass | VERIFIED | Lines 234, 246: Engine.dtype confirmed; no `native_schema`, `ibis_native_dtype`, `ibis_schema` in file |

### Requirements Coverage

No explicit requirement IDs were declared in either plan's `requirements:` field. The phase goal maps to PR #2223 review comments; no REQUIREMENTS.md entries to cross-reference.

### Anti-Patterns Found

No new anti-patterns introduced. Previously flagged missing TODOs are now resolved:

| File | Line | Pattern | Severity | Status |
|------|------|---------|----------|--------|
| `checks.py` | 218-220 | Missing TODO for apply() unwrapping | Warning | RESOLVED — TODO added |
| `components.py` | 242-244 | Missing TODO for schema construction root fix | Warning | RESOLVED — TODO added |

No stub implementations, placeholder returns, or empty handlers found in the modified files.

### Human Verification Required

None — all checks are programmatically verifiable.

### Re-verification Summary

Both gaps from the initial verification (2026-03-22) are now closed:

**Gap 1 — IbisCheckBackend delegation TODO (checks.py) — CLOSED:** The delegation block at lines 214-223 now contains the required TODO at lines 218-220: "apply() should unwrap NarwhalsData to the type the check function expects (via type annotation inspection), making this IbisCheckBackend delegation unnecessary." This is the exact wording specified in plan 02-01 must_haves.

**Gap 2 — check_dtype schema construction TODO (components.py) — CLOSED:** The required TODO phrase "root fix is in schema construction" is now present at line 242 within a multi-line comment block (lines 238-244) that fully documents the known limitation. The wording satisfies the plan 02-02 must_haves artifact `contains: "TODO: root fix is in schema construction"`.

No regressions detected: all previously-passing key patterns (`with_columns(results_df[CHECK_OUTPUT_KEY])`, `nw.get_native_namespace`, `with_columns(null_expr.alias(...))`, `narwhals_engine.Engine.dtype`) remain intact. No `how="horizontal"` or `import polars as pl` reintroduced. No `native_schema`/`ibis_native_dtype`/`ibis_schema` variables present.

---

_Verified: 2026-03-22_
_Verifier: Claude (gsd-verifier)_
