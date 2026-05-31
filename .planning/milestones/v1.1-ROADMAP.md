# Roadmap: Pandera Narwhals Backend

## Milestones

- ✅ **v1.0 Narwhals Backend** — Phases 1-5 (shipped 2026-03-15)

## Phases

<details>
<summary>✅ v1.0 Narwhals Backend (Phases 1-5) — SHIPPED 2026-03-15</summary>

- [x] Phase 1: Foundation (2/2 plans) — completed 2026-03-09
- [x] Phase 2: Check Backend (3/3 plans) — completed 2026-03-10
- [x] Phase 3: Column Backend (2/2 plans) — completed 2026-03-14
- [x] Phase 4: Container Backend and Polars Registration (5/5 plans) — completed 2026-03-14
- [x] Phase 5: Ibis Registration and Integration (6/6 plans) — completed 2026-03-15

See `.planning/milestones/v1.0-ROADMAP.md` for full phase details.

</details>

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. PR Review Architecture Fixes | 3/3 | Complete | 2026-03-22 |
| 2. Remaining PR Review Fixes | 2/2 | Complete | 2026-03-22 |
| 3. Fix IbisCheckBackend delegation via apply() type-dispatch | 2/2 | Complete | 2026-03-22 |
| 4. Lazy postprocess — always-lazy failure_cases | 3/3 | Complete | 2026-03-23 |
| 5. Expression-based check protocol | 3/3 | Complete | 2026-03-23 |
| 6. Eliminate unnecessary materialization | 3/3 | Complete | 2026-03-23 |
| 7. v1.0 Tech Debt Cleanup | 2/2 | Complete   | 2026-03-24 |
| 8. Fix lazy=True critical regressions | 2/2 | Complete   | 2026-03-25 |
| 9. Accumulate check outputs into single wide table for narwhals-idiomatic drop_invalid_rows | 2/2 | Complete | 2026-03-25 |

### Phase 1: PR Review Architecture Fixes

**Goal:** Address architectural feedback from PR Review #2223 — separate ibis logic from base ErrorHandler, create NarwhalsErrorHandler, remove polars-specific coupling from narwhals container backend, fix misleading comments, and fix Narwhals capitalization.
**Requirements**: ARCH-01, ARCH-02, ARCH-03, ARCH-04
**Depends on:** v1.0 Narwhals Backend
**Plans:** 3 plans

Plans:
- [x] 01-01-PLAN.md — ErrorHandler architecture: strip ibis from base, create NarwhalsErrorHandler (1/1 complete)
- [x] 01-02-PLAN.md — Wire NarwhalsErrorHandler into backends, fix container polars coupling, fix comment, fix capitalization
- [x] 01-03-PLAN.md — Gap closure: fix validate() premature materialization, remaining capitalization nits, ROADMAP marker

### Phase 2: Remaining PR Review Fixes

**Goal:** Address the remaining unresolved PR #2223 review comments — redesign horizontal concat in checks/components, remove Polars-specific code from postprocess_bool_output, investigate custom checks Ibis delegation, and fix backend-specific dtype logic in check_dtype.
**Requirements**: TBD
**Depends on:** Phase 1
**Plans:** 2 plans

Plans:
- [x] 02-01-PLAN.md — checks.py: replace horizontal concat with with_columns, replace polars import in postprocess_bool_output, document IbisCheckBackend delegation
- [x] 02-02-PLAN.md — components.py: refactor check_nullable to with_columns, simplify check_dtype to single narwhals-engine pass

### Phase 3: Fix IbisCheckBackend delegation via apply() type-dispatch

**Goal:** Remove IbisCheckBackend delegation from NarwhalsCheckBackend by introducing a native flag on Check that controls what apply() passes to the check function. Unify the calling convention for all checks to check_fn(frame, key). No new user-facing capabilities — purely architectural clean-up.
**Requirements**: TBD
**Depends on:** Phase 2
**Plans:** 2/2 plans complete

Plans:
- [x] 03-01-PLAN.md — Add native param to Check, propagate native=False for builtins, refactor all 14 builtin check signatures (1/1 complete — 2026-03-22)
- [x] 03-02-PLAN.md — Rewrite apply() with native-flag dispatch, remove ibis delegation from __call__, add normalization helper and tests

### Phase 4: Lazy postprocess — always-lazy failure_cases

**Goal:** Make `postprocess_lazyframe_output` fully lazy — `apply()` attaches `CHECK_OUTPUT_KEY` to the full frame via `with_columns` (returning the same lazy type as input), `postprocess_lazyframe_output` builds `passed` and `failure_cases` lazily without materializing `check_obj.frame`, and materialization only happens in `run_check` when evaluating the scalar `passed` boolean. Fixes `failure_cases` being `pyarrow.Table` for ibis builtin checks — it will instead be a narwhals-wrapped lazy ibis Table.
**Requirements**: LAZY-01, LAZY-02, LAZY-03, LAZY-04, LAZY-05, LAZY-06, LAZY-07, LAZY-08
**Depends on:** Phase 3
**Plans:** 3/3 plans complete — completed 2026-03-23

Plans:
- [x] 04-01-PLAN.md — Write failing test stubs + update ibis e2e failure_cases assertions (RED baseline)
- [x] 04-02-PLAN.md — Rewrite apply() wide-table + lazy postprocess_lazyframe_output (checks.py only)
- [x] 04-03-PLAN.md — Remove _to_native from run_check + narwhals-ify failure_cases_metadata (base.py only)

### Phase 5: Expression-based check protocol — eliminate framework-specific apply() branching

**Goal:** Redesign check function protocol so checks return declarative narwhals expressions, enabling `apply()` to use `frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))` uniformly for polars and ibis — eliminating the ibis row_number join hack entirely.
**Requirements**: EXPR-01, EXPR-02, EXPR-03, EXPR-04, EXPR-05, EXPR-06, EXPR-07
**Depends on:** Phase 4
**Plans:** 3 plans — completed 2026-03-23

Plans:
- [x] 05-01-PLAN.md — Update routing tests to nw.Expr protocol (RED baseline)
- [x] 05-02-PLAN.md — Rewrite all 14 builtin checks: nw.Expr in, nw.Expr out (Dispatcher re-keyed)
- [x] 05-03-PLAN.md — Rewrite apply() — delete ibis row_number join, Dispatcher workaround, reassembly block

### Phase 6: Eliminate unnecessary materialization — lazy-first failure_cases and check_output

**Goal:** Enforce a single principle throughout the narwhals backend: execution is triggered only once — to evaluate the scalar boolean "did the check pass?" — and everything else is returned in the user's original type. `failure_cases` and `check_output` must stay as lazy ibis Tables when the input was ibis, as `pl.LazyFrame` when the input was polars lazy, etc. The user calls `nw.to_native()` to unwrap; pandera never calls `.collect()` or `.execute()` on their behalf except for the pass/fail boolean. This collapses the dead `_is_ibis_result` bifurcation in `run_check()`, removes the spurious `fc.collect()` and `_materialize(check_output)` calls, fixes `subsample()` materializing before `.head()`/`.tail()`, and fixes `check_nullable()` materializing the whole frame to evaluate a scalar `.any()`.
**Requirements**: TBD
**Depends on:** Phase 5
**Plans:** 3 plans — completed 2026-03-23

Plans:
- [x] 06-01-PLAN.md — RED baseline tests: subsample() lazy-first contracts + failure_cases type contracts
- [x] 06-02-PLAN.md — run_check() unified (no _is_ibis_result), check_nullable() scalar-only, SchemaError.failure_cases native
- [x] 06-03-PLAN.md — failure_cases_metadata redesign: ibis.Table in SchemaErrors.failure_cases

### Phase 7: v1.0 Tech Debt Cleanup

**Goal:** Address all tech debt identified in the v1.0 milestone audit — fix dead code in `_count_failure_cases`, update the `Check.native` docstring to reflect the current expression-based API, fix the ibis API rename in `test_custom_check_receives_table_and_key`, promote 4 xpassed tests to strict passing, delete one hollow test, and mark stale ROADMAP.md plan checkboxes as complete.
**Requirements:** Tech debt from v1.0 audit
**Depends on:** Phase 6
**Plans:** 2/2 plans complete

Plans:
- [x] 07-01-PLAN.md — Code correctness: fix `_count_failure_cases` dead branch, fix ibis DatabaseTable→Table rename
- [x] 07-02-PLAN.md — Docs & test hygiene: update Check.native docstring, promote 4 xpassed tests, delete hollow test, fix stale ROADMAP checkboxes

### Phase 8: Fix lazy=True critical regressions

**Goal:** Close the two critical integration breaks found in the v1.0 post-audit: (1) `failure_cases_metadata()` collapsing N polars lazy failure rows to a single repr string, and (2) `_count_failure_cases()` crashing with `TypeError` when `failure_cases` is a bool scalar. Both fixes must be narwhals-idiomatic — no native type-dependent `isinstance` checks — and the lazy=True path must work for both polars and ibis.
**Requirements:** MISSING-01, MISSING-02 (gap closure from v1.0 audit)
**Gap Closure:** Closes MISSING-01, MISSING-02, FLOW-BROKEN-01, FLOW-BROKEN-02
**Depends on:** Phase 7
**Plans:** 2/2 plans complete

Plans:
- [ ] 08-01-PLAN.md — RED baseline: write 3 failing regression tests covering MISSING-01 (polars + ibis per-row failure_cases) and MISSING-02 (bool scalar crash)
- [ ] 08-02-PLAN.md — GREEN: fix failure_cases_metadata() ibis-only rewrap → unified nw.from_native guard; fix _count_failure_cases() → try/except TypeError scalar fallback

### Phase 9: Accumulate check outputs into single wide table for narwhals-idiomatic drop_invalid_rows

**Goal:** Refactor the narwhals backend check loop so that per-check boolean outputs accumulate into a single wide table during iteration, enabling drop_invalid_rows to be a pure narwhals all_horizontal filter — no backend-specific isinstance checks, no IbisSchemaBackend delegation.
**Requirements**: DIR-01, DIR-02, DIR-03, DIR-04, DIR-05, DIR-06, DIR-07
**Depends on:** Phase 8
**Plans:** 2 plans

Plans:
- [x] 09-01-PLAN.md — RED baseline: confirm 20 failing drop_invalid_rows tests, add xfail parity test
- [x] 09-02-PLAN.md — GREEN: apply() returns nw.Expr, add postprocess_expr_output(), replace drop_invalid_rows() with nw.all_horizontal accumulation
