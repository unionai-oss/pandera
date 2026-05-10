# Roadmap: Pandera Narwhals Backend

## Milestones

- ✅ **v1.0 Narwhals Backend** — Phases 1-5 (shipped 2026-03-15)
- ✅ **v1.1 Ibis Parity & Lazy-First Architecture** — Phases 1-9 (shipped 2026-03-25)
- 🚧 **v1.2 PR Review Cleanup & Test Strategy** — Phases 1-3 (in progress)

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

<details>
<summary>✅ v1.1 Ibis Parity & Lazy-First Architecture (Phases 1-9) — SHIPPED 2026-03-25</summary>

- [x] Phase 1: PR Review Architecture Fixes (3/3 plans) — completed 2026-03-22
- [x] Phase 2: Remaining PR Review Fixes (2/2 plans) — completed 2026-03-22
- [x] Phase 3: Fix IbisCheckBackend delegation via apply() type-dispatch (2/2 plans) — completed 2026-03-22
- [x] Phase 4: Lazy postprocess — always-lazy failure_cases (3/3 plans) — completed 2026-03-23
- [x] Phase 5: Expression-based check protocol (3/3 plans) — completed 2026-03-23
- [x] Phase 6: Eliminate unnecessary materialization (3/3 plans) — completed 2026-03-23
- [x] Phase 7: v1.0 Tech Debt Cleanup (2/2 plans) — completed 2026-03-24
- [x] Phase 8: Fix lazy=True critical regressions (2/2 plans) — completed 2026-03-25
- [x] Phase 9: Accumulate check outputs for drop_invalid_rows (2/2 plans) — completed 2026-03-25

See `.planning/milestones/v1.1-ROADMAP.md` for full phase details.

</details>

### 🚧 v1.2 PR Review Cleanup & Test Strategy (In Progress)

**Milestone Goal:** Address all feedback from PR review 4027330818 — eliminate backend-specific coupling, unify native type detection, fix eager execution, fix custom checks, polish documentation, and establish a cohesive CI test strategy.

- [x] **Phase 1: Structural Cleanup** — Unify type detection, isolate backend code, eliminate eager execution, fix custom checks (completed 2026-03-30)
- [x] **Phase 2: Documentation Polish** — Clarify `native` param scope, fix "Narwhals" capitalization everywhere (completed 2026-04-10)
- [ ] **Phase 3: CI Test Strategy** — CI infrastructure in place; xfails for narwhals backend gaps pending

## Phase Details

### Phase 1: Structural Cleanup
**Goal**: The narwhals backend has no Polars-specific coupling, no unnecessary eager execution, unified type detection utilities, and custom checks work end-to-end
**Depends on**: Nothing (first phase of v1.2)
**Requirements**: TYPES-01, TYPES-02, TYPES-03, CLEAN-01, CLEAN-02, CLEAN-03, CLEAN-04, EAGER-01, EAGER-02, CHECKS-01
**Success Criteria** (what must be TRUE):
  1. A single `_is_lazy(frame)` utility (or equivalent constants) is used everywhere in `narwhals/` in place of ad-hoc `isinstance`/`hasattr` checks — no inline lazy-detection logic remains in `base.py`, `container.py`, or `components.py`
  2. `narwhals/checks.py`, `narwhals/container.py`, and `narwhals/base.py` contain no Polars-specific imports and no `pandera.api.polars` imports
  3. `narwhals_engine.py` and `container.py`/`components.py` do not call `.collect()` on entire frames for coerce, concat, or dtype-check operations
  4. All inner imports (stdlib and narwhals engine) in `container.py` and `narwhals_engine.py` are moved to module-level top-of-file
  5. A user-defined custom check passes validation through the Narwhals backend for both `pl.DataFrame` and `ibis.Table` inputs, and a regression test covers this case
**Plans**: 5 plans

Plans:
- [x] 01-01-PLAN.md — Create _is_lazy utility, hoist inner imports, add TYPE_CHECKING guard (TYPES-01, TYPES-02, CLEAN-04)
- [x] 01-02-PLAN.md — Replace full-frame collect in try_coerce with head(1) probe; audit container/components materializations (EAGER-01, EAGER-02)
- [x] 01-03-PLAN.md — Rewrite failure_cases_metadata dispatch to use _is_lazy; eliminate unconditional polars imports from base.py (TYPES-03, CLEAN-03)
- [x] 01-04-PLAN.md — Add infer_columns() to DataFrameSchema base; wire container.py; add CLEAN-01/CLEAN-02 arch tests (CLEAN-01, CLEAN-02)
- [x] 01-05-PLAN.md — Fix _normalize_native_output for pl.Series/pl.DataFrame returns; add regression tests (CHECKS-01)

### Phase 2: Documentation Polish
**Goal**: Docstrings and comments accurately describe the `native` parameter's scope and consistently spell "Narwhals" with a capital N
**Depends on**: Phase 1
**Requirements**: DOCS-01, DOCS-02
**Success Criteria** (what must be TRUE):
  1. The `native` parameter docstring in `pandera/api/checks.py` states explicitly that it only applies when using the Narwhals backend
  2. All occurrences of "narwhals" in comments, docstrings, and `register.py` files that refer to the library name are capitalized as "Narwhals"
**Plans**: 2 plans

Plans:
- [x] 02-01-PLAN.md — Append Narwhals-backend caveat to :param native: docstring in pandera/api/checks.py (DOCS-01)
- [x] 02-02-PLAN.md — Capitalize "Narwhals" prose across pandera/ .py files, preserve code identifiers (DOCS-02)

### Phase 3: CI Test Strategy
**Goal**: The test suite is structured so existing Polars/Ibis backend tests run cleanly without Narwhals installed, and the Narwhals backend tests exercise all supported frame types with a documented CI matrix
**Depends on**: Phase 1
**Requirements**: TEST-01, TEST-02, TEST-03
**Success Criteria** (what must be TRUE):
  1. Existing Polars and Ibis backend tests (`tests/backends/polars/`, `tests/backends/ibis/`) pass (or are `xfail`-marked with a comment justifying the mark) when Narwhals is installed alongside those backends
  2. Narwhals backend tests in `tests/backends/narwhals/` are parametrized and each test case runs against `pl.DataFrame`, `pl.LazyFrame`, and `ibis.Table` — no frame type is silently skipped
  3. A CI matrix is documented (in code comments, a conftest note, or tox/pixi config) that covers: (a) existing backends tested in an environment without Narwhals installed, and (b) Narwhals backend tested with all supported frame types
**Plans**: TBD

Plans:
- [ ] TBD (run /gsd:plan-phase 3 to break down)

## Progress

**Execution Order:** 1 → 2 → 3 (Phase 2 and Phase 3 are independent once Phase 1 is complete)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Structural Cleanup | 6/6 | Complete   | 2026-04-10 |
| 2. Documentation Polish | 2/2 | Complete   | 2026-04-10 |
| 3. CI Test Strategy | 3/3 | In Progress | — |
