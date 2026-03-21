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

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Foundation | v1.0 | 2/2 | Complete | 2026-03-09 |
| 2. Check Backend | v1.0 | 3/3 | Complete | 2026-03-10 |
| 3. Column Backend | v1.0 | 2/2 | Complete | 2026-03-14 |
| 4. Container Backend and Polars Registration | v1.0 | 5/5 | Complete | 2026-03-14 |
| 5. Ibis Registration and Integration | v1.0 | 6/6 | Complete | 2026-03-15 |

### Phase 1: PR Review Architecture Fixes

**Goal:** Address architectural feedback from PR Review #2223 — separate ibis logic from base ErrorHandler, create NarwhalsErrorHandler, remove polars-specific coupling from narwhals container backend, fix misleading comments, and fix Narwhals capitalization.
**Requirements**: ARCH-01, ARCH-02, ARCH-03, ARCH-04
**Depends on:** v1.0 Narwhals Backend
**Plans:** 2 plans

Plans:
- [ ] 01-01-PLAN.md — ErrorHandler architecture: strip ibis from base, create NarwhalsErrorHandler
- [ ] 01-02-PLAN.md — Wire NarwhalsErrorHandler into backends, fix container polars coupling, fix comment, fix capitalization
