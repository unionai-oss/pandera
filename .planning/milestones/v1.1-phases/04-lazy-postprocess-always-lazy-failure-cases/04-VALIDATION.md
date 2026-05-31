---
phase: 4
slug: lazy-postprocess-always-lazy-failure-cases
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-22
audited: 2026-03-24
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | pyproject.toml |
| **Quick run command** | `pytest tests/backends/narwhals/test_checks.py tests/backends/narwhals/test_e2e.py -x -q` |
| **Full suite command** | `pytest tests/backends/narwhals/ -q` |
| **Estimated runtime** | ~2 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/backends/narwhals/test_checks.py tests/backends/narwhals/test_e2e.py -x -q`
- **After every plan wave:** Run `pytest tests/backends/narwhals/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 4-01-01 | 01 | 0 | LAZY-01,02,03,07,08 | unit stubs | `pytest tests/backends/narwhals/test_checks.py -k "wide_table or postprocess_lazy or ignore_na or n_failure_cases" -x -q` | ✅ | ✅ green |
| 4-01-02 | 01 | 0 | LAZY-04,05 | e2e update | `pytest tests/backends/narwhals/test_e2e.py -k "TestBuiltinChecksIbis" -x -q` | ✅ | ✅ green |
| 4-01-03 | 01 | 1 | LAZY-01 | unit | `pytest tests/backends/narwhals/test_checks.py -k "wide_table" -x -q` | ✅ | ✅ green |
| 4-01-04 | 01 | 1 | LAZY-02,03 | unit | `pytest tests/backends/narwhals/test_checks.py -k "postprocess_lazyframe" -x -q` | ✅ | ✅ green |
| 4-01-05 | 01 | 1 | LAZY-07,08 | unit | `pytest tests/backends/narwhals/test_checks.py -k "ignore_na_lazy or n_failure_cases_lazy" -x -q` | ✅ | ✅ green |
| 4-01-06 | 01 | 2 | LAZY-04,05 | e2e | `pytest tests/backends/narwhals/test_e2e.py -k "TestBuiltinChecksIbis and (failure_cases_type or failure_cases_values)" -x -q` | ✅ | ✅ green |
| 4-01-07 | 01 | 2 | LAZY-06 | e2e regression | `pytest tests/backends/narwhals/test_e2e.py -k "TestBuiltinChecksPolars" -x -q` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Requirement Coverage

| Requirement | Test | Status |
|-------------|------|--------|
| LAZY-01: apply() returns wide table | `test_apply_returns_wide_table[polars,ibis]` | ✅ green |
| LAZY-02: no materialization in postprocess (polars) | `test_postprocess_lazyframe_no_materialization_polars[polars]` | ✅ green |
| LAZY-03: no materialization in postprocess (ibis) | `test_postprocess_lazyframe_no_materialization_ibis[ibis]` | ✅ green |
| LAZY-04: failure_cases is nw.DataFrame (ibis) | `test_greater_than_fails_failure_cases_type` | ✅ green |
| LAZY-05: failure_cases values via nw.to_native().execute() | `test_greater_than_fails_failure_cases_values` | ✅ green |
| LAZY-06: polars regression — no regressions | `TestBuiltinChecksPolars` (all 7) | ✅ green |
| LAZY-07: ignore_na=True treats None as pass (lazy) | `test_ignore_na_lazy[polars]` | ✅ green |
| LAZY-08: n_failure_cases limits failure rows lazily | `test_n_failure_cases_lazy[polars]` | ✅ green |

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** complete

---

## Validation Audit 2026-03-24

| Metric | Count |
|--------|-------|
| Gaps found | 3 |
| Resolved | 3 |
| Escalated | 0 |

**Gaps resolved:**
- LAZY-02: Fixed `test_postprocess_lazyframe_no_materialization_polars` — assertion corrected from `nw.DataFrame` to `(nw.LazyFrame, nw.DataFrame)` (polars filter on LazyFrame stays lazy)
- LAZY-07: Fixed `test_ignore_na_lazy` — added missing `ignore_na=True` to Check; fixed `len()` on LazyFrame via `.collect()` guard
- LAZY-08: Fixed `test_n_failure_cases_lazy` — fixed `len()` on LazyFrame via `.collect()` guard before `nw.to_native()`
