---
phase: 8
slug: fix-lazy-true-critical-regressions
status: final
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-25
finalized: 2026-03-25
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `pytest tests/backends/narwhals/test_lazy_regressions.py -x -q` |
| **Full suite command** | `pytest tests/backends/narwhals/ -x -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/backends/narwhals/test_lazy_regressions.py -x -q`
- **After every plan wave:** Run `pytest tests/backends/narwhals/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 8-01-01 | 01 | 1 | MISSING-01 | unit | `pytest tests/backends/narwhals/test_lazy_regressions.py::test_lazy_failure_cases_per_row_polars -x -q` | ✅ | ✅ green |
| 8-01-02 | 01 | 1 | MISSING-02 | unit | `pytest tests/backends/narwhals/test_lazy_regressions.py::test_lazy_bool_output_check_does_not_crash -x -q` | ✅ | ✅ green |
| 8-02-01 | 02 | 2 | MISSING-01 | integration | `pytest tests/backends/narwhals/test_lazy_regressions.py::test_lazy_failure_cases_per_row_polars -x -q` | ✅ | ✅ green |
| 8-02-02 | 02 | 2 | MISSING-01 | integration | `pytest tests/backends/narwhals/test_lazy_regressions.py::test_lazy_failure_cases_per_row_ibis -x -q` | ✅ | ✅ green |
| 8-02-03 | 02 | 2 | MISSING-02 | integration | `pytest tests/backends/narwhals/test_lazy_regressions.py::test_lazy_bool_output_check_does_not_crash -x -q` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `tests/backends/narwhals/test_lazy_regressions.py` — stubs for MISSING-01, MISSING-02

*Existing pytest infrastructure (conftest.py, fixtures) covers all other needs.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 15s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** 2026-03-25

---

## Validation Audit 2026-03-25

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
| Tests verified green | 3 |

All 3 tests in `test_lazy_regressions.py` pass. Coverage is complete for MISSING-01 (polars + ibis) and MISSING-02 (bool scalar). VALIDATION.md was in draft state post-execution — finalized with correct test names and commands.
