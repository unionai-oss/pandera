---
phase: 1
slug: structural-cleanup
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-30
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.4.2 |
| **Config file** | `pyproject.toml` (pytest section) |
| **Quick run command** | `python -m pytest tests/backends/narwhals/ -x -q` |
| **Full suite command** | `python -m pytest tests/backends/narwhals/ -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/backends/narwhals/ -x -q`
- **After every plan wave:** Run `python -m pytest tests/backends/narwhals/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | TYPES-01, TYPES-02, TYPES-03 | unit (source inspection) | `python -m pytest tests/backends/narwhals/test_phase01_arch.py -k "lazy" -x` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 1 | CLEAN-01, CLEAN-02, CLEAN-03 | unit (source inspection) | `python -m pytest tests/backends/narwhals/test_phase01_arch.py -k "clean" -x` | ❌ W0 | ⬜ pending |
| 1-01-03 | 01 | 1 | CLEAN-04 | unit (source inspection) | `python -m pytest tests/backends/narwhals/test_phase01_arch.py -k "import" -x` | ❌ W0 | ⬜ pending |
| 1-02-01 | 02 | 1 | EAGER-01 | unit | `python -m pytest tests/backends/narwhals/ -k "coerce" -x` | Partial | ⬜ pending |
| 1-02-02 | 02 | 1 | EAGER-02 | unit (source inspection) | `python -m pytest tests/backends/narwhals/test_phase01_arch.py -k "eager" -x` | ❌ W0 | ⬜ pending |
| 1-03-01 | 03 | 2 | CHECKS-01 | integration | `python -m pytest tests/backends/narwhals/test_e2e.py -k "custom" -x` | Partial | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/backends/narwhals/test_phase01_arch.py` — add test functions for TYPES-01/02/03, CLEAN-01/02/03/04, EAGER-02 (source-inspection style, matching existing tests in file)
- [ ] `tests/backends/narwhals/test_e2e.py` — add `TestCustomChecksPolarsRowLevel` and `TestCustomChecksIbisRowLevel` classes covering `native=True` checks returning `pl.Series`, `pl.DataFrame`, and ibis `BooleanColumn`

*Existing infrastructure (pytest + narwhals test suite, 221 green baseline) covers the majority of requirements; these are additive tests.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| No Polars imports at runtime when only ibis installed | CLEAN-01, CLEAN-02, CLEAN-03 | Requires environment without polars | Install pandera without polars extras; run ibis-only validation |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
