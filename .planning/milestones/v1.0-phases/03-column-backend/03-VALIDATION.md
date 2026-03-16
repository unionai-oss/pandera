---
phase: 3
slug: column-backend
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-13
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing) |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `pytest tests/backends/narwhals/test_components.py -x` |
| **Full suite command** | `pytest tests/backends/narwhals/ -x` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/backends/narwhals/test_components.py -x`
- **After every plan wave:** Run `pytest tests/backends/narwhals/ -x`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 3-01-01 | 01 | 0 | COLUMN-01 | unit | `pytest tests/backends/narwhals/test_components.py -x` | ❌ W0 | ⬜ pending |
| 3-01-02 | 01 | 1 | COLUMN-01 | unit | `pytest tests/backends/narwhals/test_components.py::test_check_nullable_fails -x` | ❌ W0 | ⬜ pending |
| 3-01-03 | 01 | 1 | COLUMN-01 | unit | `pytest tests/backends/narwhals/test_components.py::test_check_nullable_passes -x` | ❌ W0 | ⬜ pending |
| 3-01-04 | 01 | 1 | COLUMN-01 | unit | `pytest tests/backends/narwhals/test_components.py::test_check_nullable_catches_nan -x` | ❌ W0 | ⬜ pending |
| 3-01-05 | 01 | 1 | COLUMN-01 | unit | `pytest tests/backends/narwhals/test_components.py::test_check_unique_fails -x` | ❌ W0 | ⬜ pending |
| 3-01-06 | 01 | 1 | COLUMN-01 | unit | `pytest tests/backends/narwhals/test_components.py::test_check_unique_passes -x` | ❌ W0 | ⬜ pending |
| 3-01-07 | 01 | 1 | COLUMN-01 | unit | `pytest tests/backends/narwhals/test_components.py::test_check_dtype_wrong -x` | ❌ W0 | ⬜ pending |
| 3-01-08 | 01 | 1 | COLUMN-01 | unit | `pytest tests/backends/narwhals/test_components.py::test_check_dtype_correct -x` | ❌ W0 | ⬜ pending |
| 3-01-09 | 01 | 1 | COLUMN-01 | unit | `pytest tests/backends/narwhals/test_components.py::test_run_checks -x` | ❌ W0 | ⬜ pending |
| 3-02-01 | 01 | 1 | COLUMN-02 | unit | `pytest tests/backends/narwhals/test_components.py -x -k ibis` | ❌ W0 | ⬜ pending |
| 3-02-02 | 01 | 1 | COLUMN-01+02 | unit | `pytest tests/backends/narwhals/test_components.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/backends/narwhals/test_components.py` — stubs for COLUMN-01, COLUMN-02 (parameterized with polars and ibis fixtures)
- No new framework or fixture infrastructure needed — `make_narwhals_frame` fixture in existing `conftest.py` handles both polars and ibis frame creation

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| None | — | — | — |

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
