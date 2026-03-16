---
phase: 2
slug: check-backend
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-09
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (project-standard) |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `pytest tests/backends/narwhals/test_checks.py -x -q` |
| **Full suite command** | `pytest tests/backends/narwhals/ -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/backends/narwhals/ -x -q`
- **After every plan wave:** Run `pytest tests/backends/narwhals/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 2-01-01 | 01 | 0 | TEST-01 | unit | `pytest tests/backends/narwhals/test_checks.py -x -q` | ❌ W0 | ⬜ pending |
| 2-01-02 | 01 | 0 | CHECKS-01 | unit | `pytest tests/backends/narwhals/test_checks.py::test_builtin_check_routing -x` | ❌ W0 | ⬜ pending |
| 2-01-03 | 01 | 0 | CHECKS-01 | unit | `pytest tests/backends/narwhals/test_checks.py::test_user_defined_check_routing -x` | ❌ W0 | ⬜ pending |
| 2-02-01 | 02 | 1 | CHECKS-02 | unit | `pytest tests/backends/narwhals/test_checks.py::test_builtin_checks -x` | ❌ W0 | ⬜ pending |
| 2-02-02 | 02 | 1 | CHECKS-02 | unit | `pytest tests/backends/narwhals/test_checks.py::test_builtin_checks_fail -x` | ❌ W0 | ⬜ pending |
| 2-03-01 | 03 | 1 | CHECKS-03 | unit | `pytest tests/backends/narwhals/test_checks.py::test_element_wise_sql_lazy_raises -x` | ❌ W0 | ⬜ pending |
| 2-03-02 | 03 | 1 | TEST-01 | integration | `pytest tests/backends/narwhals/test_checks.py -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/backends/narwhals/conftest.py` — `make_narwhals_frame` fixture + backend registration
- [ ] `tests/backends/narwhals/test_checks.py` — stubs for CHECKS-01, CHECKS-02, CHECKS-03, TEST-01
- [ ] `pandera/backends/narwhals/__init__.py` — empty module init
- [ ] `pandera/backends/narwhals/checks.py` — `NarwhalsCheckBackend` stub
- [ ] `pandera/backends/narwhals/builtin_checks.py` — 14 builtin check registration stubs

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
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
