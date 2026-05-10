---
phase: 5
slug: ibis-registration-and-integration
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-14
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing throughout project) |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `pytest tests/backends/narwhals/ -x -q` |
| **Full suite command** | `pytest tests/backends/narwhals/ tests/ibis/ -q` |
| **Estimated runtime** | ~60 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/backends/narwhals/ -x -q`
- **After every plan wave:** Run `pytest tests/backends/narwhals/ tests/ibis/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 5-W0-01 | W0 | 0 | TEST-04 | integration | `pytest tests/backends/narwhals/test_parity.py -x -q` | ❌ W0 | ⬜ pending |
| 5-W0-02 | W0 | 0 | REGISTER-03 | unit | `pytest tests/backends/narwhals/test_container.py -x -k test_ibis_narwhals_auto_activated -q` | ❌ W0 | ⬜ pending |
| 5-01-01 | 01 | 1 | REGISTER-03 | integration | `pytest tests/backends/narwhals/ -x -k ibis -q` | ✅ | ⬜ pending |
| 5-01-02 | 01 | 1 | REGISTER-03 | unit | `pytest tests/backends/narwhals/test_container.py -x -k test_ibis_narwhals_auto_activated -q` | ❌ W0 | ⬜ pending |
| 5-02-01 | 02 | 1 | TEST-02 | unit | `pytest tests/backends/narwhals/test_checks.py -x -q` | ✅ | ⬜ pending |
| 5-02-02 | 02 | 1 | TEST-02 | unit | `pytest tests/backends/narwhals/test_components.py -x -q` | ✅ | ⬜ pending |
| 5-03-01 | 03 | 2 | TEST-02 | integration | `pytest tests/backends/narwhals/test_container.py -x -k "lazy or native" -q` | ✅ | ⬜ pending |
| 5-04-01 | 04 | 3 | TEST-04 | integration | `pytest tests/backends/narwhals/test_parity.py -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/backends/narwhals/test_parity.py` — new parity test file covering TEST-04 (validation depth, strict/filter, lazy, decorators across Polars and Ibis)
- [ ] `tests/backends/narwhals/test_container.py` — add `test_ibis_narwhals_auto_activated` stub covering REGISTER-03 registration verification

*All other infrastructure — pytest, fixtures, conftest — already exists; only new test file and targeted additions needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `element_wise=True` raises `NotImplementedError` with clear message on ibis | TEST-02 | Error message quality check | Run `schema.validate(ibis_table)` with a `Check` using `element_wise=True`; verify the exception message is clear and actionable |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
