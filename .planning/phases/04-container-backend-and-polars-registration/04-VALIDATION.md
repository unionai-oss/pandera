---
phase: 4
slug: container-backend-and-polars-registration
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-13
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing) |
| **Config file** | `pyproject.toml` (existing pytest configuration) |
| **Quick run command** | `python -m pytest tests/backends/narwhals/test_container.py -x -q` |
| **Full suite command** | `python -m pytest tests/backends/narwhals/ -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/backends/narwhals/ -x -q`
- **After every plan wave:** Run `python -m pytest tests/backends/narwhals/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 4-01-01 | 01 | 0 | CONTAINER-01, CONTAINER-02, CONTAINER-03, CONTAINER-04, REGISTER-01, REGISTER-02, REGISTER-04, TEST-03 | stub | `python -m pytest tests/backends/narwhals/test_container.py -x -q` | ❌ W0 | ⬜ pending |
| 4-02-01 | 02 | 1 | CONTAINER-01 | unit | `python -m pytest tests/backends/narwhals/test_container.py::test_failure_cases_metadata -x` | ❌ W0 | ⬜ pending |
| 4-02-02 | 02 | 1 | CONTAINER-02 | integration | `python -m pytest tests/backends/narwhals/test_container.py::test_validate_polars_dataframe tests/backends/narwhals/test_container.py::test_validate_polars_lazyframe -x` | ❌ W0 | ⬜ pending |
| 4-02-03 | 02 | 1 | CONTAINER-03 | unit | `python -m pytest tests/backends/narwhals/test_container.py::test_strict_true_rejects_extra_columns tests/backends/narwhals/test_container.py::test_strict_filter_drops_extra_columns -x` | ❌ W0 | ⬜ pending |
| 4-02-04 | 02 | 1 | CONTAINER-04 | integration | `python -m pytest tests/backends/narwhals/test_container.py::test_lazy_mode_collects_all_errors -x` | ❌ W0 | ⬜ pending |
| 4-02-05 | 02 | 1 | TEST-03 | unit | `python -m pytest tests/backends/narwhals/test_container.py::test_failure_cases_is_native -x` | ❌ W0 | ⬜ pending |
| 4-03-01 | 03 | 2 | REGISTER-01, REGISTER-02, REGISTER-04 | unit | `python -m pytest tests/backends/narwhals/test_container.py::test_register_is_idempotent tests/backends/narwhals/test_container.py::test_polars_backends_registered tests/backends/narwhals/test_container.py::test_narwhals_not_registered_by_default -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/backends/narwhals/test_container.py` — stubs for CONTAINER-01, CONTAINER-02, CONTAINER-03, CONTAINER-04, REGISTER-01, REGISTER-02, REGISTER-04, TEST-03

*Existing pytest infrastructure covers all phase requirements — no new framework installation needed.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
