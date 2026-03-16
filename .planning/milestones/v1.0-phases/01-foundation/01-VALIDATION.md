---
phase: 1
slug: foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-09
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (project-installed version) |
| **Config file** | `pyproject.toml` — `[tool.pytest.ini_options]` |
| **Quick run command** | `python -m pytest tests/backends/narwhals/test_narwhals_dtypes.py -x -q` |
| **Full suite command** | `python -m pytest tests/backends/narwhals/ -q` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/backends/narwhals/test_narwhals_dtypes.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/backends/narwhals/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 0 | INFRA-01 | smoke | `python -c "import narwhals.stable.v1 as nw"` | ✅ runtime | ⬜ pending |
| 1-01-02 | 01 | 0 | INFRA-02 | unit | `python -m pytest tests/backends/narwhals/test_narwhals_dtypes.py::test_narwhals_data_type -x` | ❌ W0 | ⬜ pending |
| 1-01-03 | 01 | 0 | INFRA-03 | unit | `python -m pytest tests/backends/narwhals/test_narwhals_dtypes.py::test_to_native -x` | ❌ W0 | ⬜ pending |
| 1-02-01 | 02 | 1 | ENGINE-01 | unit | `python -m pytest tests/backends/narwhals/test_narwhals_dtypes.py::test_engine_dtype -x` | ❌ W0 | ⬜ pending |
| 1-02-02 | 02 | 1 | ENGINE-02 | unit | `python -m pytest tests/backends/narwhals/test_narwhals_dtypes.py::test_dtype_registration -x` | ❌ W0 | ⬜ pending |
| 1-02-03 | 02 | 1 | ENGINE-03 | unit | `python -m pytest tests/backends/narwhals/test_narwhals_dtypes.py::test_coerce -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/backends/__init__.py` — package init (if not exists)
- [ ] `tests/backends/narwhals/__init__.py` — package init
- [ ] `tests/backends/narwhals/test_narwhals_dtypes.py` — stubs for INFRA-02, INFRA-03, ENGINE-01, ENGINE-02, ENGINE-03
- [ ] `pandera/api/narwhals/__init__.py` — package init for new API package

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `narwhals>=2.15.0` installable via `pandera[narwhals]` extra | INFRA-01 | pyproject.toml extras only verifiable via install test | `pip install -e ".[narwhals]"` in clean env; verify `import narwhals.stable.v1 as nw` succeeds |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
