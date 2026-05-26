---
phase: 8
slug: test-quality-improvements
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-25
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `pytest tests/pyspark/test_pyspark_error.py tests/narwhals/ -x -q` |
| **Full suite command** | `pytest tests/pyspark/ tests/narwhals/ -q` |
| **Estimated runtime** | ~60–120 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/pyspark/test_pyspark_error.py tests/narwhals/ -x -q`
- **After every plan wave:** Run `pytest tests/pyspark/ tests/narwhals/ -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 8-01-01 | 01 | 1 | TQ-01 | — | N/A | unit | `pytest tests/pyspark/test_pyspark_error.py -x -q` | ✅ | ⬜ pending |
| 8-01-02 | 01 | 1 | TQ-02 | — | N/A | unit | `pytest tests/narwhals/ -x -q -k concat` | ✅ | ⬜ pending |
| 8-01-03 | 01 | 1 | TQ-03 | — | N/A | behavioral | `pytest tests/narwhals/test_arch03_schema_driven_dispatch.py -x -q` | ✅ | ⬜ pending |
| 8-01-04 | 01 | 1 | TQ-04 | — | N/A | unit | `pytest tests/narwhals/ -x -q -k register` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/pyspark/conftest.py` — add `_cmp_errors` helper for TQ-01
- [ ] `tests/narwhals/conftest.py` — add session-scoped `spark` fixture for TQ-03 PySpark behavioral tests (if not already present)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| PySpark registration comment accuracy | TQ-04 | Comment correctness requires reading code intent | Verify `register_pyspark_backends()` comment matches actual behavior |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
