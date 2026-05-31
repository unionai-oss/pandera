---
phase: 4
slug: eliminate-backend-specific-dispatch-branches
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-25
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` (pytest section) |
| **Quick run command** | `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/ tests/backends/narwhals/ -x -q` |
| **Full suite command** | `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/ tests/backends/narwhals/ tests/polars/ tests/ibis/ -q` |
| **Estimated runtime** | ~120 seconds (pyspark session startup) |

---

## Sampling Rate

- **After every task commit:** Run quick run command
- **After every plan wave:** Run full suite command
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 04-01-T2 | 01 | 1 | ARCH-01 | unit | `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/narwhals/test_container.py tests/pyspark/test_pyspark_error.py -x -q` | ⬜ pending |
| 04-02-T1 | 02 | 2 | ARCH-02 | unit | `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/test_pyspark_error.py -x -q` | ⬜ pending |
| 04-03-T1 | 03 | 1 | ARCH-03 | unit | `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/test_pyspark_dtypes.py -x -q` | ⬜ pending |
| 04-04-T1 | 04 | 1 | ARCH-04 | unit | `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/test_pyspark_error.py tests/pyspark/test_pyspark_check.py -x -q` | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `_materialize()` no longer calls `.execute()` on PySpark frame | ARCH-01 | Runtime behavior depends on PySpark session | `grep -n "\.execute()" pandera/api/narwhals/utils.py` — must return no matches |
| No `module.startswith("pyspark")` string in narwhals source | ARCH-02 | Static grep verification | `grep -rn 'startswith.*pyspark' pandera/backends/narwhals/` — must return no matches |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
