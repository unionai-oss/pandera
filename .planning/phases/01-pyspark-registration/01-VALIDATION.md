---
phase: 1
slug: pyspark-registration
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-10
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | pyproject.toml |
| **Quick run command** | `python -m pytest tests/narwhals/test_container.py -x -q` |
| **Full suite command** | `PANDERA_USE_NARWHALS_BACKEND=True python -m pytest tests/narwhals/ -x -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/narwhals/test_container.py -x -q`
- **After every plan wave:** Run `PANDERA_USE_NARWHALS_BACKEND=True python -m pytest tests/narwhals/ -x -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | REG-01 | — | N/A | unit | `python -m pytest tests/narwhals/test_container.py -k pyspark -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/narwhals/test_container.py` — add test stubs for PySpark narwhals registration (mirroring `test_polars_narwhals_activated_when_opted_in` and `test_ibis_narwhals_activated_when_opted_in`)

*Existing infrastructure (pytest, conftest, nox) covers all other phase requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Flag-off leaves native PySpark backends intact | REG-01 | Requires PySpark runtime (Java/Spark); not available in standard CI | Run `PANDERA_USE_NARWHALS_BACKEND=False python -c "import pandera; from pandera.backends.pyspark.register import register_pyspark_backends; register_pyspark_backends(); ..."` and confirm native backends registered |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
