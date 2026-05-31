---
phase: 5
slug: correctness-and-behavioral-parity
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-25
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pixi.toml` (task: `pyspark-test`) |
| **Quick run command** | `pixi run pyspark-test tests/pyspark/test_pyspark_model.py tests/pyspark/test_pyspark_accessor.py tests/pyspark/test_pyspark_config.py` |
| **Full suite command** | `pixi run pyspark-test` |
| **Estimated runtime** | ~120 seconds |

---

## Sampling Rate

- **After every task commit:** Run quick run command targeting affected test file
- **After every plan wave:** Run `pixi run pyspark-test`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 5-01-01 | 01 | 1 | CORR-01 | — | N/A | integration | `pixi run pyspark-test tests/pyspark/test_pyspark_model.py -k test_strict_filter` | ✅ | ⬜ pending |
| 5-01-02 | 01 | 1 | CORR-02 | — | N/A | integration | `pixi run pyspark-test tests/pyspark/test_pyspark_accessor.py -k test_schema` | ✅ | ⬜ pending |
| 5-01-03 | 01 | 1 | CORR-01, CORR-02 | — | N/A | unit | `pixi run pyspark-test tests/pyspark/test_pyspark_model.py` | ✅ | ⬜ pending |
| 5-02-01 | 02 | 1 | TEST-FIX-01 | — | N/A | unit | `pixi run pyspark-test tests/pyspark/test_pyspark_config.py` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements. No new test stubs needed — all three fixes convert existing xfail tests to passing tests.

---

## Manual-Only Verifications

All phase behaviors have automated verification.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
