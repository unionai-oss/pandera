---
phase: 2
slug: remaining-pr-review-fixes
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-22
audited: 2026-03-24
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | pyproject.toml |
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
| 2-01-01 | 01 | 1 | postprocess_lazyframe_output refactor | unit | `python -m pytest tests/backends/narwhals/test_checks.py -k "builtin_checks_pass" -x -q` | ✅ | ✅ green |
| 2-01-02 | 01 | 1 | postprocess_lazyframe_output failure cases | unit | `python -m pytest tests/backends/narwhals/test_checks.py -k "builtin_checks_fail" -x -q` | ✅ | ✅ green |
| 2-01-03 | 01 | 1 | postprocess_bool_output backend-agnostic | unit | `python -m pytest tests/backends/narwhals/test_checks.py -x -q` | ✅ | ✅ green |
| 2-02-01 | 02 | 1 | check_nullable inline with_columns | unit | `python -m pytest tests/backends/narwhals/test_components.py -k "nullable" -x -q` | ✅ | ✅ green |
| 2-02-02 | 02 | 1 | check_dtype single-pass | unit | `python -m pytest tests/backends/narwhals/test_components.py -k "dtype" -x -q` | ✅ | ✅ green |
| 2-02-03 | 02 | 1 | ibis delegation comment (no regression) | integration | `python -m pytest tests/backends/narwhals/ -q` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements. No new test files needed.

*Baseline: 125 passed, 1 skipped, 3 xfailed, 4 xpassed (as of 2026-03-22)*
*Audited: 202 passed, 8 skipped, 4 xfailed (as of 2026-03-24 — xpassed tests promoted to passing)*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** 2026-03-24

## Validation Audit 2026-03-24
| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
