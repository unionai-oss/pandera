---
phase: 9
slug: accumulate-check-outputs-into-single-wide-table-for-narwhals-idiomatic-drop-invalid-rows
status: final
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-25
finalized: 2026-03-25
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `python -m pytest tests/backends/narwhals/ tests/polars/test_polars_container.py tests/ibis/test_ibis_container.py -x -q` |
| **Full suite command** | `python -m pytest tests/backends/narwhals/ tests/polars/test_polars_container.py tests/ibis/test_ibis_container.py -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run quick run command
- **After every plan wave:** Run full suite command
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 9-01-01 | 01 | 1 | RED baseline | unit | `python -m pytest tests/polars/test_polars_container.py tests/ibis/test_ibis_container.py -k "drop_invalid" -q` | ✅ | ✅ green |
| 9-02-01 | 02 | 2 | apply() Expr | unit | `python -m pytest tests/backends/narwhals/ -x -q` | ✅ | ✅ green |
| 9-02-02 | 02 | 2 | drop_invalid_rows | integration | `python -m pytest tests/polars/test_polars_container.py tests/ibis/test_ibis_container.py -k "drop_invalid" -q` | ✅ | ✅ green |
| 9-02-03 | 02 | 2 | no regressions | suite | `python -m pytest tests/backends/narwhals/ tests/polars/test_polars_container.py tests/ibis/test_ibis_container.py -q` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

---

## Manual-Only Verifications

All phase behaviors have automated verification.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** 2026-03-25

---

## Validation Audit 2026-03-25

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
| Tests verified green | 26 drop_invalid + 221 narwhals suite |

All drop_invalid_rows tests pass (26: polars + ibis). Narwhals backend suite: 221 passed, 8 skipped, 1 xfailed. Requirements DIR-01–DIR-07 satisfied. VALIDATION.md was in draft state post-execution — finalized with correct statuses.

Note: The `-k "drop_invalid"` filter on task map commands was corrected — the original commands had a shell syntax issue (`-k` applied only to the first file arg). Commands updated to pass both test files before the `-k` flag.
