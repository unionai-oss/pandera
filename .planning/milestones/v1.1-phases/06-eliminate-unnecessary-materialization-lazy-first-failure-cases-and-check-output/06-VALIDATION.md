---
phase: 6
slug: eliminate-unnecessary-materialization-lazy-first-failure-cases-and-check-output
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-23
audited: 2026-03-24
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `pytest tests/backends/narwhals/ -x -q` |
| **Full suite command** | `pytest tests/backends/narwhals/ -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/backends/narwhals/ -x -q`
- **After every plan wave:** Run `pytest tests/backends/narwhals/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 6-01-01 | 01 | 1 | lazy-first run_check | unit | `pytest tests/backends/narwhals/test_checks.py -x -q` | ✅ | ✅ green |
| 6-01-02 | 01 | 1 | failure_cases narwhals wrapper | unit | `pytest tests/backends/narwhals/test_checks.py -x -q` | ✅ | ✅ green |
| 6-01-03 | 01 | 1 | subsample lazy head/tail | unit | `pytest tests/backends/narwhals/test_components.py -x -q` | ✅ | ✅ green |
| 6-01-04 | 01 | 1 | check_nullable scalar-only materialize | unit | `pytest tests/backends/narwhals/test_components.py -x -q` | ✅ | ✅ green |
| 6-02-01 | 02 | 2 | failure_cases_metadata ibis return type | unit | `pytest tests/backends/narwhals/ -k failure_cases_metadata -x -q` | ✅ | ✅ green |
| 6-02-02 | 02 | 2 | failure_cases_metadata polars-lazy return type | unit | `pytest tests/backends/narwhals/ -k failure_cases_metadata -x -q` | ✅ | ✅ green |
| 6-03-01 | 03 | 3 | SchemaError.failure_cases native polars | e2e | `pytest tests/backends/narwhals/test_e2e.py -x -q` | ✅ | ✅ green |
| 6-03-02 | 03 | 3 | SchemaError.failure_cases native ibis.Table | e2e | `pytest tests/backends/narwhals/test_e2e.py -x -q` | ✅ | ✅ green |
| 6-03-03 | 03 | 3 | SchemaErrors.failure_cases ibis lazy | e2e | `pytest tests/backends/narwhals/test_e2e.py -x -q` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `tests/backends/narwhals/test_components.py` — new tests for `subsample()` lazy head/tail and NotImplementedError for ibis tail= (`TestSubsample` added in Plan 01)
- [x] `tests/backends/narwhals/test_components.py` — new test for `failure_cases_metadata()` with ibis input asserting ibis.Table return type (`test_failure_cases_metadata_ibis_returns_ibis_table` added in Plan 01)

*All Wave 0 requirements fulfilled. All tests GREEN.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `_is_ibis_result` block fully deleted | lazy-first principle | Code review / grep | `grep -r "_is_ibis_result" pandera/` — only appears in comment; no live code ✅ |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** ✅ Nyquist-compliant — 205 passed, 8 skipped, 1 xfailed (2026-03-24)

---

## Validation Audit 2026-03-24

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
| Total tasks | 9 |
| Status | All COVERED ✅ |
