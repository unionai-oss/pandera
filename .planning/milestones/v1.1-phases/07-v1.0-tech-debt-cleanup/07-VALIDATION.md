---
phase: 7
slug: v1-0-tech-debt-cleanup
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-24
audited: 2026-03-24
---

# Phase 7 — Validation Strategy

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
| 07-01-01 | 01 | 1 | code-correctness | integration | `pytest tests/backends/narwhals/ -x -q -k "failure_cases"` | ✅ | ✅ green |
| 07-01-02 | 01 | 1 | ibis-assertion | integration | `pytest tests/backends/narwhals/test_e2e.py -x -k "test_custom_check_receives_table_and_key"` | ✅ | ✅ green |
| 07-02-01 | 02 | 2 | docstring | manual | Visual inspection of `pandera/api/checks.py` lines 86-90 | ✅ | ✅ green |
| 07-02-02 | 02 | 2 | xfail-promotion | unit | `pytest tests/backends/narwhals/test_container.py::test_failure_cases_metadata tests/backends/narwhals/test_container.py::test_ibis_narwhals_auto_activated tests/backends/narwhals/test_container.py::test_ibis_backend_is_narwhals tests/backends/narwhals/test_checks.py::test_postprocess_lazyframe_no_materialization_ibis -x` | ✅ | ✅ green |
| 07-02-03 | 02 | 2 | delete-hollow-test | unit | `pytest tests/backends/narwhals/test_container.py -x -q` | ✅ | ✅ green |
| 07-02-04 | 02 | 2 | roadmap-checkboxes | manual | Visual inspection of `.planning/ROADMAP.md` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `Check.native` docstring accuracy | docs-hygiene | Documentation — no automated test | `checks.py:86-91` — correctly describes `nw.col(key)` (a `nw.Expr`) as sole argument ✅ |
| ROADMAP checkboxes updated | docs-hygiene | Markdown file — no automated test | 23 `[x]`, 0 `[ ]` — all phases complete ✅ |

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
| Total tasks | 6 |
| Status | All COVERED ✅ |
