---
phase: 5
slug: expression-based-check-protocol-eliminate-framework-specific-apply-branching
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-23
audited: 2026-03-24
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | pyproject.toml |
| **Quick run command** | `python -m pytest tests/backends/narwhals/test_checks.py -q` |
| **Full suite command** | `python -m pytest tests/backends/narwhals/ -q` |
| **Estimated runtime** | ~2 seconds (quick), ~5 seconds (full) |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/backends/narwhals/test_checks.py -q`
- **After every plan wave:** Run `python -m pytest tests/backends/narwhals/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | EXPR-01 | unit/RED→GREEN | `python -m pytest tests/backends/narwhals/test_checks.py::test_builtin_check_routing -q` | ✅ | ✅ green |
| 05-01-02 | 01 | 1 | EXPR-06 | unit/RED→GREEN | `python -m pytest tests/backends/narwhals/test_checks.py::test_native_false_user_check -q` | ✅ | ✅ green |
| 05-02-01 | 02 | 2 | EXPR-02, EXPR-03 | unit/GREEN | `python -m pytest tests/backends/narwhals/test_checks.py -k "builtin_checks_pass or builtin_checks_fail" -q` | ✅ | ✅ green |
| 05-03-01 | 03 | 3 | EXPR-01, EXPR-04, EXPR-05, EXPR-06, EXPR-07 | integration | `python -m pytest tests/backends/narwhals/test_checks.py -q` | ✅ | ✅ green |
| 05-03-02 | 03 | 3 | EXPR-07 | regression | `python -m pytest tests/backends/narwhals/ -q` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

### Requirement Coverage

| Requirement | Description | Covered By | Status |
|-------------|-------------|------------|--------|
| EXPR-01 | Builtin checks dispatch on nw.Expr (not frame+key) | `test_builtin_check_routing` | ✅ COVERED |
| EXPR-02 | All 14 builtins annotated `col_expr: nw.Expr` | `test_builtin_checks_pass[*]` (28 cases) | ✅ COVERED |
| EXPR-03 | Builtins return nw.Expr, no frame.select() | `test_builtin_checks_pass/fail[*]` (56 cases) | ✅ COVERED |
| EXPR-04 | No ibis row_number join / no backend branching in apply() | `test_builtin_checks_pass/fail[ibis-*]` (28 cases) | ✅ COVERED |
| EXPR-05 | No reassembly block — apply() returns wide table directly | `test_apply_returns_wide_table` | ✅ COVERED |
| EXPR-06 | native=False check receives nw.col(key) expression | `test_native_false_user_check` | ✅ COVERED |
| EXPR-07 | No regressions in narwhals backend suite | Full suite: 72 passed, 8 skipped, 0 failed | ✅ COVERED |

---

## Wave 0 Requirements

- Existing test infrastructure covers all phase requirements.
- Wave 1 established RED stubs in `tests/backends/narwhals/test_checks.py`; Plans 05-02 + 05-03 turned them GREEN.

---

## Manual-Only Verifications

All phase behaviors have automated verification.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 10s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** 2026-03-24

---

## Validation Audit 2026-03-24

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

All 7 requirements (EXPR-01 through EXPR-07) are COVERED by existing automated tests.
72 tests pass, 8 skipped (backend-specific skips), 0 failures.
VALIDATION.md updated from `pending` to `complete` / `nyquist_compliant: true`.
