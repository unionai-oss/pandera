---
phase: 3
slug: fix-ibischeckbackend-delegation-via-apply-type-dispatch
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-22
audited: 2026-03-24
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | pyproject.toml |
| **Quick run command** | `pytest tests/core/test_checks.py -x -q` |
| **Full suite command** | `pytest tests/core/test_checks.py tests/backends/ -x -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/core/test_checks.py -x -q`
- **After every plan wave:** Run `pytest tests/core/test_checks.py tests/backends/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 3-01-01 | 01 | 1 | builtin-sig: Check.native flag + from_builtin_check_name native=False | unit | `pytest tests/core/test_checks.py::TestNativeFlag tests/core/test_checks.py::TestBuiltinNativeFalse -x -q` | ✅ | ✅ green |
| 3-01-02 | 01 | 1 | builtin-sig: all 14 builtins callable with (col_expr: nw.Expr, ...) | unit | `pytest tests/core/test_checks.py::TestBuiltinCheckSignatures -x -q` | ✅ | ✅ green |
| 3-02-01 | 02 | 1 | dispatch-remove: apply() dispatches on native flag | unit | `pytest tests/backends/narwhals/test_checks.py -k "builtin_check_routing or builtin_checks" -x -q` | ✅ | ✅ green |
| 3-02-02 | 02 | 2 | ibis-normalize: ibis receives (ibis.Table, key) | unit | `pytest tests/backends/narwhals/test_checks.py -k "native_true_user_check_ibis" -x -q` | ✅ | ✅ green (ibis) / ⏭ skip (polars) |
| 3-02-03 | 02 | 2 | ibis-normalize: ir.BooleanScalar normalizes to bool | integration | `pytest tests/backends/narwhals/test_checks.py -k "ibis_boolean_scalar" -x -q` | ✅ | ✅ green (ibis) / ⏭ skip (polars) |

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

**Approval:** 2026-03-24

## Validation Audit 2026-03-24
| Metric | Count |
|--------|-------|
| Gaps found | 1 |
| Resolved | 1 |
| Escalated | 0 |

**Gap resolved:** `TestBuiltinCheckSignatures` (14 tests) updated from Phase 3 `(frame, key, ...)` calling convention to Phase 5 `(col_expr: nw.Expr, ...)` protocol. All 33 tests in `tests/core/test_checks.py` now pass. Committed as `cd9f310`.
