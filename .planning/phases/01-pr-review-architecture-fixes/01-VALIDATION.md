---
phase: 1
slug: pr-review-architecture-fixes
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-21
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | setup.cfg / pyproject.toml |
| **Quick run command** | `python -m pytest tests/core/test_pandas_engine.py tests/core/test_polars_engine.py -x -q 2>/dev/null || python -m pytest tests/ -k "narwhals" -x -q` |
| **Full suite command** | `python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~60 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -k "narwhals or error_handler" -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | ARCH-01 | unit | `python -c "from pandera.backends.base.error_handler import SchemaErrorHandler; import inspect; src=inspect.getsource(SchemaErrorHandler); assert 'ibis' not in src"` | ✅ | ⬜ pending |
| 1-01-02 | 01 | 1 | ARCH-01 | unit | `python -c "from pandera.api.narwhals.error_handler import NarwhalsErrorHandler; print('OK')"` | ✅ | ⬜ pending |
| 1-02-01 | 02 | 2 | ARCH-02, ARCH-03 | unit | `python -c "from pandera.backends.narwhals import container, components, base; import inspect; assert 'NarwhalsErrorHandler' in inspect.getsource(container)"` | ✅ | ⬜ pending |
| 1-02-02 | 02 | 2 | ARCH-04 | unit | `python -m pytest tests/ -k "narwhals" -x -q` | ✅ | ⬜ pending |
| 1-03-01 | 03 | 3 | ARCH-02 | unit | `python -c "from pandera.backends.narwhals.container import NarwhalsSchemaBackend; import inspect; src=inspect.getsource(NarwhalsSchemaBackend.validate); lines=src.split('\n'); idx=[i for i,l in enumerate(lines) if 'subsample' in l][0]; assert '_to_frame_kind_nw' not in '\n'.join(lines[:idx])"` | ✅ | ⬜ pending |
| 1-03-02 | 03 | 3 | ARCH-04 | unit | `python -m pytest tests/ -k "narwhals" -x -q` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

*Existing infrastructure covers all phase requirements.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
