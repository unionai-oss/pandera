---
phase: 3
slug: documentation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-18
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Sphinx/MyST (doc build) + manual review |
| **Config file** | `docs/source/conf.py` |
| **Quick run command** | `grep -n "PySpark" docs/source/supported_libraries.md docs/source/pyspark_sql.md` |
| **Full suite command** | `cd docs && make html 2>&1 | tail -20` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run quick grep to confirm PySpark text present
- **After every plan wave:** Run `cd docs && make html` to verify no build errors
- **Before `/gsd-verify-work`:** Full doc build must complete without warnings on modified files
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 3-01-01 | 01 | 1 | DOCS-01 | — | N/A | manual | `grep -n "PySpark" docs/source/supported_libraries.md` | ✅ | ⬜ pending |
| 3-01-02 | 01 | 1 | DOCS-01 | — | N/A | manual | `grep -n "PySpark\|pyspark" docs/source/pyspark_sql.md` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements. No new test stubs needed — this phase modifies documentation files only.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| PySpark narwhals note mirrors ibis.md structure | DOCS-01 | Visual review needed for note formatting and prose quality | Compare rendered `{note}` block in pyspark_sql.md against ibis.md |
| PySpark listed in narwhals backends section with correct limitations | DOCS-01 | Prose accuracy requires human review | Read `supported_libraries.md` narwhals-powered backends section and confirm PySpark is named with "no element-wise checks, no row sampling" |
| Doc build completes without errors | DOCS-01 | Build errors block doc publication | `cd docs && make html` exits 0 |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
