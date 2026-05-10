# Phase 1: Structural Cleanup - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md — this log preserves the discussion.

**Date:** 2026-03-30
**Phase:** 01-structural-cleanup
**Mode:** discuss
**Areas analyzed:** Lazy/SQL Detection, Backend Isolation, Eager Execution, Custom Checks

## Gray Areas Presented

### Lazy/SQL Detection Unification
| Gray Area | Decision | Notes |
|-----------|----------|-------|
| Single utility vs split helpers | Single `_is_lazy` utility replacing `_is_lazy_or_sql` + inline hasattr checks | Existing utility barely used; consolidate |
| Dispatch rewrite scope | Keep three-branch structure, fix detection conditions only | Restructuring to registry would expand scope |

### Backend Isolation
| Gray Area | Decision | Notes |
|-----------|----------|-------|
| CLEAN-02: annotation import fix | `TYPE_CHECKING` guard | User confirmed — minimal diff, no behavioral change |
| CLEAN-03: polars in failure_cases_metadata | Rewrite branches to narwhals ops | Confident — polars must remain optional |
| CLEAN-05 (folded todo) scope | Schema API abstraction fix (not just Polars) | User clarified: broader abstraction issue, not just Polars coupling |

### Eager Execution
| Gray Area | Decision | Notes |
|-----------|----------|-------|
| try_coerce probe strategy | `.head(1).collect()` | User: "seems reasonable, we'll see in tests" |

### Custom Checks
| Gray Area | Decision | Notes |
|-----------|----------|-------|
| CHECKS-01 scope | Investigate + fix + regression test | Matches success criterion exactly |

## Corrections Made

### Todo framing
- **Original framing:** "synthetic column construction as a Polars coupling issue"
- **User correction:** broader abstraction issue — affects both Polars and Ibis; the backend shouldn't reach into any framework-specific schema API
- **Result:** Captured as CLEAN-05 / D-06 with corrected framing

## Folded Todos

- **Push synthetic column construction into schema API layer** (score 0.6): user confirmed fold into Phase 1 after framing clarification
