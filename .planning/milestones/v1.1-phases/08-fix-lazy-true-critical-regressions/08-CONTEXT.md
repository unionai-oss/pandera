# Phase 8: Fix lazy=True critical regressions - Context

**Gathered:** 2026-03-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix two critical regressions introduced by prior phases that break lazy=True validation:
1. `failure_cases_metadata()` collapsing N polars lazy failure rows to a single repr string (MISSING-01)
2. `_count_failure_cases()` crashing with TypeError when `failure_cases` is a bool scalar (MISSING-02)

No new functionality. Both fixes must be narwhals-idiomatic with no native type-dependent isinstance checks.

</domain>

<decisions>
## Implementation Decisions

### MISSING-01: failure_cases_metadata() rewrap approach
- Replace the ibis-specific `try/import ibis/isinstance(ibis.Table)` guard with a unified `try: fc = nw.from_native(fc, eager_or_interchange_only=False) / except TypeError: pass` block
- This handles pl.DataFrame, ibis.Table, and pl.LazyFrame in one shot — no native isinstance needed
- Keep existing post-rewrap branching structure exactly as-is (minimal diff): the two `isinstance(fc, (nw.LazyFrame, nw.DataFrame))` checks below are unchanged; only the rewrap block changes

### MISSING-02: _count_failure_cases() scalar fallback
- Wrap the entire `nw.from_native(...)` call in `try/except TypeError`
- Fallback: `return 0 if failure_cases is None else 1`
- Remove the existing `isinstance(failure_cases, str)` guard at line 13 — it becomes dead code since `nw.from_native(str)` also raises TypeError and falls to the except branch returning 1

### Test structure
- New dedicated file: `tests/backends/narwhals/test_lazy_regressions.py`
- Prefer a single parametrized test over polars and ibis if the setup is clean; fall back to separate test functions per backend if parametrization gets complex
- Assertions must check both row count (len == N) AND that failure_case column contains individual values, not a repr string

### Claude's Discretion
- Exact pytest parametrize fixture setup (polars LazyFrame vs ibis Table inputs)
- Whether the bool-output test goes in the same file or a separate function

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `nw.from_native(..., eager_or_interchange_only=False)`: already used throughout; `eager_only=False` is equivalent for this use case
- `_is_lazy_or_sql(fc)`: existing helper in base.py that distinguishes polars-lazy from eager
- `_materialize(fc)`: used in the eager polars branch — unchanged

### Established Patterns
- Phase 6 contract: `failure_cases` is always native (pl.DataFrame or ibis.Table) at SchemaError boundary — both fixes must accept native types as input
- Narwhals-is-internal: `nw.from_native()` wraps for internal ops; user-facing outputs stay native
- `try/except TypeError` for narwhals wrapping non-frame scalars: consistent with MISSING-02 fix pattern

### Integration Points
- `pandera/backends/narwhals/base.py` lines 180–187: replace ibis-specific rewrap block with unified `try: nw.from_native / except TypeError: pass`
- `pandera/api/narwhals/error_handler.py` lines 12–24: replace with try/except TypeError wrapping the nw.from_native count, remove str guard
- `tests/backends/narwhals/test_lazy_regressions.py`: new file

</code_context>

<specifics>
## Specific Ideas

- The minimal-diff preference for MISSING-01: only the rewrap block changes, the branching logic below is untouched
- The str guard removal in MISSING-02 is intentional — not an accidental omission

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 08-fix-lazy-true-critical-regressions*
*Context gathered: 2026-03-24*
