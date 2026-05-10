# Phase 1: PR Review Architecture Fixes - Context

**Gathered:** 2026-03-21
**Status:** Ready for planning
**Source:** PR Review #2223 by @cosmicBboy

<domain>
## Phase Boundary

This phase addresses architectural feedback from the PR review of the Narwhals backend
implementation (PR #2223). The reviewer identified separation-of-concerns violations,
backend-specific logic leaking into wrong layers, and capitalization nits.

This phase does NOT add new features — it is purely a refactoring pass to align the
implementation with the reviewer's architectural expectations before the PR is merged.

</domain>

<decisions>
## Implementation Decisions

### Locked: ErrorHandler Architecture
- Remove ibis-specific logic (`isinstance(failure_cases, ibis.Table)` branch) from
  `pandera/api/base/error_handler.py:_count_failure_cases`. The base class must not
  know about ibis.
- Create `pandera/api/narwhals/error_handler.py` with a `NarwhalsErrorHandler` subclass
  that overrides `_count_failure_cases` to handle ibis.Table failure cases. Pattern
  mirrors the existing `pandera/api/ibis/error_handler.py`.
- All narwhals backends (`container.py`, `components.py`, `base.py`) must use
  `NarwhalsErrorHandler` instead of base `ErrorHandler`.

### Locked: No Polars Imports in Narwhals Container Backend
- `pandera/backends/narwhals/container.py` must not use `issubclass(return_type, pl.DataFrame)`
  in `_to_frame_kind_nw`. Replace with a backend-agnostic check (e.g., `hasattr(native, "collect")`
  combined with a name check to distinguish lazy from eager).
- `collect_schema_components` must not hardcode `from pandera.api.polars.components import Column`.
  Determine the correct Column class dynamically based on the schema type (e.g., inspect
  `schema.__class__.__module__` to detect ibis vs polars schemas).

### Locked: Accurate Comments in run_schema_component_checks
- The comment "Convert to native pl.LazyFrame for column component dispatch" in
  `run_schema_component_checks` is wrong — it also handles ibis.Table. Update the
  comment to accurately describe what `_to_native()` does for all backends.

### Locked: Capitalization
- All docstrings and comments referring to the framework must use "Narwhals" (capital N),
  not "narwhals". Apply consistently across all files in `pandera/backends/narwhals/`.

### Claude's Discretion
- Exact wording of updated comments
- Whether to remove `import polars as pl` from container.py entirely after fixing
  `_to_frame_kind_nw` (remove only if no other uses remain)
- The ibis-specific branching in `run_check` (base.py lines 84-131) may also be
  cleaned up if it can be done without breaking ibis tests, but this is secondary to
  the above locked items

</decisions>

<specifics>
## Specific References

- `pandera/api/base/error_handler.py` — remove ibis block from `_count_failure_cases` (lines 79-86)
- `pandera/api/narwhals/error_handler.py` — new file, mirrors `pandera/api/ibis/error_handler.py`
- `pandera/backends/narwhals/container.py` — fix `_to_frame_kind_nw`, fix `collect_schema_components`
- `pandera/backends/narwhals/components.py` — use `NarwhalsErrorHandler`
- `pandera/backends/narwhals/base.py` — use `NarwhalsErrorHandler` in `failure_cases_metadata`
- Existing pattern: `pandera/api/ibis/error_handler.py` is the model to follow for the
  new `NarwhalsErrorHandler`

</specifics>

<deferred>
## Deferred Ideas

- Full removal of ibis-specific branching from `run_check` in `base.py` (lines 84-131)
  requires converting ibis CheckResult types inside NarwhalsCheckBackend — this is a
  larger architectural change and may break ibis check-level tests that expect ibis lazy
  types back. Defer to a follow-up phase.
- `drop_invalid_rows` ibis delegation cleanup — already noted as a v2 concern in STATE.md
- Registering Column backend for `nw.LazyFrame` / `nw.DataFrame` types to eliminate
  `_to_native()` in `run_schema_component_checks` — defer pending broader narwhals
  type registration strategy

</deferred>

---

*Phase: 01-pr-review-architecture-fixes*
*Context gathered: 2026-03-21 from PR Review #2223*
