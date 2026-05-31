# Phase 1: Structural Cleanup - Context

**Gathered:** 2026-03-30
**Status:** Ready for planning

<domain>
## Phase Boundary

The narwhals backend has no Polars-specific coupling, no unnecessary eager execution, unified type detection utilities, and custom checks work end-to-end. Scope is confined to `pandera/backends/narwhals/`, `pandera/engines/narwhals_engine.py`, and the schema API layer where the `infer_columns` abstraction lives.

</domain>

<decisions>
## Implementation Decisions

### Lazy/SQL Detection Unification (TYPES-01, TYPES-02, TYPES-03)
- **D-01:** A single `_is_lazy(frame)` utility replaces the existing `_is_lazy_or_sql` function (used only once) and consolidates the three inline `hasattr(native, "execute")` checks in `base.py:71`, `container.py:157`, and `components.py:341`.
- **D-02:** The `failure_cases_metadata` three-branch dispatch in `base.py:209-324` (lazy/SQL path, eager polars path, scalar path) is rewritten to call `_is_lazy` consistently — the branching structure itself is kept; only the detection conditions are unified.

### Backend Isolation (CLEAN-01, CLEAN-02, CLEAN-03, CLEAN-04, CLEAN-05)
- **D-03:** `container.py:14` (`from pandera.api.polars.container import DataFrameSchema`) is moved to a `TYPE_CHECKING` guard — zero behavioral change, used only as a type annotation.
- **D-04:** The inner `import polars as pl` calls in `base.py` (lines 240, 294, 317, 320, 323) are eliminated by rewriting those branches to use narwhals operations only — satisfying CLEAN-03 (no polars required for ibis-only users). The existing v1.1 decision that polars is optional is preserved.
- **D-05:** All inner stdlib imports (`import re` in `container.py:449`, `components.py:102`) and narwhals engine imports (`from pandera.api.narwhals.types import NarwhalsData` at `narwhals_engine.py:34`, `from pandera.api.narwhals.utils import _to_native` at `narwhals_engine.py:56`, and `from pandera.api.narwhals.types import NarwhalsData` at `container.py`) are hoisted to module-level top-of-file. The inner `import polars as pl` calls are handled separately under D-04 (not simply hoisted, as polars is optional).
- **D-06 (folded todo — CLEAN-05):** The `importlib` dynamic import in `container.py:318-323` that synthesizes framework-specific Column objects is replaced by adding `schema.infer_columns(frame_column_names)` to the schema API layer. The backend calls this method instead of reaching into `pandera.api.polars.components` or `pandera.api.ibis.components` directly. This is a broader abstraction fix — the schema knows its own Column type; the backend should not.

### Eager Execution (EAGER-01, EAGER-02)
- **D-07:** `narwhals_engine.py:try_coerce` replaces `lf.collect()` (which materializes the full frame just to detect errors) with `lf.head(1).collect()` as a bounded probe. `.head(1)` exercises the actual cast path with one row, catching both schema-level and execution-time cast errors. The returned value remains the un-collected `lf`.
- **D-08:** `container.py` and `components.py` are audited for `_materialize` / `.collect()` calls that operate on full frames for non-error-detection purposes (lazy concat, dtype checks) and replaced with lazy-safe narwhals alternatives.

### Custom Checks (CHECKS-01)
- **D-09:** Investigate root cause of custom check failures through the narwhals backend. The gap is likely in `checks.py:postprocess_bool_output` for `native=True` checks returning row-wise Series/Column output (not just scalar `bool`). Fix the code path and add a regression test covering both `pl.DataFrame` and `ibis.Table` inputs.

### Claude's Discretion
- Exact placement of `_is_lazy` utility (new `utils.py` vs inline in `base.py`) — follow existing narwhals backend conventions
- Exact narwhals equivalent for the scalar failure_cases path (currently `pl.DataFrame(scalar_failure_cases)`) — use `nw.from_dict` or equivalent that works across backends
- Whether `schema.infer_columns()` lives on `BaseSchema` or as an abstract method on a narwhals-specific mixin

### Folded Todos
- **Push synthetic column construction into schema API layer** (`container.py:318-323`): backend should not reach into framework-specific schema API (Polars or Ibis) to construct Column objects. Fix: `schema.infer_columns(frame_column_names)` method on schema API layer. Captured as D-06.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Core narwhals backend files
- `pandera/backends/narwhals/base.py` — `_is_lazy_or_sql` (line 21-28), `failure_cases_metadata` dispatch (lines 209-324), inline `hasattr` checks (line 71)
- `pandera/backends/narwhals/container.py` — Polars import (line 14), inner imports (line 449), `collect_schema_components` importlib pattern (lines 318-323), `_materialize` usage
- `pandera/backends/narwhals/components.py` — inline `hasattr` check (line 341), inner `import re` (line 102)
- `pandera/backends/narwhals/checks.py` — custom check `_normalize_native_output` (lines 81-101), `postprocess_bool_output` (lines 181-205)
- `pandera/engines/narwhals_engine.py` — `try_coerce` (lines 46-79), inner imports (lines 34, 56), `coerce` method (lines 27-44)

### Schema API layer (for D-06)
- `pandera/api/base/schema.py` — base class to add `infer_columns` method to
- `pandera/api/polars/container.py` — reference for what Polars Column class looks like
- `pandera/api/ibis/container.py` — reference for what Ibis Column class looks like

### Requirements
- `.planning/REQUIREMENTS.md` §Native Type Detection, §Backend Isolation, §Eager Execution, §Custom Checks — acceptance criteria for all D-01 through D-09

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_is_lazy_or_sql` in `base.py:21-28`: existing lazy detection utility — extend or rename rather than replace from scratch
- `nw.from_native(failure_cases, eager_only=False)`: unified pattern from v1.1 for failure case handling — continue using
- `COERCION_ERRORS` tuple in `narwhals_engine.py:14-18`: already captures the right exception types for `try_coerce`

### Established Patterns
- Polars is an optional dependency: any new code path must not `import polars` at module level without a `try/except ImportError` guard
- Inner imports that are optional dependencies (`polars`, `ibis`) stay lazy; inner imports that are always-available (`stdlib`, `pandera.*`) should be hoisted per CLEAN-04
- `TYPE_CHECKING` guard pattern is established in the codebase for annotation-only imports

### Integration Points
- `schema.infer_columns()` (D-06): new method on `BaseSchema` — touches `pandera/api/base/schema.py` and both `pandera/api/polars/container.py` + `pandera/api/ibis/container.py` for the concrete implementations
- `try_coerce` return contract: must still return the un-collected `nw.LazyFrame` — the probe change is only on the error-detection side
- `failure_cases_metadata` scalar path: must produce a native frame type (not narwhals wrapper) consistent with v1.1 `SchemaError.failure_cases` contract

</code_context>

<specifics>
## Specific Ideas

- `.head(1).collect()` as the bounded probe in `try_coerce` — user noted "we'll see when we get to the tests" if ibis needs a different approach
- The `infer_columns` todo is a general schema abstraction fix (not just Polars), intentionally scoped broadly: the schema API layer owns Column type knowledge, the backend should not

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within Phase 1 scope.

### Reviewed Todos (not folded)
None — the only matched todo was folded as D-06.

</deferred>

---

*Phase: 01-structural-cleanup*
*Context gathered: 2026-03-30*
