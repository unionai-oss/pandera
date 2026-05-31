# Phase 3: Fix IbisCheckBackend delegation via apply() type-dispatch - Context

**Gathered:** 2026-03-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Remove the `IbisCheckBackend` delegation from `NarwhalsCheckBackend.__call__` by introducing
a `native` flag on `Check` that controls what `apply()` passes to the check function. Unify
the calling convention for all checks (builtin and user-defined) to `check_fn(frame, key)`.
No new user-facing validation capabilities — purely an architectural clean-up of the narwhals
check dispatch path.

</domain>

<decisions>
## Implementation Decisions

### native flag on Check
- Add `native: bool` parameter to `Check.__init__` (default `True`).
- `native=True`: check function receives the raw native frame — `nw.to_native(frame)` — and
  the column key as a second positional arg: `check_fn(native_frame, key)`.
- `native=False`: check function receives the narwhals-wrapped frame and key:
  `check_fn(nw_frame, key)`. For builtins this is a `nw.LazyFrame`; the narwhals wrapper
  is never unwrapped.
- `native=False` is not exclusively for builtins — library authors or advanced users can
  write Narwhals-syntax checks with `native=False`.

### apply() dispatch — pure flag, no inspection
- `apply()` dispatches solely on `self.check.native`. Remove all Dispatcher/annotation-
  inspection logic (the `isinstance(inner_fn, Dispatcher)` branch and `inspect.signature`
  first-param annotation check). The `native` flag is the single source of truth.
- No type inspection of the check function signature is needed.

### Builtin checks get native=False
- All builtin checks (registered via `MetaCheck` / `CHECK_FUNCTION_REGISTRY` in
  `builtin_checks.py`) must set `native=False` at registration time.
- Builtin check function signatures refactored from `check_fn(data: NarwhalsData, ...)` to
  `check_fn(frame: nw.LazyFrame, key: str, ...)`. `NarwhalsData` is removed from builtin
  check function signatures entirely.

### User checks: consistent (frame, key) calling convention for all backends
- `native=True` (default) user checks on Polars: receive `(pl.LazyFrame, key)` — fixes the
  current gap where Polars user checks received only the raw frame with no key.
- `native=True` user checks on Ibis: receive `(ibis.Table, key)` — replaces the
  `IbisCheckBackend` delegation entirely.
- Both backends use the same two-arg convention. Type errors from running an Ibis-syntax
  check on a Polars input are the user's responsibility (duck typing).

### Ibis output normalization in apply()
- When `native=True` and the check function returns an ibis type (`ir.BooleanScalar`,
  `ir.BooleanColumn`, `ibis.Table`), normalize before returning from `apply()`:
  - `ir.BooleanScalar` → execute to Python `bool`
  - `ir.BooleanColumn` → attach to a single-column table, wrap with `nw.from_native()`
  - `ibis.Table` → wrap with `nw.from_native()`
- After normalization, `postprocess()` always receives `nw.LazyFrame`/`nw.DataFrame` or
  `bool`. No new postprocess branches needed.

### IbisCheckBackend delegation removed
- The entire `try: import ibis` delegation block in `NarwhalsCheckBackend.__call__` is
  removed. All checks (ibis-backed or not) go through the narwhals `apply()` path.

### Breaking change: accepted
- Existing user-defined checks written against `pa.PolarsData` or `pa.IbisData` and run
  through the narwhals backend will need to update their signature to `(frame, key)`. This
  is an acceptable breaking change — the narwhals backend is experimental.

### Claude's Discretion
- Exact normalization implementation for `ir.BooleanColumn` → `nw.DataFrame` (e.g. whether
  to use `ibis.Table.mutate` or column `.name()` before wrapping).
- Whether `NarwhalsData` named tuple is kept (unused but harmless) or removed from
  `pandera/api/narwhals/types.py`.
- Exact parameter names for the refactored builtin check signatures (`frame`/`key` vs
  `data`/`col` vs other).

</decisions>

<specifics>
## Specific Ideas

- The `native` flag is analogous to the existing `element_wise` flag on `Check` — a simple
  boolean that changes dispatch behavior, not a type annotation or decorator.
- Phase 2 TODO comment in `checks.py` is the direct specification for this phase:
  "apply() should unwrap NarwhalsData to the type the check function expects (via type
  annotation inspection), making IbisCheckBackend delegation unnecessary." — Phase 3
  replaces annotation inspection with the explicit `native` flag.
- After this phase, `run_check` in `base.py` also becomes simpler: the ibis-result
  detection block (`_is_ibis_result`) may be reducible or removable since ibis outputs
  are normalized to narwhals in `apply()` before `postprocess()` returns a standard
  `CheckResult`.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `NarwhalsCheckBackend.apply()` (checks.py): dispatch logic lives here — modify to use
  `self.check.native` instead of Dispatcher/annotation inspection.
- `NarwhalsCheckBackend.__call__()` (checks.py): delegation block to remove is lines
  ~170-200 (the `try: import ibis` block).
- `NarwhalsCheckBackend._materialize()` (checks.py): unchanged — still used in
  `postprocess_lazyframe_output`.
- `builtin_checks.py`: all builtin function signatures to refactor from `(data: NarwhalsData)`.
- `run_check` in `base.py`: ibis-result detection block (`_is_ibis_result`) may simplify
  after normalization is done in `apply()`.

### Established Patterns
- `element_wise` flag on `Check` — model for `native` flag (same pattern: boolean on Check,
  checked in backend dispatch).
- `nw.from_native()` / `nw.to_native()` — the standard narwhals wrapping/unwrapping API.
- `try: import ibis as _ibis` guard — existing pattern for optional ibis detection.

### Integration Points
- `pandera/api/checks.py` — `Check.__init__` needs `native` parameter.
- `pandera/backends/narwhals/checks.py` — `apply()` and `__call__()` in
  `NarwhalsCheckBackend`.
- `pandera/backends/narwhals/builtin_checks.py` — all builtin function signatures.
- `tests/backends/narwhals/` — existing tests for custom checks and builtin checks need
  updating to match new signatures; new tests for `native=True` ibis/polars dispatch.

</code_context>

<deferred>
## Deferred Ideas

- Remove `NarwhalsData` from `pandera/api/narwhals/types.py` entirely — if no code uses it
  after the builtin refactor, it can be cleaned up in a follow-up.
- `run_check` `_is_ibis_result` block in `base.py` — evaluate simplification after Phase 3
  implementation; may become a follow-up clean-up.
- Schema construction fix (`pandera.polars`/`pandera.ibis` producing narwhals engine dtypes
  when narwhals backend is active) — still deferred from Phase 2.

</deferred>

---

*Phase: 03-fix-ibischeckbackend-delegation-via-apply-type-dispatch*
*Context gathered: 2026-03-22*
