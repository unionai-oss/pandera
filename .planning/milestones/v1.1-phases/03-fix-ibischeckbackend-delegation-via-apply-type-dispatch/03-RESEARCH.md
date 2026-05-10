# Phase 3: Fix IbisCheckBackend Delegation via apply() Type-Dispatch — Research

**Researched:** 2026-03-22
**Domain:** Narwhals check backend dispatch / ibis integration
**Confidence:** HIGH (all findings from direct codebase inspection)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**native flag on Check**
- Add `native: bool` parameter to `Check.__init__` (default `True`).
- `native=True`: check function receives `nw.to_native(frame)` and key as second positional arg: `check_fn(native_frame, key)`.
- `native=False`: check function receives the narwhals-wrapped frame and key: `check_fn(nw_frame, key)`. For builtins this is a `nw.LazyFrame`; the narwhals wrapper is never unwrapped.
- `native=False` is not exclusively for builtins — library authors or advanced users can write Narwhals-syntax checks with `native=False`.

**apply() dispatch — pure flag, no inspection**
- `apply()` dispatches solely on `self.check.native`. Remove all Dispatcher/annotation-inspection logic (the `isinstance(inner_fn, Dispatcher)` branch and `inspect.signature` first-param annotation check). The `native` flag is the single source of truth.
- No type inspection of the check function signature is needed.

**Builtin checks get native=False**
- All builtin checks registered via `MetaCheck` / `CHECK_FUNCTION_REGISTRY` in `builtin_checks.py` must set `native=False` at registration time.
- Builtin check function signatures refactored from `check_fn(data: NarwhalsData, ...)` to `check_fn(frame: nw.LazyFrame, key: str, ...)`. `NarwhalsData` is removed from builtin check function signatures entirely.

**User checks: consistent (frame, key) calling convention for all backends**
- `native=True` (default) user checks on Polars: receive `(pl.LazyFrame, key)`.
- `native=True` user checks on Ibis: receive `(ibis.Table, key)` — replaces `IbisCheckBackend` delegation entirely.
- Both backends use the same two-arg convention. Type errors from running an Ibis-syntax check on a Polars input are the user's responsibility (duck typing).

**Ibis output normalization in apply()**
- When `native=True` and the check function returns an ibis type (`ir.BooleanScalar`, `ir.BooleanColumn`, `ibis.Table`), normalize before returning from `apply()`:
  - `ir.BooleanScalar` → execute to Python `bool`
  - `ir.BooleanColumn` → attach to a single-column table, wrap with `nw.from_native()`
  - `ibis.Table` → wrap with `nw.from_native()`
- After normalization, `postprocess()` always receives `nw.LazyFrame`/`nw.DataFrame` or `bool`. No new postprocess branches needed.

**IbisCheckBackend delegation removed**
- The entire `try: import ibis` delegation block in `NarwhalsCheckBackend.__call__` is removed. All checks (ibis-backed or not) go through the narwhals `apply()` path.

**Breaking change: accepted**
- Existing user-defined checks written against `pa.PolarsData` or `pa.IbisData` run through the narwhals backend will need to update their signature to `(frame, key)`. Acceptable — the narwhals backend is experimental.

### Claude's Discretion
- Exact normalization implementation for `ir.BooleanColumn` → `nw.DataFrame` (e.g. whether to use `ibis.Table.mutate` or column `.name()` before wrapping).
- Whether `NarwhalsData` named tuple is kept (unused but harmless) or removed from `pandera/api/narwhals/types.py`.
- Exact parameter names for the refactored builtin check signatures (`frame`/`key` vs `data`/`col` vs other).

### Deferred Ideas (OUT OF SCOPE)
- Remove `NarwhalsData` from `pandera/api/narwhals/types.py` entirely — follow-up clean-up.
- `run_check` `_is_ibis_result` block in `base.py` — evaluate simplification after Phase 3; may become a follow-up clean-up.
- Schema construction fix (`pandera.polars`/`pandera.ibis` producing narwhals engine dtypes when narwhals backend is active) — still deferred from Phase 2.
</user_constraints>

---

## Summary

Phase 3 removes a fragile dispatch mechanism in `NarwhalsCheckBackend` — the `try: import ibis` delegation block in `__call__` plus the `Dispatcher`/`inspect.signature` detection in `apply()` — and replaces it with a single explicit `native: bool` flag on `Check`. This flag becomes the authoritative source of truth for how `apply()` unwraps its argument before calling the check function.

The phase touches five files: `pandera/api/checks.py` (add `native` param), `pandera/backends/narwhals/checks.py` (rewrite `apply()`, delete ibis delegation in `__call__`), `pandera/backends/narwhals/builtin_checks.py` (refactor all 14 function signatures), `pandera/api/narwhals/types.py` (keep or remove `NarwhalsData` per discretion), and `tests/backends/narwhals/test_checks.py` (update + add tests).

The `run_check` ibis-result detection block in `base.py` is explicitly deferred. It remains structurally intact through this phase — the goal is that after normalization in `apply()`, ibis outputs are already narwhals types before reaching `run_check`, making the ibis branch there dead code, but removing it is deferred.

**Primary recommendation:** Implement in a single plan: (1) add `native` to `Check`, (2) rewrite `apply()` + delete ibis delegation, (3) refactor all 14 builtin check signatures, (4) update/add tests. The refactors are tightly coupled; doing them in one wave avoids a broken intermediate state.

---

## Standard Stack

### Core Libraries in Use

| Library | Version in Use | Role |
|---------|---------------|------|
| `narwhals.stable.v1` | project-pinned | Wrapping/unwrapping frames (`nw.to_native`, `nw.from_native`) |
| `ibis` | optional import | Target backend for native=True ibis checks |
| `ibis.expr.types` | optional import | Type-checking ibis outputs in normalization |

No new dependencies are introduced. All needed APIs (`nw.to_native`, `nw.from_native`, `try: import ibis`) are already used elsewhere in the file.

---

## Architecture Patterns

### Pattern: element_wise flag as model for native flag

`element_wise` on `Check` is the exact pattern to replicate. It is:
- Stored as `self.element_wise` in `Check.__init__` (line 177 of `checks.py`)
- Read in `NarwhalsCheckBackend.apply()` as `self.check.element_wise` (line 44)
- Never inspected from function signatures — it is a pure boolean on the `Check` object

`native` follows the same pattern: stored at construction time, read at dispatch time.

### Pattern: Builtin registration sets native=False via from_builtin_check_name

Builtin checks are created via `Check.from_builtin_check_name(...)`. The `native=False` default must propagate from there to `Check.__init__`. Two implementation options:

**Option A:** `Check.from_builtin_check_name` hardcodes `native=False` in the call to `Check(...)`. All `Check.equal_to(...)`, `Check.greater_than(...)`, etc. factory methods call `from_builtin_check_name` — they all get `native=False` automatically without touching each factory method.

**Option B:** Register `native=False` via a keyword in `register_builtin_check` decorator. This is more declarative but requires touching the registration infrastructure.

Option A is simpler given the existing code structure. `from_builtin_check_name` is the single choke point for all builtin creation.

### Pattern: Builtin function body after signature refactor

Current builtin signature:
```python
def equal_to(data: NarwhalsData, value: Any) -> nw.LazyFrame:
    return data.frame.select(nw.col(data.key) == value)
```

After refactor (`native=False` path — builtins receive narwhals frame and key):
```python
def equal_to(frame: nw.LazyFrame, key: str, value: Any) -> nw.LazyFrame:
    return frame.select(nw.col(key) == value)
```

This is a mechanical substitution: `data.frame` → `frame`, `data.key` → `key`. Applies identically to all 14 builtins.

### Pattern: apply() dispatch after refactor

Current `apply()` has ~25 lines of Dispatcher/annotation detection before calling the check function. After the refactor:

```python
def apply(self, check_obj: NarwhalsData):
    if self.check.element_wise:
        # unchanged element_wise path
        ...
    elif self.check.native:
        # native=True: unwrap to native, call with (native_frame, key)
        native_frame = nw.to_native(check_obj.frame)
        out = self.check_fn(native_frame, check_obj.key)
        # normalize ibis outputs if present
        out = self._normalize_native_output(out)
    else:
        # native=False: pass narwhals frame and key (builtins path)
        out = self.check_fn(check_obj.frame, check_obj.key)
    ...
```

Note: `check_fn` is a `partial(check._check_fn, **check._check_kwargs)`. For builtins, this means the extra `**kwargs` (e.g. `value=5`) are already bound. The function call `self.check_fn(frame, key)` correctly passes `frame` and `key` as positional args, and the bound kwargs flow in via partial.

### Pattern: Ibis output normalization

The normalization helper for `native=True` returns:
- `ir.BooleanScalar` → `bool` via `.execute()`
- `ir.BooleanColumn` → `nw.DataFrame`: use `.name(CHECK_OUTPUT_KEY)` then `ibis.Table` wrapping with `nw.from_native()`. The column-to-table promotion uses `ibis.Table.select()` or the column's `.as_table()` method if available — Claude's discretion.
- `ibis.Table` → `nw.from_native(table, eager_or_interchange_only=False)`. This matches the existing pattern in `conftest.py` where `ibis.memtable(...)` is wrapped the same way.
- All other types → pass through unchanged (polars LazyFrame, bool from polars check, etc.)

The normalization must be guarded with `try: import ibis` to keep ibis optional.

### Pattern: __call__ simplification

Current `__call__` has 30+ lines. After removing the ibis delegation block, it collapses to the same 4 lines it was before the delegation was added:

```python
def __call__(self, check_obj, key=None):
    check_obj = self.preprocess(check_obj, key)
    narwhals_data = NarwhalsData(check_obj, key or "*")
    check_output = self.apply(narwhals_data)
    return self.postprocess(narwhals_data, check_output)
```

The `NarwhalsData` named tuple is still used internally in `apply()` (as the argument type) and in `postprocess()`/`postprocess_lazyframe_output()`. It is not removed in this phase.

### Recommended Project Structure (files to change)

```
pandera/
├── api/
│   ├── checks.py                          # Add native: bool param to __init__
│   └── narwhals/
│       └── types.py                       # NarwhalsData stays (used internally)
└── backends/
    └── narwhals/
        ├── checks.py                      # Rewrite apply(), remove ibis delegation
        └── builtin_checks.py              # Refactor all 14 builtin signatures
tests/
└── backends/
    └── narwhals/
        └── test_checks.py                 # Update existing, add native=True ibis tests
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Ibis type detection | Custom isinstance chains | `try: import ibis.expr.types as ir; isinstance(out, ir.BooleanScalar)` — already used in ibis checks.py |
| Narwhals unwrapping | Manual `.__class__.__module__` inspection | `nw.to_native(frame)` — already used in `__call__` |
| Narwhals wrapping of ibis Table | Custom adapter | `nw.from_native(table, eager_or_interchange_only=False)` — used in conftest.py fixture |
| BooleanScalar → bool | Manual ibis execution logic | `.execute()` — used in `run_check` ibis path |

---

## Common Pitfalls

### Pitfall 1: Partial binding order with refactored builtins

**What goes wrong:** Builtin check functions are stored as `partial(check._check_fn, **check._check_kwargs)`. After the signature change from `(data, value)` to `(frame, key, value)`, calling `self.check_fn(frame, key)` works correctly because `value` is already bound via kwargs. But if `value` were a positional-only arg, the partial would shadow it.

**How to avoid:** Keep extra statistics args as keyword arguments in builtin signatures (they already are: `value: Any`, `min_value: Any`, etc.). The refactor only moves `data: NarwhalsData` → two positional args `(frame, key)`.

**Warning signs:** `TypeError: got multiple values for argument` when running builtin checks.

### Pitfall 2: native=True user checks must pass key as second positional arg

**What goes wrong:** Current user-defined check path calls `check_fn(native_frame)` (no key). After the phase, the contract changes to `check_fn(native_frame, key)`. Existing user code that defines `def my_check(frame): ...` (one arg) will get a `TypeError`.

**How to avoid:** This is a documented breaking change. The research confirms it is intentional. Tests must validate the new two-arg convention passes correctly.

**Warning signs:** `TypeError: my_check() takes 1 positional argument but 2 were given` from user check tests.

### Pitfall 3: ibis BooleanColumn normalization — column needs a name

**What goes wrong:** `nw.from_native(bool_column)` is not valid — narwhals expects a `Table`, not a `Column`. An `ir.BooleanColumn` must be promoted to a table before wrapping.

**How to avoid:** Use `bool_column.name(CHECK_OUTPUT_KEY)` to name it (per `IbisCheckBackend.postprocess_boolean_column_output` which does exactly this on line 161 of `backends/ibis/checks.py`), then place it in a table. Alternatively, call `ibis.table(...)` or use `column.as_table()` if available.

**Established pattern from ibis/checks.py:**
```python
# Line 161 — existing ibis backend does this:
check_output = check_output.name(CHECK_OUTPUT_KEY)
failure_cases = check_obj.table.filter(~check_output)
```

For normalization in `apply()`, the goal is a `nw.LazyFrame` or `nw.DataFrame` with a boolean column. The simplest approach: create a one-column ibis table by doing `check_obj.frame`'s native table `.select(bool_column.name(CHECK_OUTPUT_KEY))`, then `nw.from_native(...)`.

### Pitfall 4: from_builtin_check_name must pass native=False down

**What goes wrong:** If `Check.__init__` defaults `native=True` but `from_builtin_check_name` doesn't explicitly pass `native=False`, all builtins will default to `native=True` and try to call `check_fn(native_frame, key)` — which will fail because the refactored signatures expect `(frame, key)` (narwhals, not native).

**How to avoid:** Explicitly set `native=False` in the call inside `from_builtin_check_name`. Verify by running the existing `test_builtin_checks_pass` parametrized test.

### Pitfall 5: inspect / Dispatcher imports become dead code

**What goes wrong:** After removing the Dispatcher detection, `import inspect` and `from pandera.api.function_dispatch import Dispatcher` at the top of `checks.py` become unused.

**How to avoid:** Remove both imports from `pandera/backends/narwhals/checks.py` when removing the detection logic. Flake8/ruff will catch this, but proactively removing prevents confusion.

### Pitfall 6: unique_values_eq is missing from narwhals builtin_checks.py

**What goes wrong:** The narwhals `builtin_checks.py` has 14 checks but `unique_values_eq` (which exists in pandas, polars, and ibis backends) is absent. This is a pre-existing gap, not introduced by this phase.

**How to avoid:** Do not add it in this phase — it is out of scope. Do not break existing behavior by accidentally deleting checks that exist. The 14 checks to refactor are exactly the ones already present in the file.

---

## Code Examples

### Check.__init__ — adding native parameter (modeled on element_wise)

```python
# pandera/api/checks.py — Check.__init__ signature addition
def __init__(
    self,
    check_fn: Callable,
    groups=None,
    groupby=None,
    ignore_na: bool = True,
    element_wise: bool = False,
    native: bool = True,        # NEW: True=raw native frame, False=narwhals frame
    name=None,
    error=None,
    ...
) -> None:
    ...
    self.element_wise = element_wise
    self.native = native          # NEW: stored like element_wise
```

### from_builtin_check_name — propagating native=False

```python
# Check.from_builtin_check_name must pass native=False for all builtins
# The exact location depends on how this method is implemented,
# but the pattern is: wherever Check(...) is constructed for a builtin,
# add native=False.
```

### apply() — simplified dispatch

```python
def apply(self, check_obj: NarwhalsData):
    if self.check.element_wise:
        # unchanged — element_wise path
        selector = nw.col(check_obj.key or "*")
        try:
            out = check_obj.frame.with_columns(
                selector.map_batches(self.check_fn, return_dtype=nw.Boolean)
            ).select(selector)
        except NotImplementedError:
            raise NotImplementedError(
                "element_wise checks are not supported on SQL-lazy backends ..."
            )
    elif self.check.native:
        # native=True: unwrap to backend-native type, call (native_frame, key)
        native_frame = nw.to_native(check_obj.frame)
        out = self.check_fn(native_frame, check_obj.key)
        out = self._normalize_native_output(out, check_obj)
    else:
        # native=False: pass narwhals frame and key (builtin checks path)
        out = self.check_fn(check_obj.frame, check_obj.key)

    if isinstance(out, bool):
        return out
    # rename / reduce to CHECK_OUTPUT_KEY ... (unchanged)
```

### _normalize_native_output — ibis output normalization

```python
@staticmethod
def _normalize_native_output(out, check_obj: NarwhalsData):
    """Normalize ibis outputs from native=True checks to narwhals types."""
    try:
        import ibis
        import ibis.expr.types as ir
        if isinstance(out, ir.BooleanScalar):
            return bool(out.execute())
        elif isinstance(out, ir.BooleanColumn):
            # Promote to table and wrap with narwhals
            native = nw.to_native(check_obj.frame)
            tbl = native.select(out.name(CHECK_OUTPUT_KEY))
            return nw.from_native(tbl, eager_or_interchange_only=False)
        elif isinstance(out, ibis.Table):
            return nw.from_native(out, eager_or_interchange_only=False)
    except ImportError:
        pass
    return out  # polars or bool — already correct type
```

### Builtin function signature refactor pattern

```python
# Before:
@register_builtin_check(aliases=["eq"], error="equal_to({value})")
def equal_to(data: NarwhalsData, value: Any) -> nw.LazyFrame:
    return data.frame.select(nw.col(data.key) == value)

# After:
@register_builtin_check(aliases=["eq"], error="equal_to({value})")
def equal_to(frame: nw.LazyFrame, key: str, value: Any) -> nw.LazyFrame:
    return frame.select(nw.col(key) == value)
```

Same mechanical pattern for all 14 builtins. The `register_builtin_check` decorator reads the first parameter's annotation to build the `Dispatcher` registry — the annotation changes from `NarwhalsData` to `nw.LazyFrame`. This affects `Dispatcher._function_registry` keying, but since the Dispatcher detection is being removed from `apply()`, the registry is no longer consulted for dispatch decisions.

### __call__ after ibis delegation removal

```python
def __call__(self, check_obj: nw.LazyFrame, key: Optional[str] = None) -> CheckResult:
    check_obj = self.preprocess(check_obj, key)
    narwhals_data = NarwhalsData(check_obj, key or "*")
    check_output = self.apply(narwhals_data)
    return self.postprocess(narwhals_data, check_output)
```

---

## State of the Art

| Old Approach | New Approach | Impact |
|--------------|--------------|--------|
| Dispatcher registry lookup + `inspect.signature` first-param annotation | `self.check.native` boolean flag | Eliminates runtime reflection; dispatch is O(1) attribute read |
| `try: import ibis` delegation to `IbisCheckBackend` for user-defined ibis checks | `native=True` passes `(ibis.Table, key)` directly; normalization in `apply()` | Single code path for all backends |
| `check_fn(native_frame)` — one-arg call for user checks (no key) | `check_fn(native_frame, key)` — consistent two-arg convention | Polars and Ibis user checks have same signature |
| Builtins detected via Dispatcher registry + annotation | Builtins set `native=False` at construction | Simple, explicit, no magic |

---

## Open Questions

1. **NarwhalsData import — used elsewhere after builtin refactor?**
   - What we know: `NarwhalsData` is currently imported in `checks.py` (for type hints) and `builtin_checks.py` (for annotations). After the refactor, builtin functions no longer annotate with `NarwhalsData`. `apply()` still accepts `check_obj: NarwhalsData` and `__call__` constructs `NarwhalsData(check_obj, key or "*")`.
   - What's unclear: Whether to keep `NarwhalsData` as the internal container type in `apply()` or switch to a plain tuple/dataclass. Keeping it is simplest and is explicitly in scope (deferred removal).
   - Recommendation: Keep `NarwhalsData` as the internal container — just remove it from builtin function signatures.

2. **register_builtin_check decorator — first-param annotation change**
   - What we know: `register_builtin_check` in `extensions.py` reads `fn_sig.parameters[0].annotation` to determine what type to register in the `Dispatcher` registry (line 56 of extensions.py). After the refactor, the first param annotation of builtin functions changes from `NarwhalsData` to `nw.LazyFrame`.
   - What's unclear: Whether any code path still dispatches through the `Dispatcher` after `apply()` removes the Dispatcher detection. The `Dispatcher.__call__` resolves by `type(args[0])`, but `apply()` will no longer call `check_fn(check_obj)` where `check_obj` is a `NarwhalsData`.
   - Recommendation: The Dispatcher is still called for other backends (pandas, polars). For the narwhals backend, `apply()` bypasses Dispatcher dispatch entirely and calls the registered function directly. The annotation on the narwhals builtin functions just needs to not break the `Dispatcher.register()` call — `nw.LazyFrame` is a valid annotation.

3. **BooleanColumn normalization — which ibis API to use for column-to-table promotion**
   - What we know: `ir.BooleanColumn.name(str)` exists (used on line 161 of `backends/ibis/checks.py`). The resulting named column needs to become a table that narwhals can wrap.
   - What's unclear: Whether `ibis.Table.select(named_column)` is the correct promotion or whether another API exists.
   - Recommendation: Claude's discretion per CONTEXT.md. The safest path is to use the existing ibis backend's pattern: `col.name(CHECK_OUTPUT_KEY)` then construct a table expression. Validate by running the ibis parametrized tests.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | none detected (uses pytest defaults) |
| Quick run command | `pytest tests/backends/narwhals/test_checks.py -x` |
| Full suite command | `pytest tests/backends/narwhals/ -x` |

### Phase Requirements to Test Map

| Behavior | Test Type | Automated Command |
|----------|-----------|-------------------|
| `native=True` user check on Polars receives `(pl.LazyFrame, key)` | unit | `pytest tests/backends/narwhals/test_checks.py::test_native_true_user_check_polars -x` |
| `native=True` user check on Ibis receives `(ibis.Table, key)` | unit | `pytest tests/backends/narwhals/test_checks.py::test_native_true_user_check_ibis -x` |
| `native=False` user check receives narwhals frame and key | unit | `pytest tests/backends/narwhals/test_checks.py::test_native_false_user_check -x` |
| All 14 builtin checks still pass on Polars after refactor | parametrized | `pytest tests/backends/narwhals/test_checks.py::test_builtin_checks_pass -x` |
| All 14 builtin checks still pass on Ibis after refactor | parametrized | `pytest tests/backends/narwhals/test_checks.py::test_builtin_checks_pass -x` |
| Ibis `ir.BooleanScalar` output from native=True check normalizes to bool | unit | `pytest tests/backends/narwhals/test_checks.py::test_ibis_boolean_scalar_normalization -x` |
| Ibis `ir.BooleanColumn` output from native=True check normalizes to nw.LazyFrame | unit | `pytest tests/backends/narwhals/test_checks.py::test_ibis_boolean_column_normalization -x` |
| Ibis `ibis.Table` output from native=True check normalizes to nw.LazyFrame | unit | `pytest tests/backends/narwhals/test_checks.py::test_ibis_table_normalization -x` |
| element_wise on SQL-lazy still raises NotImplementedError | unit | `pytest tests/backends/narwhals/test_checks.py::test_element_wise_sql_lazy_raises -x` |

### Sampling Rate
- **Per task commit:** `pytest tests/backends/narwhals/test_checks.py -x`
- **Per wave merge:** `pytest tests/backends/narwhals/ -x`
- **Phase gate:** Full narwhals backend suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] New test functions for `native=True` dispatch (Polars and Ibis variants) — covers the new convention tests
- [ ] `test_builtin_check_routing` is currently marked `xfail(strict=False)` — update to pass after refactor (the xfail reason references "plan 02-03" which is now complete; this test should be un-xfailed and its assertion adjusted for the new `native=False` semantics)

---

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection — `pandera/backends/narwhals/checks.py` (full file)
- Direct codebase inspection — `pandera/backends/narwhals/builtin_checks.py` (full file)
- Direct codebase inspection — `pandera/api/checks.py` (full file)
- Direct codebase inspection — `pandera/backends/narwhals/base.py` (full file)
- Direct codebase inspection — `pandera/backends/ibis/checks.py` (full file)
- Direct codebase inspection — `pandera/api/function_dispatch.py` (full file)
- Direct codebase inspection — `pandera/api/extensions.py` (full file)
- Direct codebase inspection — `pandera/api/narwhals/types.py` (full file)
- Direct codebase inspection — `tests/backends/narwhals/test_checks.py` (full file)
- Direct codebase inspection — `tests/backends/narwhals/conftest.py` (full file)

### Secondary (MEDIUM confidence)
- `.planning/phases/03-fix-ibischeckbackend-delegation-via-apply-type-dispatch/03-CONTEXT.md` — user decisions, locked architecture choices

---

## Metadata

**Confidence breakdown:**
- Dispatch mechanism (apply, __call__, native flag): HIGH — read all relevant source files directly
- Builtin check refactor (14 functions): HIGH — read all 14 functions; pattern is mechanical
- Ibis normalization: MEDIUM — pattern extrapolated from existing `IbisCheckBackend` code; exact BooleanColumn table-promotion API is Claude's discretion
- Test infrastructure: HIGH — read existing test files and conftest directly

**Research date:** 2026-03-22
**Valid until:** Stable — this is an internal refactor with no external dependency changes
