# Phase 5: Expression-based Check Protocol — Research

**Researched:** 2026-03-22
**Domain:** narwhals check backend, expression protocol, Dispatcher dispatch
**Confidence:** HIGH

## Summary

The current `apply()` method in `NarwhalsCheckBackend` has two distinct code paths for polars
vs ibis backends. The ibis path requires a row_number join hack because narwhals cannot pass
a Series from one ibis relation into `with_columns` of another. The root cause is that builtin
check functions return a computed `nw.LazyFrame` (a 1-column bool result from `frame.select(...)`)
rather than a declarative `nw.Expr` (an unevaluated expression). When the result is an expression
instead, `frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))` works uniformly for both polars and
ibis — no backend-specific branching required.

The key architectural insight is that `nw.col('x') > 0` returns a `nw.Expr` object that can be
passed as an argument to `frame.with_columns()` regardless of whether `frame` is a polars
`nw.LazyFrame` or an ibis `nw.DataFrame`. The narwhals translation layer handles the backend
difference transparently. This is verified experimentally: both polars and ibis frames accept
the same expression and produce a wide table with the CHECK_OUTPUT_KEY column appended.

The scope covers three call paths inside `apply()`: the `native=False` builtin-check path (the
primary target), the `element_wise` path (already uses expressions via `map_batches`, minor
cleanup), and the `native=True` custom-check path (unchanged — receives native frame and key,
must handle its own output normalization).

**Primary recommendation:** Change builtin check functions to accept `nw.Expr` as their first
argument (instead of `frame: nw.LazyFrame, key: str`) and return `nw.Expr`. Rekey the
`Dispatcher` on `nw.Expr`. Simplify `apply()` native=False path to a single
`frame.with_columns(check_fn(nw.col(key)).alias(CHECK_OUTPUT_KEY))` call for column checks,
and `frame.with_columns(check_fn(frame).alias(CHECK_OUTPUT_KEY))` for frame-level checks.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Check functions will return narwhals expressions, not computed series.
- This covers both builtin checks and the custom check protocol.
- Narrow fix (only builtins, keep row_number fallback for custom) was explicitly rejected in
  favor of the wide approach.
- `apply()` should use `frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))` for all backends
  — polars and ibis alike — with no isinstance/hasattr branching.
- The row_number join in the current ibis path of `apply()` must be eliminated.
- Check functions currently receive a materialized series and return a bool series. New protocol:
  check functions receive a narwhals column expression (or frame) and return a bool expression.
- Simple checks are transparent: `lambda s: s > 10` works the same way since narwhals
  expressions support the same operators as series.
- `element_wise=True` checks use `map_elements(fn)` which is a valid narwhals expression.
  These remain fully supported — arbitrary Python still runs per-element.
- Checks that need to compute something in Python at validation time should compute the scalar
  eagerly and embed it as a literal in the returned expression.
- Checks that imperatively manipulate a materialized series are the main backward-compat concern.
  The tradeoff is accepted.
- Not a primary concern for this codebase — internal architectural change. For the public API,
  checks using standard narwhals/pandas-style operators will be transparent. Checks calling
  materialization methods (.to_pandas(), .to_list()) will break.

### Claude's Discretion
- Exact shape of what the check function receives (column expression vs frame + key)
- Whether element_wise checks need a separate protocol path
- Migration path for any existing pandera tests that use imperative check functions
- Whether to introduce a deprecation shim or just update all call sites

### Deferred Ideas (OUT OF SCOPE)
- Cross-row aggregation checks that require Python-level multi-pass computation (acknowledged
  as an edge case, not blocking)
- Making the new protocol extensible for future backends beyond polars/ibis
</user_constraints>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| narwhals.stable.v1 | 2.15.0 | Expression API, frame operations | Already used throughout the backend |
| narwhals.stable.v1.Expr | 2.15.0 | Declarative expression type for Dispatcher dispatch | `nw.Expr` is the unified type from both `nw.col()` and expression operations |

### Key narwhals APIs
| API | Returns | Purpose |
|-----|---------|---------|
| `nw.col(key)` | `nw.Expr` | Creates a column expression |
| `nw.col(key) > value` | `nw.Expr` | Comparison expression |
| `nw.col(key).is_between(min, max, closed=...)` | `nw.Expr` | Range check expression |
| `nw.col(key).is_in(values)` | `nw.Expr` | Membership check expression |
| `nw.col(key).str.contains(pattern)` | `nw.Expr` | String pattern check expression |
| `nw.col(key).str.starts_with(s)` | `nw.Expr` | String prefix check expression |
| `nw.col(key).str.ends_with(s)` | `nw.Expr` | String suffix check expression |
| `nw.col(key).str.len_chars()` | `nw.Expr` | String length expression |
| `nw.col(key).map_batches(fn, return_dtype=nw.Boolean)` | `nw.Expr` | Batch-mapped fn expression (element_wise) |
| `frame.with_columns(expr.alias(name))` | `nw.LazyFrame` or `nw.DataFrame` | Appends expression as named column |

**Installation:** No new dependencies — narwhals 2.15.0 is already installed.

## Architecture Patterns

### Current Call Chain (to understand what changes)

```
Check.__call__(check_obj, column)
  -> backend = Check.get_backend(check_obj)(self)   # NarwhalsCheckBackend
  -> backend(check_obj, column)                     # __call__
     -> preprocess(check_obj, key)                  # identity
     -> narwhals_data = NarwhalsData(frame, key)
     -> apply(narwhals_data)                        # <--- target of this phase
     -> postprocess(narwhals_data, check_output)
```

### Current apply() Structure (to be replaced)

The current `apply()` has three branches:

1. **element_wise branch** — `nw.col(key).map_batches(check_fn).select(selector)`
   - Returns: 1-column `nw.LazyFrame` or `nw.DataFrame`
   - Problem: `.select(selector)` discards data columns; then complex reassembly

2. **native=True branch** — `check_fn(native_frame, key)` + `_normalize_native_output()`
   - Returns: ibis Column/Table or polars frame
   - Problem: needs `_normalize_native_output`, then same reassembly path

3. **native=False branch** — the builtin-check path
   - Has a Dispatcher workaround for ibis (ibis frame is `nw.DataFrame`, Dispatcher keyed on
     `nw.LazyFrame`, so lookup fails at runtime → explicit `_function_registry[nw.LazyFrame]` hack)
   - Returns 1-col result from `frame.select(...)` that must be renamed + joined back via
     row_number hack for ibis

After these branches, the result is "reassembled" into a wide table. For ibis, this requires
the row_number join (lines 111-142 in `checks.py`). This is what must be eliminated.

### New apply() Structure (target state)

```python
def apply(self, check_obj: NarwhalsData):
    frame = check_obj.frame
    key = check_obj.key

    if self.check.element_wise:
        # element_wise: nw.Expr via map_batches — raises NotImplementedError on ibis
        selector = nw.col(key or "*")
        try:
            expr = selector.map_batches(self.check_fn, return_dtype=nw.Boolean)
        except NotImplementedError:
            raise NotImplementedError("element_wise checks not supported on SQL-lazy backends...")
        return frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))

    elif self.check.native:
        # native=True: unchanged — receive native frame+key, normalize output
        native_frame = nw.to_native(frame)
        out = self.check_fn(native_frame, key)
        return self._normalize_native_output(out, check_obj)

    else:
        # native=False: expression protocol
        # Column check (key != "*"): pass nw.col(key) expression
        # Frame check (key == "*"): pass frame itself
        if key and key != "*":
            expr = self.check_fn(nw.col(key))
        else:
            expr = self.check_fn(frame)
        return frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))
```

**Key result:** `apply()` for native=False now returns `frame.with_columns(expr.alias(...))` —
same for polars (`nw.LazyFrame`) and ibis (`nw.DataFrame`). No row_number join, no
`isinstance` branching, no Dispatcher hack.

### Builtin Check Function Signature Change

**Current signature** (all 14 builtins):
```python
def equal_to(frame: nw.LazyFrame, key: str, value: Any) -> nw.LazyFrame:
    return frame.select(nw.col(key) == value)
```

**New signature** (all 14 builtins):
```python
def equal_to(col_expr: nw.Expr, value: Any) -> nw.Expr:
    return col_expr == value
```

The Dispatcher changes dispatch key from `nw.LazyFrame` to `nw.Expr` because
`register_builtin_check` uses the first argument's type annotation to key the registry.

### Frame-level checks (key == "*")

For frame-level checks, the check function receives the narwhals frame rather than an expression:

```python
def check_fn(frame):
    return nw.col('col_a') > nw.col('col_b')  # still returns nw.Expr
```

The `apply()` would call `check_fn(frame)` and pass the result to `frame.with_columns(...)`.
This covers the case where the check needs to reference multiple columns by name.

### element_wise protocol (unchanged logic, simplified code)

Current element_wise path does `.select(selector)` at the end to narrow to 1 column, then
the reassembly path joins it back. In the new protocol:

```python
# Current:
selector.map_batches(check_fn, return_dtype=nw.Boolean)
# -> assigned to out which is later reassembled

# New: directly produce wide table
expr = nw.col(key).map_batches(check_fn, return_dtype=nw.Boolean)
return frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))
```

Note: narwhals has `map_batches` on `Expr`, not `map_elements`. The current code already
uses `map_batches` (verified in source). The ibis guard (NotImplementedError) is preserved.

### Recommended Builtin Function Shapes (all 14)

```python
# Comparison checks (6):
def equal_to(col_expr: nw.Expr, value: Any) -> nw.Expr:
    return col_expr == value

def not_equal_to(col_expr: nw.Expr, value: Any) -> nw.Expr:
    return col_expr != value

def greater_than(col_expr: nw.Expr, min_value: Any) -> nw.Expr:
    return col_expr > min_value

def greater_than_or_equal_to(col_expr: nw.Expr, min_value: Any) -> nw.Expr:
    return col_expr >= min_value

def less_than(col_expr: nw.Expr, max_value: Any) -> nw.Expr:
    return col_expr < max_value

def less_than_or_equal_to(col_expr: nw.Expr, max_value: Any) -> nw.Expr:
    return col_expr <= max_value

# Range check (1):
def in_range(col_expr: nw.Expr, min_value, max_value, include_min=True, include_max=True) -> nw.Expr:
    closed = _CLOSED_MAP[(include_min, include_max)]
    return col_expr.is_between(min_value, max_value, closed=closed)

# Set membership (2):
def isin(col_expr: nw.Expr, allowed_values) -> nw.Expr:
    return col_expr.is_in(allowed_values)

def notin(col_expr: nw.Expr, forbidden_values) -> nw.Expr:
    return ~col_expr.is_in(forbidden_values)

# String checks (5):
def str_matches(col_expr: nw.Expr, pattern) -> nw.Expr:
    pattern = pattern.pattern if isinstance(pattern, re.Pattern) else pattern
    if not pattern.startswith("^"):
        pattern = f"^{pattern}"
    return col_expr.str.contains(pattern)

def str_contains(col_expr: nw.Expr, pattern) -> nw.Expr:
    pattern = pattern.pattern if isinstance(pattern, re.Pattern) else pattern
    return col_expr.str.contains(pattern)

def str_startswith(col_expr: nw.Expr, string: str) -> nw.Expr:
    return col_expr.str.starts_with(string)

def str_endswith(col_expr: nw.Expr, string: str) -> nw.Expr:
    return col_expr.str.ends_with(string)

def str_length(col_expr: nw.Expr, min_value=None, max_value=None, exact_value=None) -> nw.Expr:
    n_chars = col_expr.str.len_chars()
    if exact_value is not None:
        return n_chars == exact_value
    if min_value is None:
        return n_chars <= max_value
    if max_value is None:
        return n_chars >= min_value
    return n_chars.is_between(min_value, max_value, closed="both")
```

### Anti-Patterns to Avoid

- **Calling frame.select() inside a builtin check:** This materializes the check result from
  the frame, breaking the expression protocol. Return the expression directly.
- **Returning nw.LazyFrame from a builtin:** Any non-Expr return type breaks the uniform
  `frame.with_columns(expr.alias(...))` pattern.
- **Re-introducing isinstance(frame, nw.LazyFrame) branching:** The entire point of this
  phase is to eliminate backend-specific branching. The expression protocol removes the need.
- **Using Dispatcher keyed on nw.LazyFrame for new protocol:** After the change, the Dispatcher
  must be keyed on nw.Expr (the new first-arg type annotation in builtins).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Attaching a 1-col result to the frame | row_number join, horizontal concat | `frame.with_columns(expr.alias(name))` | narwhals handles backend translation |
| Dispatching on frame type | isinstance checks, explicit registry lookup | Dispatcher keyed on nw.Expr | Dispatcher already supports arbitrary type dispatch |
| Element-wise Python functions | Custom loop or series.map | `nw.col(key).map_batches(fn, return_dtype=nw.Boolean)` | narwhals Expr API |
| String operations | Custom regex per backend | `nw.Expr.str.contains/starts_with/ends_with/len_chars` | narwhals string namespace |

**Key insight:** `frame.with_columns(nw.Expr.alias(name))` is the universal operation.
Once check functions return `nw.Expr`, backend-specific paths disappear.

## Common Pitfalls

### Pitfall 1: Dispatcher dispatch key mismatch
**What goes wrong:** Builtin signatures change first arg from `nw.LazyFrame` to `nw.Expr`,
but `register_builtin_check` uses the first-arg annotation to key the Dispatcher. If the
annotation is not updated, the Dispatcher remains keyed on `nw.LazyFrame`, and calling
`check_fn(nw.col(key))` fails with a `KeyError` because `type(nw.col(key))` is `nw.Expr`.
**Why it happens:** `register_builtin_check` calls `get_first_arg_type(fn)` which reads the
type annotation of the first parameter.
**How to avoid:** Annotate the first parameter of all 14 builtins as `col_expr: nw.Expr`.
**Warning signs:** `KeyError` in `Dispatcher.__call__` referencing `nw.Expr`.

### Pitfall 2: test_builtin_check_routing test captures nw.LazyFrame
**What goes wrong:** `test_builtin_check_routing` monkey-patches the Dispatcher registry under
`nw.LazyFrame` and asserts the check function receives `(nw.LazyFrame, key)`. After the
protocol change, the builtin receives `(nw.Expr,)` and is keyed under `nw.Expr`.
**How to avoid:** Update `test_builtin_check_routing` to capture `nw.Expr` and assert the
check receives a `nw.Expr` argument. Update `_function_registry[nw.Expr]` patch target.

### Pitfall 3: postprocess expects the right output type
**What goes wrong:** The new `apply()` for the ibis path returns `nw.DataFrame`
(from `ibis_frame.with_columns(...)`), whereas the old code returned `nw.LazyFrame`
(from the row_number join which used `nw.from_native(wide_native, ...)`).
`postprocess_lazyframe_output` already handles both `nw.LazyFrame` and `nw.DataFrame`,
so this should work. However, `run_check` in `base.py` has `_is_ibis_result` detection
that checks `check_result.check_passed` for `ir.BooleanScalar/Column`.
**How to avoid:** Confirm `_is_ibis_result` detection is not triggered by the new ibis path.
The new ibis path returns `nw.DataFrame` from `postprocess`, not `ibis.BooleanScalar`, so
`_is_ibis_result` should be False — the polars/narwhals path in `run_check` handles it correctly.

### Pitfall 4: frame-level checks (key == "*") protocol ambiguity
**What goes wrong:** When `key == "*"`, the current code calls `check_fn(frame, key)`. The new
protocol's choice matters: if we call `check_fn(nw.col("*"))`, the function gets an
expression for all columns, but frame-level checks often need to reference multiple named
columns (e.g., `nw.col('a') > nw.col('b')`). The frame argument is needed as context.
**How to avoid:** For `key == "*"` (frame-level checks), pass the frame: `check_fn(frame)`.
For column checks (`key` is a real column name), pass `nw.col(key)`.

### Pitfall 5: native=True path _normalize_native_output still needed
**What goes wrong:** The `native=True` path still returns ibis `BooleanScalar`, `BooleanColumn`,
or `ibis.Table` from user-supplied check functions. `_normalize_native_output` must remain.
**How to avoid:** Do not touch the `native=True` branch during this phase.

### Pitfall 6: Dispatcher registry cache_clear between tests
**What goes wrong:** `test_builtin_check_routing` patches `Dispatcher._function_registry`
directly. After the change, if the Dispatcher is re-created via `cache_clear()`, the patch
may persist or may not. The conftest already calls `register_polars_backends.cache_clear()`.
**How to avoid:** Ensure the test restores the original registry entry on teardown (the current
test already has a `finally` block doing this — verify it uses the correct key `nw.Expr`).

## Code Examples

Verified patterns from experimental testing (narwhals 2.15.0):

### Expression-based column check — uniform across polars and ibis
```python
# Source: experimental verification 2026-03-22
import narwhals.stable.v1 as nw

# Works for both polars nw.LazyFrame and ibis nw.DataFrame:
expr = nw.col("x") > 0          # -> nw.Expr
result = frame.with_columns(expr.alias("__check_output__"))
# result type: nw.LazyFrame (polars), nw.DataFrame (ibis)
# both contain original data cols + "__check_output__" bool col
```

### Builtin check new signature (equal_to example)
```python
# Source: experimental verification 2026-03-22
def equal_to(col_expr: nw.Expr, value: Any) -> nw.Expr:
    return col_expr == value

# Called as: check_fn(nw.col(key)) -> nw.Expr
# Then: frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))
```

### Dispatcher keyed on nw.Expr
```python
# Source: experimental verification 2026-03-22
from pandera.api.function_dispatch import Dispatcher

d = Dispatcher()

def equal_to(col_expr: nw.Expr, value) -> nw.Expr:
    return col_expr == value

d.register(equal_to)
# d._function_registry -> {nw.Expr: equal_to}
# d(nw.col("x"), value=5) -> col(x).__eq__(5)  [nw.Expr]
```

### element_wise expression (map_batches, not map_elements)
```python
# Source: narwhals docs + experimental verification
# narwhals.stable.v1.Expr has map_batches (not map_elements)
expr = nw.col(key).map_batches(check_fn, return_dtype=nw.Boolean)
result = frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))
```

### Lambda check transparency (operator works on nw.Expr)
```python
# Source: experimental verification 2026-03-22
check_fn = lambda s: s > 10
expr = check_fn(nw.col("x"))   # -> nw.Expr  (transparent!)
# nw.Expr.__gt__ returns nw.Expr, same as on Series
```

### str_length new builtin
```python
# Source: derived from current builtin
def str_length(col_expr: nw.Expr, min_value=None, max_value=None, exact_value=None) -> nw.Expr:
    n_chars = col_expr.str.len_chars()
    if exact_value is not None:
        return n_chars == exact_value
    if min_value is None:
        return n_chars <= max_value
    if max_value is None:
        return n_chars >= min_value
    return n_chars.is_between(min_value, max_value, closed="both")
```

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Builtin returns `nw.LazyFrame` from `frame.select(...)` | Builtin returns `nw.Expr` directly | Eliminates row_number join |
| Dispatcher keyed on `nw.LazyFrame` | Dispatcher keyed on `nw.Expr` | No ibis workaround needed |
| ibis path: row_number join to attach bool col | `frame.with_columns(expr.alias(...))` | ~30 lines → 1 line |
| apply() has 3 complex branches + reassembly | apply() has 3 lean branches, all returning wide table | No backend-specific code |

**Deprecated/removed after this phase:**
- `_normalize_native_output` remains for native=True; no longer needed for native=False
- The ibis row_number join block in `apply()` (lines 103-142 in current `checks.py`)
- The Dispatcher ibis workaround in `apply()` (lines 69-83 in current `checks.py`)
- The rename/all_horizontal normalization block (lines 89-97 in current `checks.py`)

## Open Questions

1. **Frame-level custom checks with native=False**
   - What we know: Check CONTEXT.md says `check_fn(frame).alias(CHECK_OUTPUT_KEY)` for key=="*"
   - What's unclear: Does passing `frame` (nw.DataFrame for ibis, nw.LazyFrame for polars)
     change behavior for existing user custom checks that use `native=False`?
   - Recommendation: For key=="*", call `check_fn(frame)` and require the return to be `nw.Expr`.
     Document this as the protocol. User checks that returned `nw.LazyFrame` previously will break.

2. **_normalize_native_output after native=False expression path**
   - What we know: `_normalize_native_output` is only on the `native=True` branch — not affected
   - What's unclear: Whether `postprocess` needs any adjustment for ibis `nw.DataFrame` results
   - Recommendation: No change to `postprocess` — it already handles both `nw.LazyFrame` and
     `nw.DataFrame`. Verify `run_check`'s `_is_ibis_result` detection does not mis-fire.

3. **unique_values_eq check (not in narwhals builtins)**
   - What we know: `unique_values_eq` is defined in pandas/polars/ibis builtins but NOT in
     narwhals builtins. It's a set-level check (not row-level bool), so it cannot return a
     row-bool `nw.Expr`.
   - What's unclear: Is there a plan to add it to narwhals builtins?
   - Recommendation: Out of scope for Phase 5 — it's not in the narwhals builtins.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | pyproject.toml (pytest section) |
| Quick run command | `pytest tests/backends/narwhals/test_checks.py -x -q` |
| Full suite command | `pytest tests/backends/narwhals/ -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EXPR-01 | apply() native=False uses `frame.with_columns(expr.alias(...))` for all backends | unit | `pytest tests/backends/narwhals/test_checks.py::test_builtin_check_routing -x` | ✅ (needs update) |
| EXPR-02 | All 14 builtin checks pass on valid data (polars + ibis) | unit | `pytest tests/backends/narwhals/test_checks.py::test_builtin_checks_pass -x` | ✅ (should pass after changes) |
| EXPR-03 | All 14 builtin checks fail on invalid data (polars + ibis) | unit | `pytest tests/backends/narwhals/test_checks.py::test_builtin_checks_fail -x` | ✅ (should pass after changes) |
| EXPR-04 | element_wise=True still raises NotImplementedError on ibis | unit | `pytest tests/backends/narwhals/test_checks.py::test_element_wise_sql_lazy_raises -x` | ✅ |
| EXPR-05 | apply() returns wide table (data cols + CHECK_OUTPUT_KEY) for ibis | unit | `pytest tests/backends/narwhals/test_checks.py::test_apply_returns_wide_table -x` | ✅ |
| EXPR-06 | native=False check routing (after protocol change) | unit | `pytest tests/backends/narwhals/test_checks.py::test_native_false_user_check -x` | ✅ (needs update) |
| EXPR-07 | native=True checks unchanged | unit | `pytest tests/backends/narwhals/test_checks.py -k "native_true" -x` | ✅ |

### Sampling Rate
- **Per task commit:** `pytest tests/backends/narwhals/test_checks.py -x -q`
- **Per wave merge:** `pytest tests/backends/narwhals/ -q`
- **Phase gate:** Full narwhals test suite green before `/gsd:verify-work`

### Wave 0 Gaps
None — existing test infrastructure covers all phase requirements. The key test file
`tests/backends/narwhals/test_checks.py` already exists and covers all 14 builtins
plus routing and element_wise behavior. Some tests need content updates but no new
files are needed.

## Sources

### Primary (HIGH confidence)
- Experimental verification on narwhals 2.15.0 — `nw.col(key)` returns `nw.Expr`, `frame.with_columns(expr.alias(name))` works on both polars and ibis backends
- `pandera/backends/narwhals/checks.py` — full apply() implementation, row_number join, Dispatcher workaround
- `pandera/backends/narwhals/builtin_checks.py` — all 14 builtin function signatures (current)
- `pandera/api/function_dispatch.py` — Dispatcher uses first-arg type annotation for registry key
- `pandera/api/base/checks.py` — `from_builtin_check_name` sets `native=False`; `register_builtin_check` calls `get_first_arg_type(fn)`
- `tests/backends/narwhals/test_checks.py` — existing tests that will need routing test updates
- `pandera/backends/narwhals/base.py` — `run_check` `_is_ibis_result` detection logic

### Secondary (MEDIUM confidence)
- narwhals stable.v1 API docs — `nw.Expr` is the public expression type; `map_batches` is the batch-map method (not `map_elements`)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — narwhals 2.15.0 installed, all APIs verified experimentally
- Architecture: HIGH — full end-to-end tested: polars + ibis, expression + frame protocol
- Pitfalls: HIGH — most derived from reading actual source code; one (postprocess type) verified

**Research date:** 2026-03-22
**Valid until:** 2026-04-22 (stable narwhals; check if narwhals updates stable.v1 API)
