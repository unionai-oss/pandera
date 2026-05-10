# Phase 9: Accumulate Check Outputs Into Single Wide Table — Research

**Researched:** 2026-03-24
**Domain:** Narwhals backend check accumulation and drop_invalid_rows
**Confidence:** HIGH

## Summary

Phase 9 fixes 20 failing tests (`test_drop_invalid_rows`: 8 polars + 12 ibis) by
eliminating all backend-specific logic from `drop_invalid_rows`. The root cause is
that `check_output` stored on `SchemaError` is now a `nw.LazyFrame` (wide table
with `CHECK_OUTPUT_KEY` boolean column), but `drop_invalid_rows` tries to index it
with `co[CHECK_OUTPUT_KEY]` — a `TypeError` on `nw.LazyFrame`. The ibis path fails
even earlier by passing a `nw.LazyFrame` to `IbisSchemaBackend.drop_invalid_rows`
which expects a native `ibis.Table`.

The fix is architectural: change `apply()` to return a `nw.Expr` instead of a wide
table for `native=False` and `element_wise` checks. `drop_invalid_rows` then
collects all boolean expressions from `SchemaError.check_output`, builds a single
wide table via one `with_columns` call, filters with `nw.all_horizontal(...)`, and
drops the temporary columns. This pattern works identically for polars lazy frames
and ibis tables via narwhals — no `isinstance` guards, no `IbisSchemaBackend`
delegation, no positional joins.

The `postprocess_lazyframe_output` path still needs the wide table per check to
compute `failure_cases` (which rows failed THIS check) and the scalar `passed`
boolean. The solution is to add a `postprocess_expr_output` branch that builds the
wide table internally from the `nw.Expr` plus `check_obj.frame`, while storing the
`nw.Expr` itself as `check_output` on `CheckResult`. The `failure_cases_metadata`
eager-polars path must be updated to reconstruct the wide table from `err.data +
nw.Expr` when `check_output is nw.Expr`.

**Primary recommendation:** Return `nw.Expr` from `apply()` for `native=False` and
`element_wise` paths; accumulate exprs in `drop_invalid_rows` with single
`with_columns` + `nw.all_horizontal` filter; update `postprocess` and
`failure_cases_metadata` to handle `nw.Expr` `check_output`.

## Standard Stack

### Core APIs
| API | Purpose | Notes |
|-----|---------|-------|
| `nw.Expr` | Narwhals expression object | Both polars and ibis backends |
| `frame.with_columns([expr.alias(name), ...])` | Attach multiple bool cols in one call | Works on nw.LazyFrame and nw.DataFrame (ibis) |
| `nw.all_horizontal(*[nw.col(c) for c in bool_cols])` | Row-wise AND across bool columns | Cross-backend, works on ibis Table |
| `isinstance(check_output, nw.Expr)` | Detect expr-style check_output | Valid isinstance check on nw.Expr |
| `nw.from_native(check_obj, eager_or_interchange_only=False)` | Wrap native frame to nw | Handles pl.LazyFrame, ibis.Table |

**Verified via direct testing (2026-03-24):**

```python
# polars
import narwhals.stable.v1 as nw
import polars as pl
frame = nw.from_native(pl.LazyFrame({'a': [1,-1,2], 'b': ['x','y','z']}))
expr0 = nw.col('a') >= 0
expr1 = ~(nw.col('b') == 'z')
bool_cols = ['__check_output_0__', '__check_output_1__']
result = (
    frame
    .with_columns([expr0.alias(bool_cols[0]), expr1.alias(bool_cols[1])])
    .filter(nw.all_horizontal(*[nw.col(c) for c in bool_cols]))
    .drop(bool_cols)
)
# -> shape (1,2): row where a=1 and b='x' only

# ibis — identical code, different native backend
import ibis
ibis.set_backend('duckdb')
t = ibis.memtable({'a': [1,-1,2], 'b': ['x','y','z']})
frame = nw.from_native(t, eager_or_interchange_only=False)
# same expr0, expr1, bool_cols, result code — produces same filtered result
```

**`ignore_na` expr modification:**

```python
# if check.ignore_na: modify expr before storing
# current: check_output.with_columns(nw.col(COK) | nw.col(COK).is_null())
# new: expr_for_drop = expr | expr.is_null()
# This passes the row if the check passes OR if the result is null (original was null)
expr_with_ignore_na = expr | expr.is_null()
```

## Architecture Patterns

### Current Flow (broken)
```
apply()
  native=False: frame.with_columns(expr.alias(COK))  -> nw.LazyFrame (wide table)
  element_wise: frame.with_columns(map_batches.alias(COK)) -> nw.LazyFrame
  native=True:  native.mutate(COK=bool_col) -> nw.DataFrame (ibis wide table)

postprocess() -> postprocess_lazyframe_output(wide_table)
  CheckResult(check_output=wide_table)  # nw.LazyFrame

run_check() -> CoreCheckResult(check_output=wide_table)
SchemaError(check_output=wide_table)  # nw.LazyFrame

drop_invalid_rows():
  Polars: co[CHECK_OUTPUT_KEY]  # TypeError: Slicing not supported on LazyFrame
  Ibis: IbisSchemaBackend().drop_invalid_rows(native, ...) # IbisTypeError: nw.LazyFrame not ibis.Table
```

### Target Flow (Phase 9)
```
apply()
  native=False: return expr              # nw.Expr — NOT wide table
  element_wise: return map_batches_expr  # nw.Expr — NOT wide table
  native=True:  UNCHANGED (returns wide table or bool scalar)

postprocess()
  isinstance(check_output, nw.Expr): -> postprocess_expr_output(check_obj, expr)
    wide = frame.with_columns(expr.alias(COK))  # built internally
    failure_cases = wide.filter(~nw.col(COK))
    passed = wide.select(nw.col(COK).all())
    return CheckResult(check_output=expr, ...)  # store EXPR not wide table

run_check() -> CoreCheckResult(check_output=expr)  # nw.Expr
SchemaError(check_output=expr)  # nw.Expr

drop_invalid_rows(check_obj, error_handler):
  check_exprs = [(i, err.check_output) for i, err in enumerate(errors)
                 if isinstance(err.check_output, nw.Expr)]
  if not check_exprs: return check_obj
  frame = nw.from_native(check_obj, eager_or_interchange_only=False)
  bool_cols = [f'__check_output_{i}__' for i, _ in check_exprs]
  wide = frame.with_columns([
      expr.alias(f'__check_output_{i}__') for i, expr in check_exprs
  ])
  filtered = wide.filter(nw.all_horizontal(*[nw.col(c) for c in bool_cols]))
  result = nw.to_native(filtered.drop(bool_cols))
  return result
```

### File Change Map

| File | What Changes |
|------|-------------|
| `pandera/backends/narwhals/checks.py` | `apply()`: 3 paths; `postprocess()`: add `nw.Expr` branch; new `postprocess_expr_output()` |
| `pandera/backends/narwhals/base.py` | `drop_invalid_rows()`: full replacement (no ibis delegation); `failure_cases_metadata()`: handle `nw.Expr` check_output |

No changes needed to `container.py`, `components.py`, or `ibis/base.py`.

### Anti-Patterns to Avoid

- **Positional concat for boolean combination:** `pl.concat([s0, s1], how='horizontal')` — polars-specific, breaks ibis.
- **Indexing nw.LazyFrame:** `co[CHECK_OUTPUT_KEY]` — raises `TypeError: Slicing is not supported on LazyFrame`.
- **Delegating to IbisSchemaBackend:** `IbisSchemaBackend().drop_invalid_rows(native, ...)` — requires native `ibis.Table`, receives `nw.LazyFrame`.
- **Re-importing polars in drop_invalid_rows:** The new design eliminates the polars-specific merged_pl path entirely.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Row-wise boolean AND across multiple columns | Custom fold/reduce loop | `nw.all_horizontal(*cols)` | Cross-backend, stays lazy |
| Detecting bool expression type | `hasattr(x, '_name')` or similar | `isinstance(x, nw.Expr)` | Verified to work on nw.Expr instances |
| Attaching multiple bool columns | Multiple sequential `with_columns` calls | Single `with_columns([e0, e1, ...])` | One query plan node, works for ibis |
| Null handling in filter | Custom null-coalesce | `expr | expr.is_null()` for ignore_na | Pure narwhals, works cross-backend |

## Common Pitfalls

### Pitfall 1: native=True path not handled
**What goes wrong:** If `apply()` also returns `nw.Expr` for `native=True` (ibis `BooleanColumn`), `_normalize_native_output` can't produce a `nw.Expr` from `ir.BooleanColumn` — they're different types.
**How to avoid:** Leave `native=True` path UNCHANGED. It still returns wide table (ibis-wrapped `nw.LazyFrame` or native bool). `drop_invalid_rows` only processes `nw.Expr` entries; `native=True` entries are skipped (or handled separately if tests require it).
**Warning signs:** If `test_drop_invalid_rows` tests use user-defined `native=True` checks (they don't — all use built-in `native=False` checks).

### Pitfall 2: postprocess_expr_output building wide table for wrong frame
**What goes wrong:** `postprocess_expr_output` must use `check_obj.frame` (the narwhals LazyFrame), NOT `nw.to_native(check_obj.frame)`. Building `wide` from native then re-wrapping adds unnecessary round-trips.
**How to avoid:** `wide = check_obj.frame.with_columns(expr.alias(COK))` — `check_obj` is `NarwhalsData(frame=nw.LazyFrame, key=str)`.

### Pitfall 3: failure_cases_metadata losing row indices
**What goes wrong:** The eager polars path in `failure_cases_metadata` uses `err.check_output` with `CHECK_OUTPUT_KEY` to compute row indices. If `check_output` is now `nw.Expr`, must reconstruct wide table using `err.data`.
**How to avoid:**
```python
# In failure_cases_metadata eager polars path:
co = err.check_output
if isinstance(co, nw.Expr):
    # Reconstruct wide table from original frame
    data_frame = nw.from_native(err.data, eager_or_interchange_only=False)
    co = data_frame.with_columns(co.alias(CHECK_OUTPUT_KEY))
# then proceed: co_eager = _materialize(co), co_indexed = ...
```
**Warning signs:** Row indices showing as `None` in `failure_cases` output when they should be integers.

### Pitfall 4: ignore_na expr modification
**What goes wrong:** `postprocess_lazyframe_output` currently modifies the wide table AFTER apply returns it. If check_output is now `nw.Expr`, `ignore_na` must modify the expr BEFORE building the wide table.
**How to avoid:** In `postprocess_expr_output`:
```python
if self.check.ignore_na:
    expr = expr | expr.is_null()
wide = check_obj.frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))
```
The `expr` stored on `CheckResult.check_output` MUST be the `ignore_na`-modified version, so `drop_invalid_rows` applies the correct filter.

### Pitfall 5: drop_invalid_rows returning wrong frame type
**What goes wrong:** `container.py` passes a native `pl.LazyFrame` to `drop_invalid_rows`. `components.py` passes a `nw.LazyFrame`. The return type must match the input type.
**How to avoid:**
```python
def drop_invalid_rows(self, check_obj, error_handler):
    # ... accumulate exprs ...
    frame = nw.from_native(check_obj, eager_or_interchange_only=False)
    # ... build wide, filter, drop ...
    result = filtered.drop(bool_cols)
    # Return native if input was native, narwhals if input was narwhals
    if isinstance(check_obj, (nw.LazyFrame, nw.DataFrame)):
        return result
    return nw.to_native(result)
```

### Pitfall 6: n_failure_cases on postprocess_expr_output
**What goes wrong:** `postprocess_lazyframe_output` applies `head(n_failure_cases)` to `failure_cases`. `postprocess_expr_output` must also apply this.
**How to avoid:** Copy the `n_failure_cases` logic from `postprocess_lazyframe_output` into `postprocess_expr_output`.

## Code Examples

### apply() native=False — return Expr

```python
# Source: direct analysis of checks.py
# Before: return frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))
# After:
else:
    # native=False: expression protocol. Return expr, not wide table.
    if key and key != "*":
        expr = self.check_fn(nw.col(key))
    else:
        expr = self.check_fn(frame)
    return expr  # nw.Expr
```

### apply() element_wise — return Expr

```python
# Before: return frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))
# After:
if self.check.element_wise:
    selector = nw.col(key or "*")
    try:
        expr = selector.map_batches(self.check_fn, return_dtype=nw.Boolean)
        return expr  # nw.Expr
    except NotImplementedError:
        raise ...
```

### postprocess() dispatch

```python
def postprocess(self, check_obj: NarwhalsData, check_output):
    if isinstance(check_output, nw.Expr):
        return self.postprocess_expr_output(check_obj, check_output)
    elif isinstance(check_output, (nw.LazyFrame, nw.DataFrame)):
        return self.postprocess_lazyframe_output(check_obj, check_output)
    elif isinstance(check_output, bool):
        return self.postprocess_bool_output(check_obj, check_output)
    raise TypeError(...)
```

### postprocess_expr_output

```python
def postprocess_expr_output(self, check_obj: NarwhalsData, expr: nw.Expr) -> CheckResult:
    """Postprocesses nw.Expr check output into a CheckResult."""
    # Modify for ignore_na BEFORE building wide table (important for drop_invalid_rows)
    if self.check.ignore_na:
        expr = expr | expr.is_null()
    # Build wide table for failure_cases computation
    frame = check_obj.frame
    wide = frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))
    passed = wide.select(nw.col(CHECK_OUTPUT_KEY).all())
    failure_cases = wide.filter(~nw.col(CHECK_OUTPUT_KEY))
    if check_obj.key != "*":
        failure_cases = failure_cases.select(check_obj.key)
    if self.check.n_failure_cases is not None:
        failure_cases = failure_cases.head(self.check.n_failure_cases)
    return CheckResult(
        check_output=expr,   # store the (possibly ignore_na-modified) EXPR
        check_passed=passed,
        checked_object=check_obj,
        failure_cases=failure_cases,
    )
```

### drop_invalid_rows — new narwhals-idiomatic implementation

```python
def drop_invalid_rows(self, check_obj, error_handler):
    """Remove invalid rows — pure narwhals, no backend delegation."""
    errors = getattr(error_handler, "schema_errors", [])
    if not errors:
        return check_obj

    # Collect nw.Expr check_outputs only (native=True and bool paths are skipped)
    check_exprs = [
        (i, err.check_output)
        for i, err in enumerate(errors)
        if isinstance(err.check_output, nw.Expr)
    ]
    if not check_exprs:
        return check_obj

    frame = nw.from_native(check_obj, eager_or_interchange_only=False)
    bool_cols = [f"__check_output_{i}__" for i, _ in check_exprs]

    wide = frame.with_columns([
        expr.alias(f"__check_output_{i}__")
        for i, expr in check_exprs
    ])
    filtered = wide.filter(nw.all_horizontal(*[nw.col(c) for c in bool_cols]))
    result = filtered.drop(bool_cols)

    # Preserve input type: native in -> native out
    if isinstance(check_obj, (nw.LazyFrame, nw.DataFrame)):
        return result
    return nw.to_native(result)
```

### failure_cases_metadata — handle nw.Expr check_output

```python
# In the eager polars path (failure_cases_metadata):
if err.check_output is not None:
    co = err.check_output
    if isinstance(co, nw.Expr):
        # Reconstruct wide table for row index computation
        data_frame = nw.from_native(err.data, eager_or_interchange_only=False)
        co = data_frame.with_columns(co.alias(CHECK_OUTPUT_KEY))
    if not isinstance(co, (nw.LazyFrame, nw.DataFrame)):
        co = nw.from_native(co, eager_or_interchange_only=False)
    co_eager = _materialize(co)
    # ... rest unchanged ...
```

## State of the Art

| Old Approach | Current (Phase 9) Approach | Why Changed |
|--------------|---------------------------|-------------|
| `apply()` returns wide LazyFrame | `apply()` returns `nw.Expr` | Enables accumulation without cross-frame ops |
| `drop_invalid_rows` extracts bool col with `co[COK]` | `drop_invalid_rows` re-evaluates expr on original frame | `co[COK]` crashes on nw.LazyFrame |
| `IbisSchemaBackend.drop_invalid_rows` via delegation | Narwhals `all_horizontal` filter | Delegation required native ibis.Table, received nw.LazyFrame |
| `postprocess_lazyframe_output` handles wide table | `postprocess_expr_output` handles nw.Expr | apply() no longer produces wide table |

**Deprecated by this phase:**
- `IbisSchemaBackend` delegation in `drop_invalid_rows` — replaced by `nw.all_horizontal` approach
- `import polars as pl` in `drop_invalid_rows` — no longer needed
- `import ibis as _ibis` in `drop_invalid_rows` — no longer needed

## Open Questions

1. **native=True check_output in drop_invalid_rows**
   - What we know: `native=True` checks return wide tables (ibis-wrapped), not `nw.Expr`. Current failing tests use only `native=False` built-in checks.
   - What's unclear: Should `drop_invalid_rows` also support `native=True` check_outputs?
   - Recommendation: Skip `native=True` entries in `drop_invalid_rows` (only process `isinstance(err.check_output, nw.Expr)`). Document as limitation. The failing tests will pass without native=True support.

2. **failure_cases_metadata row indices for lazy path**
   - What we know: The lazy/ibis path in `failure_cases_metadata` already doesn't use `check_output` for row indices (it uses `None`).
   - What's unclear: Does the eager polars path still need row indices when `check_output` is `nw.Expr`?
   - Recommendation: Yes — reconstruct wide table from `err.data + nw.Expr`. `err.data` is always a narwhals-wrapped frame at this point.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | pyproject.toml (`[tool.pytest.ini_options]`) |
| Quick run command | `python -m pytest tests/polars/test_polars_container.py::test_drop_invalid_rows tests/polars/test_polars_container.py::test_drop_invalid_rows_nullable -x -q` |
| Full suite command | `python -m pytest tests/polars/test_polars_container.py tests/ibis/test_ibis_container.py tests/backends/narwhals/ -q` |

### Phase Requirements → Test Map

| ID | Behavior | Test Type | Automated Command | File Exists? |
|----|----------|-----------|-------------------|-------------|
| DIR-01 | polars lazy drop_invalid_rows filters correctly | integration | `python -m pytest tests/polars/test_polars_container.py::test_drop_invalid_rows -k "lazy"` | Yes |
| DIR-02 | polars nullable drop_invalid_rows + ignore_na | integration | `python -m pytest tests/polars/test_polars_container.py::test_drop_invalid_rows_nullable` | Yes |
| DIR-03 | ibis duckdb drop_invalid_rows filters correctly | integration | `python -m pytest tests/ibis/test_ibis_container.py::test_drop_invalid_rows -k "duckdb"` | Yes |
| DIR-04 | ibis sqlite drop_invalid_rows filters correctly | integration | `python -m pytest tests/ibis/test_ibis_container.py::test_drop_invalid_rows -k "sqlite"` | Yes |
| DIR-05 | No regression in narwhals backend suite | integration | `python -m pytest tests/backends/narwhals/ -q` | Yes |
| DIR-06 | No regression in polars container suite | integration | `python -m pytest tests/polars/test_polars_container.py -q` | Yes |
| DIR-07 | No regression in ibis container suite | integration | `python -m pytest tests/ibis/test_ibis_container.py -q` | Yes |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/polars/test_polars_container.py::test_drop_invalid_rows tests/polars/test_polars_container.py::test_drop_invalid_rows_nullable tests/ibis/test_ibis_container.py::test_drop_invalid_rows -q`
- **Per wave merge:** `python -m pytest tests/polars/test_polars_container.py tests/ibis/test_ibis_container.py tests/backends/narwhals/ -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
None — existing test infrastructure covers all phase requirements.

## Sources

### Primary (HIGH confidence)
- Direct code analysis: `pandera/backends/narwhals/checks.py`, `base.py`, `container.py`, `components.py`, `ibis/base.py` — all read and traced
- Direct runtime testing: polars and ibis narwhals approaches verified via Python interpreter (2026-03-24)
- Test run: `python -m pytest tests/polars/test_polars_container.py::test_drop_invalid_rows` — 8 failures confirmed
- Test run: `python -m pytest tests/ibis/test_ibis_container.py::test_drop_invalid_rows` — 12 failures confirmed
- Test run: `python -m pytest tests/backends/narwhals/` — 208 passed, 0 failures (baseline green)

### Secondary (MEDIUM confidence)
- narwhals stable v1 API: `nw.all_horizontal`, `nw.Expr`, `isinstance(x, nw.Expr)` — confirmed via interpreter

## Metadata

**Confidence breakdown:**
- Root cause diagnosis: HIGH — confirmed by traceback analysis and test runs
- Target design: HIGH — all critical paths tested via interpreter
- Implementation scope: HIGH — files identified, no surprise dependencies found
- Pitfalls: HIGH — discovered through systematic trace of the call chain
- ignore_na behavior: HIGH — tested directly

**Research date:** 2026-03-24
**Valid until:** Until checks.py or base.py is structurally refactored (stable — these files have been steady since Phase 5)
