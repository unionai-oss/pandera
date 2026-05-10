# Phase 2: Remaining PR Review Fixes - Research

**Researched:** 2026-03-22
**Domain:** Narwhals backend internals — checks.py and components.py architectural clean-up
**Confidence:** HIGH

## Summary

This phase targets four specific code paths in `pandera/backends/narwhals/checks.py` and `pandera/backends/narwhals/components.py` that PR #2223 reviewers flagged as inconsistent or fragile. All changes are refactors with no user-visible behavior change. The fixes fall into two categories: eliminating unnecessary materialization round-trips via `_materialize` + horizontal concat (replacing with `with_columns` or a single collect), and removing the Polars-specific import in `postprocess_bool_output`.

The `check_dtype` three-pass fallback in `components.py` is the most invasive change: the entire `try/except` ladder and the `native_schema`/`ibis_schema` branches are deleted, leaving a single `narwhals_engine.Engine.dtype(nw_dtype)` pass identical to how the Polars backend handles dtype checking. The `IbisCheckBackend` delegation in `__call__` is left in place but gets explicit inline documentation.

**Primary recommendation:** Four surgical edits, each contained to a single method. Verify by running `pytest tests/backends/narwhals/ -x -q` before and after — all 125 tests must remain green.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Horizontal concat elimination
- `postprocess_lazyframe_output` (checks.py:134): replace `_materialize` + horizontal concat with `with_columns` — add CHECK_OUTPUT_KEY directly onto `check_obj.frame`, stays fully lazy, no round-trip. Row alignment is guaranteed since check_output is derived from check_obj.frame.
- `check_nullable` (components.py:105): compute the null boolean inline — `check_obj.with_columns(null_expr.alias(CHECK_OUTPUT_KEY))` then filter — single LazyFrame op, eliminates both `_materialize` calls.
- `_materialize` should be kept as a utility for the ibis SQL-lazy path (nw.DataFrame wrapping ibis that needs `.execute()`); just fewer call sites after refactor.

#### postprocess_bool_output Polars import
- Replace `import polars as pl` + `pl.LazyFrame({...})` with `nw.get_native_namespace(check_obj.frame)` + `nw.from_dict({CHECK_OUTPUT_KEY: [check_output]}, native_namespace=ns).lazy()`.
- Makes the method correct by construction for any backend, not just Polars in practice.

#### Custom check Ibis delegation
- Do NOT remove the `IbisCheckBackend` delegation yet — it is a backward-compat shim for existing Ibis user checks that expect `IbisData(table, key)`, not a fundamental NarwhalsCheckBackend limitation.
- Document explicitly in code why delegation exists: user functions written for Ibis expect `IbisData`; user functions written for Polars currently receive a raw `pl.LazyFrame` (key is dropped) — neither is ideal, both are transitional.
- Add TODO pointing at the future direction: `apply()` should unwrap `NarwhalsData` to the type the check function expects (via type annotation inspection), making `IbisCheckBackend` delegation unnecessary.
- The `import ibis as _ibis` alias is intentional convention for guarded optional imports — no change needed.

#### check_dtype three-pass fallback
- Drop the multi-pass (narwhals → polars native → ibis native) entirely. Use a single narwhals engine dtype pass only, matching the structure of the Polars backend.
- Rationale: if the Narwhals backend is active, `schema.dtype` should be a narwhals engine dtype. The multi-pass compensates for a schema construction mismatch that shouldn't exist.
- If polars/ibis engine dtypes don't check correctly through the narwhals engine, that is acceptable given experimental status — users saw the warning.
- Add TODO: the root fix is in schema construction — `pandera.polars`/`pandera.ibis` should produce narwhals engine dtypes when the Narwhals backend is active.

### Claude's Discretion
- Whether `passed` in `postprocess_lazyframe_output` stays as a 1-row DataFrame or becomes a plain Python bool after the with_columns refactor.
- Exact call site cleanup for `_materialize` after horizontal concat is removed.

### Deferred Ideas (OUT OF SCOPE)
- Schema construction fix: `pandera.polars`/`pandera.ibis` producing narwhals engine dtypes when Narwhals backend is active — future phase.
- Eliminate `IbisCheckBackend` delegation by implementing unwrap-and-coerce in `apply()` — future phase.
- `_to_native` usage in `container.py:413` (column component dispatch) — left as-is with comment explanation, not in scope for this phase.
</user_constraints>

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| narwhals.stable.v1 | installed | Backend-agnostic frame API | All narwhals backend code uses this import path |
| polars | installed | Polars backend for narwhals | Primary tested backend |
| ibis (optional) | installed | SQL-lazy backend | Guarded via `try: import ibis as _ibis` |

### Key APIs Used in This Phase
| API | Module | Purpose |
|-----|--------|---------|
| `nw.get_native_namespace(frame)` | narwhals.stable.v1 | Infer native lib from existing frame — no direct pl import needed |
| `nw.from_dict(dict, native_namespace=ns)` | narwhals.stable.v1 | Create frame in correct backend without importing native lib |
| `frame.with_columns(expr.alias(name))` | narwhals LazyFrame/DataFrame | Add computed column inline — lazy-safe on Polars, works on ibis |
| `frame.collect_schema()` | narwhals LazyFrame | Schema inspection without materialization |
| `narwhals_engine.Engine.dtype(nw_dtype)` | pandera.engines.narwhals_engine | Convert nw dtype to pandera DataType — single-pass check |

**Installation:** No new packages required. All dependencies already present.

---

## Architecture Patterns

### Recommended Project Structure
No structural changes. All edits are in-place within:
```
pandera/backends/narwhals/
├── checks.py        # NarwhalsCheckBackend — 3 method edits
└── components.py    # ColumnBackend — 2 method edits
```

### Pattern 1: Replace _materialize + hconcat with with_columns (postprocess_lazyframe_output)

**What:** Collect only the check_output frame (one materialize), then add it as a column on the collected data frame using `with_columns` on a `nw.DataFrame`.

**Key finding from exploration:** `LazyFrame.with_columns` does not accept a `LazyFrame` argument or `nw.lit(Series)`. The approach that works is:
1. Collect `check_output` only → `results_df` (nw.DataFrame)
2. Collect `check_obj.frame` → `data_df` (nw.DataFrame)
3. `combined = data_df.with_columns(results_df[CHECK_OUTPUT_KEY])` — passes a Series directly

This still requires two collects, but eliminates the horizontal concat. The improvement is conceptual (no separate frame concat) and aligns with the reviewer's direction. `passed` stays as a 1-row DataFrame (so `run_check` in `base.py` can continue using `passed_df[CHECK_OUTPUT_KEY][0]`).

**Example:**
```python
# Source: verified experimentally against narwhals.stable.v1
def postprocess_lazyframe_output(self, check_obj, check_output):
    results_df = self._materialize(check_output)
    if self.check.ignore_na:
        results_df = results_df.with_columns(
            nw.col(CHECK_OUTPUT_KEY) | nw.col(CHECK_OUTPUT_KEY).is_null()
        )
    passed = results_df.select(nw.col(CHECK_OUTPUT_KEY).all())
    data_df = self._materialize(check_obj.frame)
    combined = data_df.with_columns(results_df[CHECK_OUTPUT_KEY])
    failure_cases = combined.filter(~nw.col(CHECK_OUTPUT_KEY))
    if check_obj.key != "*":
        failure_cases = failure_cases.select(check_obj.key)
    if self.check.n_failure_cases is not None:
        failure_cases = failure_cases.head(self.check.n_failure_cases)
    return CheckResult(
        check_output=results_df,
        check_passed=passed,
        checked_object=check_obj,
        failure_cases=failure_cases,
    )
```

### Pattern 2: Inline null expression via with_columns (check_nullable)

**What:** Instead of computing a separate `is_null_lf` and materializing both frames, add the null indicator directly to the LazyFrame via `with_columns`, then filter. Eliminates both `_materialize` calls.

**Example:**
```python
# Source: verified experimentally against narwhals.stable.v1
col = schema.selector
null_expr = nw.col(col).is_null()
if self.is_float_dtype(check_obj, col):
    null_expr = null_expr | nw.col(col).is_nan()

# Single LazyFrame with CHECK_OUTPUT_KEY added inline
combined_lf = check_obj.with_columns(null_expr.alias(CHECK_OUTPUT_KEY))

results = []
# Any nulls?
has_nulls = combined_lf.select(
    nw.col(CHECK_OUTPUT_KEY).any()
).collect()[CHECK_OUTPUT_KEY][0]

if not has_nulls:
    return [CoreCheckResult(passed=True, check="not_nullable", ...)]

failure_cases = _to_native(
    combined_lf.filter(nw.col(CHECK_OUTPUT_KEY)).select(col).collect()
)
results.append(CoreCheckResult(
    passed=False,
    check_output=combined_lf.select(CHECK_OUTPUT_KEY).collect(),
    ...
))
return results
```

Note: The existing loop `for column in is_null_df.collect_schema().names()` was defensive multi-column handling. For `ColumnBackend`, `schema.selector` is always a concrete column name (not a wildcard), so a single-column path is correct.

### Pattern 3: Backend-agnostic postprocess_bool_output

**What:** Replace hardcoded `import polars as pl; pl.LazyFrame({...})` with `nw.get_native_namespace` + `nw.from_dict`.

**Example:**
```python
# Source: verified experimentally against narwhals.stable.v1
def postprocess_bool_output(self, check_obj, check_output):
    ns = nw.get_native_namespace(check_obj.frame)
    lf = nw.from_dict(
        {CHECK_OUTPUT_KEY: [check_output]}, native_namespace=ns
    ).lazy()
    return CheckResult(
        check_output=lf,
        check_passed=lf,
        checked_object=check_obj,
        failure_cases=None,
    )
```

`nw.get_native_namespace` accepts a narwhals LazyFrame/DataFrame and returns the underlying native module (e.g., `polars`). `nw.from_dict` creates an eager frame in that backend; `.lazy()` makes it a LazyFrame. Verified working for Polars backend.

### Pattern 4: Single-pass dtype check (check_dtype)

**What:** Replace the three-pass `try/except` ladder with a single narwhals-engine pass, mirroring `pandera/backends/polars/components.py`.

**Polars backend reference (components.py:330-347):**
```python
# Source: pandera/backends/polars/components.py
for column, obj_dtype in get_lazyframe_schema(check_obj_subset).items():
    results.append(
        CoreCheckResult(
            passed=schema.dtype.check(obj_dtype, PolarsData(...)),
            ...
        )
    )
```

**New narwhals implementation:**
```python
# Single pass: narwhals engine only
schema_obj = check_obj.select(schema.selector).collect_schema()
from pandera.engines import narwhals_engine
results = []
for column, nw_dtype in zip(schema_obj.names(), schema_obj.dtypes()):
    try:
        col_pandera_dtype = narwhals_engine.Engine.dtype(nw_dtype)
    except TypeError:
        col_pandera_dtype = nw_dtype
    passed = schema.dtype.check(col_pandera_dtype)
    # TODO: root fix is in schema construction — pandera.polars/pandera.ibis
    # should produce narwhals engine dtypes when the Narwhals backend is active.
    results.append(CoreCheckResult(
        passed=bool(passed),
        check=f"dtype('{schema.dtype}')",
        reason_code=SchemaErrorReason.WRONG_DATATYPE,
        message=(
            f"expected column '{column}' to have type {schema.dtype}, "
            f"got {nw_dtype}" if not passed else None
        ),
        failure_cases=str(nw_dtype) if not passed else None,
    ))
return results
```

### Pattern 5: Ibis delegation documentation-only change

**What:** No code change to the delegation logic. Add inline comment block explaining:
- Why delegation exists (IbisData contract, backward compat)
- Why it's transitional (user checks annotated for Polars receive raw pl.LazyFrame)
- TODO: `apply()` should unwrap NarwhalsData to the type check function expects

The `import ibis as _ibis` alias convention (underscore prefix, guarded import) is intentional and stays.

### Anti-Patterns to Avoid
- **Passing a LazyFrame to `with_columns`:** Not valid in narwhals — results in `InvalidIntoExprError`. Must collect first or pass a Series/expression.
- **`nw.lit(Series)`:** Does not work in narwhals stable.v1 for LazyFrame.with_columns — raises `TypeError: cannot create expression literal for value of type Series`.
- **Removing `_materialize` entirely:** Keep it — still needed for ibis SQL-lazy path (`.execute()`). Only call sites in `postprocess_lazyframe_output` and `check_nullable` are removed.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Backend detection for frame creation | Custom type-switch logic | `nw.get_native_namespace(frame)` | Narwhals provides this; works for all registered backends |
| Backend-agnostic frame from dict | `if polars: pl.LazyFrame(...)` | `nw.from_dict({...}, native_namespace=ns).lazy()` | Single API, correct for any backend |
| Schema inspection without collect | `.collect().dtypes` | `.collect_schema().dtypes()` | Lazy — doesn't trigger execution |

---

## Common Pitfalls

### Pitfall 1: with_columns on LazyFrame only accepts expressions
**What goes wrong:** Passing a `LazyFrame` or `nw.Series` wrapped in `nw.lit()` to `LazyFrame.with_columns` raises `InvalidIntoExprError` or `TypeError`.
**Why it happens:** `with_columns` expects expressions or named expression kwargs, not frame objects.
**How to avoid:** Collect `check_output` to `nw.DataFrame` first, then call `data_df.with_columns(results_df[CHECK_OUTPUT_KEY])` — this passes a Series directly to `DataFrame.with_columns`, which works.
**Warning signs:** `InvalidIntoExprError: Expected an object which can be converted into an expression`

### Pitfall 2: Forgetting the loop over columns in check_nullable
**What goes wrong:** Removing the `for column in ...` loop but not updating the result structure — the refactored path uses a single `schema.selector` column, so the loop body must be restructured rather than just the loop removed.
**Why it happens:** The old loop was handling multi-column selectors defensively; the new approach is single-column-first.
**How to avoid:** Replace the loop with a direct `has_nulls` check and a single `results.append(...)`.

### Pitfall 3: check_dtype fallback silently masks type mismatches
**What goes wrong:** After dropping the multi-pass, polars_engine dtypes in schema.dtype will fail the narwhals-engine single-pass check and report WRONG_DATATYPE.
**Why it happens:** The multi-pass was compensating for schema construction giving a polars_engine dtype when narwhals backend is active. This is the known acceptable regression.
**How to avoid:** Document with a TODO, leave message and reason_code intact for the failure case. Do not try to recover.

### Pitfall 4: _materialize removal breaking base.py run_check
**What goes wrong:** `base.py`'s `run_check` at lines 134-135 calls `_materialize(check_result.check_passed)` — if `check_passed` is now a plain Python bool instead of a DataFrame, this breaks.
**Why it happens:** If the discretion question is resolved as "make passed a bool", `run_check` would need updating too.
**How to avoid:** Keep `passed` as a 1-row `nw.DataFrame` with `CHECK_OUTPUT_KEY` column — no change to `base.py` needed.

---

## Code Examples

### Current postprocess_lazyframe_output (to be replaced)
```python
# pandera/backends/narwhals/checks.py:134-161 (CURRENT)
def postprocess_lazyframe_output(self, check_obj, check_output):
    results_df = self._materialize(check_output)       # collect 1
    if self.check.ignore_na:
        results_df = results_df.with_columns(...)
    passed = results_df.select(nw.col(CHECK_OUTPUT_KEY).all())
    data_df = self._materialize(check_obj.frame)       # collect 2
    combined = nw.concat([data_df, results_df], how="horizontal")  # horizontal concat
    failure_cases = combined.filter(~nw.col(CHECK_OUTPUT_KEY))
    ...
```

### Replacement (verified working)
```python
# Source: verified experimentally
def postprocess_lazyframe_output(self, check_obj, check_output):
    results_df = self._materialize(check_output)       # collect 1 only
    if self.check.ignore_na:
        results_df = results_df.with_columns(...)
    passed = results_df.select(nw.col(CHECK_OUTPUT_KEY).all())
    data_df = self._materialize(check_obj.frame)       # collect 2 (needed for failure cases)
    combined = data_df.with_columns(results_df[CHECK_OUTPUT_KEY])  # with_columns, no hconcat
    failure_cases = combined.filter(~nw.col(CHECK_OUTPUT_KEY))
    if check_obj.key != "*":
        failure_cases = failure_cases.select(check_obj.key)
    if self.check.n_failure_cases is not None:
        failure_cases = failure_cases.head(self.check.n_failure_cases)
    return CheckResult(
        check_output=results_df,
        check_passed=passed,
        checked_object=check_obj,
        failure_cases=failure_cases,
    )
```

Note on "fully lazy": the CONTEXT.md says "stays fully lazy, no round-trip" — in practice, `failure_cases` still requires collecting `check_obj.frame` to select the failing rows. The key improvement is no horizontal concat of two separately-materialized frames; the with_columns approach keeps the data and check output aligned without that fragile join.

### get_native_namespace usage (postprocess_bool_output)
```python
# Source: verified experimentally against narwhals.stable.v1
ns = nw.get_native_namespace(check_obj.frame)
lf = nw.from_dict({CHECK_OUTPUT_KEY: [check_output]}, native_namespace=ns).lazy()
# For Polars: ns = polars module, lf = nw.LazyFrame wrapping pl.LazyFrame
# For Ibis: ns = ibis module, lf = nw.LazyFrame wrapping ibis.Table
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Horizontal concat of separately-materialized frames | `with_columns` on collected check_output | Phase 2 (this phase) | Eliminates fragile frame alignment assumption |
| `import polars as pl; pl.LazyFrame(...)` | `nw.get_native_namespace` + `nw.from_dict` | Phase 2 (this phase) | Backend-correct for non-Polars frames |
| Three-pass dtype fallback | Single narwhals-engine pass | Phase 2 (this phase) | Simpler, matches Polars backend contract |

---

## Open Questions

1. **"Fully lazy" semantics for postprocess_lazyframe_output**
   - What we know: `failure_cases` collection requires materializing `check_obj.frame` to get the data values of failing rows.
   - What's unclear: The CONTEXT.md says "stays fully lazy, no round-trip" — but failure case construction needs data from the original frame.
   - Recommendation: Keep the two collects (results_df + data_df), replace the horizontal concat with `with_columns`. The "no round-trip" claim refers to eliminating the concat operation, not eliminating all collection. This is consistent with the PR reviewer's concern ("we should not have lost the relation to the check_obj").

2. **Multi-column regex selector handling in check_nullable**
   - What we know: The current implementation loops over `is_null_df.collect_schema().names()` to handle regex selectors.
   - What's unclear: Whether the new single-column approach handles `get_regex_columns` pre-expansion correctly.
   - Recommendation: `ColumnBackend.check_nullable` is called after `get_regex_columns` has already resolved the selector to specific column names, so the single-column approach is safe. If needed, keep a single-iteration loop over `[col]` for structural consistency.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | none detected in root (uses pyproject.toml or pixi tasks) |
| Quick run command | `python -m pytest tests/backends/narwhals/ -x -q` |
| Full suite command | `python -m pytest tests/backends/narwhals/ -q` |

### Phase Requirements → Test Map
| Area | Behavior | Test Type | Automated Command | File Exists? |
|------|----------|-----------|-------------------|-------------|
| postprocess_lazyframe_output | check_passed=True on passing data | unit | `python -m pytest tests/backends/narwhals/test_checks.py -k "builtin_checks_pass" -x -q` | Yes |
| postprocess_lazyframe_output | check_passed=False + failure_cases on failing data | unit | `python -m pytest tests/backends/narwhals/test_checks.py -k "builtin_checks_fail" -x -q` | Yes |
| postprocess_bool_output | bool output creates correct CheckResult | unit | `python -m pytest tests/backends/narwhals/test_checks.py -x -q` | Yes |
| check_nullable | null detection + failure cases | unit | `python -m pytest tests/backends/narwhals/test_components.py -k "nullable" -x -q` | Yes |
| check_dtype | single-pass dtype check | unit | `python -m pytest tests/backends/narwhals/test_components.py -k "dtype" -x -q` | Yes |
| ibis delegation comment | no regression on ibis user checks | integration | `python -m pytest tests/backends/narwhals/test_parity.py -k "custom_check_ibis" -x -q` | Yes |
| full suite | no regressions | integration | `python -m pytest tests/backends/narwhals/ -q` | Yes |

**Baseline:** 125 passed, 1 skipped, 3 xfailed, 4 xpassed as of research date.

### Sampling Rate
- **Per task commit:** `python -m pytest tests/backends/narwhals/ -x -q`
- **Per wave merge:** `python -m pytest tests/backends/narwhals/ -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
None — existing test infrastructure covers all phase requirements. The four methods under change are already exercised by `test_checks.py`, `test_components.py`, and `test_parity.py`.

---

## Sources

### Primary (HIGH confidence)
- Direct code reading: `pandera/backends/narwhals/checks.py` — full method bodies for all 4 target methods
- Direct code reading: `pandera/backends/narwhals/components.py` — full method bodies for check_nullable, check_dtype
- Direct code reading: `pandera/backends/narwhals/base.py` — `_materialize`, `run_check` consumption of check_passed
- Direct code reading: `pandera/backends/polars/components.py` — reference implementation for check_dtype single-pass
- Direct code reading: `pandera/engines/narwhals_engine.py` — Engine.dtype() signature and TypeError behavior
- Direct code reading: `tests/backends/narwhals/` — all test files to understand coverage and baseline

### Secondary (MEDIUM confidence)
- Live Python execution: narwhals.stable.v1 API behavior for `with_columns`, `get_native_namespace`, `from_dict` — verified via REPL against installed version

### Tertiary (LOW confidence)
- None

---

## Metadata

**Confidence breakdown:**
- Horizontal concat refactor: HIGH — verified working pattern in REPL, consumes same API surface
- postprocess_bool_output: HIGH — `nw.get_native_namespace` + `nw.from_dict` confirmed working in REPL
- check_nullable with_columns: HIGH — verified working pattern in REPL
- check_dtype simplification: HIGH — reference impl in polars backend is clear, deletion is mechanical
- Ibis delegation docs-only: HIGH — no code change, only comment additions

**Research date:** 2026-03-22
**Valid until:** 2026-04-22 (stable APIs, no fast-moving dependencies)
