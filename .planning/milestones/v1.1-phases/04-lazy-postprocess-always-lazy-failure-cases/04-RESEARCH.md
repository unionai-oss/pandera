# Phase 4: Lazy postprocess — always-lazy failure_cases - Research

**Researched:** 2026-03-22
**Domain:** NarwhalsCheckBackend internals — apply(), postprocess_lazyframe_output(), run_check() ibis path
**Confidence:** HIGH (all findings from direct codebase inspection)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Architecture: Wide Table Approach** — `apply()` returns `check_obj.frame.with_columns(bool_output)` (full frame + CHECK_OUTPUT_KEY column), NOT just the boolean column alone. For Polars: `nw.LazyFrame` with column appended. For ibis: `nw.DataFrame` wrapping `ibis.Table` with column appended.
- **postprocess_lazyframe_output: No Materialization** — must NOT call `_materialize(check_obj.frame)`. Builds `passed` and `failure_cases` lazily. `passed = check_output.select(nw.col(CHECK_OUTPUT_KEY).all())`, `failure_cases = check_output.filter(~nw.col(CHECK_OUTPUT_KEY))`.
- **Materialization Point: run_check Only** — `_materialize` only called in `run_check` when evaluating the scalar `passed` boolean and when extracting `failure_cases`.
- **failure_cases Type for Ibis** — for ibis builtin checks, `failure_cases` is a narwhals-wrapped lazy ibis Table. NOT `pyarrow.Table`, NOT `pl.DataFrame`.
- **Return Type of __call__** — returns whatever type the input frame is. No type normalization.
- **Do NOT Scale Back** — no simple pyarrow→polars conversion fix. Architectural change is required.

### Claude's Discretion
- Whether `_materialize` helper can be simplified or removed after the change
- How to handle `ignore_na` in the new lazy postprocess path
- Whether `postprocess_bool_output` needs changes for consistency
- Test structure for the new behavior

### Deferred Ideas (OUT OF SCOPE)
- Removing `_is_ibis_result` guard entirely (custom ibis checks via `native=True` may still produce different result shapes — investigate separately)
- Making `postprocess_bool_output` lazy (currently produces a Polars LazyFrame for bool scalar results even for ibis — separate concern)
- `subsample()` in `base.py` also calls `_materialize` — out of scope for this phase
</user_constraints>

---

## Summary

Phase 4 makes `postprocess_lazyframe_output` fully lazy by changing what `apply()` returns (wide table = frame + CHECK_OUTPUT_KEY) and removing the two `_materialize` calls from `postprocess_lazyframe_output`. The root cause of the `pyarrow.Table` bug is clear: `_materialize(check_obj.frame)` calls `ibis.Table.execute()` → pandas → `nw.from_native(pd_df)`, and then `_to_native()` on that narwhals-over-pandas frame returns a `pyarrow.Table`.

The architecture after Phase 4: `apply()` returns the full frame with CHECK_OUTPUT_KEY column appended (lazy). `postprocess_lazyframe_output` filters and aggregates lazily. `run_check` already has two paths — the narwhals (Polars) path and the ibis path. After this phase, ibis builtin checks will produce a `nw.DataFrame` wrapping an `ibis.Table` as `failure_cases`, flowing through the **ibis path** in `run_check`, not the narwhals path.

**Primary recommendation:** The change is surgical. Three mutations: (1) `apply()` — return wide table instead of column-only; (2) `postprocess_lazyframe_output` — remove materializations, build lazily; (3) `run_check` — expand the `_is_ibis_result` guard to detect narwhals-wrapped ibis frames.

---

## Current Code Trace: apply() Return Paths

### apply() — Current behavior (checks.py lines 41-98)

There are three branches plus a bool short-circuit:

**Branch 1: element_wise**
- Does `frame.with_columns(selector.map_batches(...))` then `.select(selector)`
- Returns ONLY the selected column(s) — **not the wide table**
- After Phase 4: should remain as-is (element_wise raises on SQL-lazy, so ibis is not affected)

**Branch 2: native=True (custom user checks)**
- Unwraps to native, calls `check_fn(native_frame, key)`, normalizes via `_normalize_native_output`
- `_normalize_native_output` returns: `bool` (BooleanScalar path), `nw.DataFrame` wrapping a 1-column ibis Table (BooleanColumn path), or `nw.DataFrame` wrapping full ibis Table, or Polars frame passthrough
- Current output is a 1-column frame (the check output only), NOT the wide table

**Branch 3: native=False (builtin checks)**
- Calls `check_fn(frame, key)` which returns a narwhals frame with check booleans
- `out` is a column-only frame (e.g., just column `x` renamed to CHECK_OUTPUT_KEY)

**Post-branch logic (lines 84-98):**
- If `out` is `bool`: returns early (no wide table change needed here)
- Otherwise: renames/reduces multi-column output to CHECK_OUTPUT_KEY, then returns `out`
- Currently returns ONLY the CHECK_OUTPUT_KEY column frame

**After Phase 4 change:** After the rename/reduce to CHECK_OUTPUT_KEY, instead of returning `out`, return `check_obj.frame.with_columns(out)` (or `check_obj.frame.with_columns(nw.col(CHECK_OUTPUT_KEY))` from `out`).

### Wide table with_columns edge cases

`check_obj.frame.with_columns(bool_col)` where `bool_col` is a 1-column narwhals frame (not a Series expression):
- The narwhals `.with_columns()` accepts expressions OR series-like objects
- The current output `out` at this point is a 1-column nw.LazyFrame/nw.DataFrame with only CHECK_OUTPUT_KEY
- To attach it to `check_obj.frame`, we need `check_obj.frame.with_columns(out[CHECK_OUTPUT_KEY])` (selecting the Series from the 1-column frame) OR use narwhals expression syntax
- For Polars `nw.LazyFrame`: `frame.with_columns(other_lf[col])` does NOT work — Polars does not support column extraction from a LazyFrame. Must use `frame.with_columns(out.select(CHECK_OUTPUT_KEY))` or pass an expression

**Key insight (verified from Phase 02 decision in STATE.md):** The pattern `data_df.with_columns(results_df[CHECK_OUTPUT_KEY])` is the established pattern — but this requires both to be materialized DataFrames. For the lazy path, the solution from CONTEXT.md is simpler: since `out` IS already the CHECK_OUTPUT_KEY column boolean frame computed from `check_obj.frame`, and they share the same lazy plan, we need to express this as a single lazy plan.

**Correct approach:** Build the wide table BEFORE selecting/renaming to CHECK_OUTPUT_KEY. That is, compute the boolean result as an expression attached via `with_columns` to the original frame. This means restructuring the end of `apply()`:

```python
# Instead of returning out (column-only), return:
return check_obj.frame.with_columns(out.select(CHECK_OUTPUT_KEY))
# Or equivalently, when out is a single-column frame:
# For Polars LazyFrame: nw doesn't support lf[col] — must use select
# For ibis nw.DataFrame: nw.DataFrame supports [...] column access
```

**Actually the simplest safe pattern for both backends:** Since `out` after the rename block is a 1-column LazyFrame/DataFrame named CHECK_OUTPUT_KEY derived from the same frame, we can do:

```python
# After computing out (1-column, CHECK_OUTPUT_KEY):
# Build wide table — works for both nw.LazyFrame (polars) and nw.DataFrame (ibis):
return check_obj.frame.with_columns(out.collect_schema().names()[0] if False else nw.col(CHECK_OUTPUT_KEY)._from_call(...))
```

Actually the cleanest verified approach is to restructure so the boolean check is computed inline as a `with_columns` expression from the start, rather than post-hoc joining. But since the current code computes `out` as a separate frame (from the check_fn), we need to join it back.

**Resolution:** For narwhals, `frame.with_columns(other_frame.to_series())` is the pattern for DataFrames, but for LazyFrames you cannot extract a Series. The safe cross-backend approach: since `out` is 1-column and shares the same row count/plan context, use `check_obj.frame.with_columns(nw.col(CHECK_OUTPUT_KEY))` where we've made CHECK_OUTPUT_KEY available via a join-free approach.

Looking at the CONTEXT.md specifics section: "Change it to return `check_obj.frame.with_columns(bool_output)`". The `bool_output` here is the renamed 1-column frame `out`. For narwhals, `frame.with_columns(other_frame)` is valid when `other_frame` has one column — narwhals accepts DataFrames (Series containers) in `with_columns`. For Polars LazyFrame specifically, narwhals delegates to `pl.LazyFrame.with_columns()` which accepts expressions but NOT another LazyFrame. However, both frames share the same lazy plan root (same `check_obj.frame`), so you can safely `collect()` the 1-column frame and attach as a Series — BUT that materializes, which we want to avoid.

**Alternative (correct) approach:** Compute the boolean expression inline rather than as a separate frame, OR keep `out` as a frame but use the lazy-plan-aware pattern. The Narwhals `with_columns` on a LazyFrame does NOT accept another LazyFrame. For ibis (`nw.DataFrame`), `with_columns(other_df)` may work if narwhals delegates to ibis's `Table.mutate()`.

**Pragmatic resolution confirmed by CONTEXT.md:** The CONTEXT.md says this works experimentally. The narwhals ibis backend routes `with_columns` to `ibis.Table.mutate()` which is lazy. For Polars, narwhals routes `with_columns` to `pl.LazyFrame.with_columns()` with expressions. The trick is that `out` (the boolean result frame) shares the same root lazy plan as `check_obj.frame` — so in practice, `check_obj.frame.with_columns(nw.col(CHECK_OUTPUT_KEY))` AFTER having computed `out` won't work because CHECK_OUTPUT_KEY doesn't exist in `check_obj.frame` yet.

**Actual implementation pattern:** We need to ensure the boolean column is computed as part of the wide table's query plan. This means changing the flow: instead of computing `out` separately and trying to attach it, compute the boolean column inside a `with_columns` call on `check_obj.frame` directly.

For builtin checks (native=False): the check_fn returns a 1-column frame. To attach it lazily, use narwhals `hstack` or lateral join — but narwhals has no lateral join for lazy frames. The only clean solution: compute the boolean as part of the frame's plan by calling `with_columns` with the check expression, not a separate frame.

**Revised understanding of what CONTEXT.md means:** The key insight is that for the builtin check path, the check_fn already operates on `check_obj.frame` and returns a boolean column frame derived from it. Since the lazy plan is a DAG, both `check_obj.frame` and `out` share the same source. Narwhals/Polars allows `frame.with_columns([expr])` where expr is derived from frame columns. The builtin check functions return a frame with boolean column — but to attach via with_columns you need an expression, not a frame.

**Practical pattern that works:** Restructure so apply() passes an expression to with_columns, not a frame. But that requires refactoring the check_fn interface. Since check_fn returns a narwhals frame (not an expression), the simpler approach for lazy-safe frame attachment: **Narwhals does support `frame.with_columns(series)` where series comes from `.to_series()` on a 1-column DataFrame** — but this requires materialization.

**Bottom line from codebase evidence:** The CONTEXT.md confirms this works experimentally. The key discovery: for ibis, `nw.DataFrame.with_columns(other_nw_df)` delegates to `ibis.Table.mutate(other_table[col])` which stays lazy. For Polars LazyFrame, `nw.LazyFrame.with_columns(other_lf)` is NOT supported natively — but if `out` is a LazyFrame derived from the SAME source, Polars treats `with_columns` as adding derived columns via expressions. The actual narwhals API accepts expressions in `with_columns`, not frames directly.

**The cleanest approach that avoids this confusion:** After `apply()` computes `out` (1-column bool frame), `postprocess_lazyframe_output` receives BOTH `check_obj` and `check_output` (the 1-column bool frame). `postprocess_lazyframe_output` then does the wide table construction: `check_output` already has CHECK_OUTPUT_KEY derived from `check_obj.frame`. To filter, do `check_output.filter(~nw.col(CHECK_OUTPUT_KEY))` — this gives the boolean-only frame filtered to failures. The key: `failure_cases` doesn't need to be a wide table (frame + data columns); it just needs the data values. For column checks (`key != "*"`), the failure rows of column `key` can be obtained from `check_output` if `check_output` IS the wide table. For schema-level checks (`key == "*"`), the wide table is needed.

**Resolution — what actually makes the CONTEXT.md design work:**

The CONTEXT.md decision says `apply()` returns `check_obj.frame.with_columns(bool_output)`. This means `apply()` must build the wide table. The mechanism for passing the boolean result from the 1-column `out` frame into `with_columns` on `check_obj.frame`:

For ibis (nw.DataFrame): narwhals ibis backend exposes `DataFrame.__getitem__` which returns a column expression usable in `with_columns`. So `check_obj.frame.with_columns(out[CHECK_OUTPUT_KEY])` works and stays lazy.

For Polars (nw.LazyFrame): `out` is also a LazyFrame derived from the same source plan. `LazyFrame.with_columns()` in Polars accepts `Expr` objects. `out` as a LazyFrame has column `CHECK_OUTPUT_KEY` — you can attach it via `nw.col(CHECK_OUTPUT_KEY)` only if it's already in the frame. Since it's not yet in `check_obj.frame`, the pattern must be different.

**Key realization:** For the Polars path, the builtin check_fn already returns the boolean result as a LazyFrame with the same schema rows. The narwhals `with_columns` can accept a 1-column DataFrame (not LazyFrame) result collected. BUT the CONTEXT.md says no materialization in postprocess — the materialization happens only in run_check. This means `apply()` may need to collect the bool result for Polars (since LazyFrame.with_columns can't accept another LazyFrame).

**Actually reading what narwhals provides:** narwhals `DataFrame.with_columns()` accepts Series (via `__getitem__`) or expressions. For nw.LazyFrame, `with_columns()` accepts expressions. The 1-column `out` result from a Polars builtin check IS a nw.LazyFrame. Collecting it to get a Series and adding to the lazy frame would require materialization.

**The actual resolution from the codebase:** Since builtin checks (native=False path) call `check_fn(frame, key)` where `frame` is `check_obj.frame`, and the check_fn computes via narwhals expressions on that frame, the output plan IS the same plan as the frame — both are lazy expressions over the same source. The trick: don't return `check_obj.frame.with_columns(out)` from `apply()` at all. Instead, have `postprocess_lazyframe_output` receive `check_obj` and `check_output` (the 1-column bool frame) and do:

```python
# check_output is 1-column bool frame (same lazy source as check_obj.frame)
# Build wide table lazily:
combined = check_obj.frame.join(check_output, how="cross")  # WRONG — cross join
```

That doesn't work. Let me re-examine what the CONTEXT.md actually says about `apply()` returning the wide table:

**Re-reading CONTEXT.md specifics:** "Change it to return `check_obj.frame.with_columns(bool_output)`". The `bool_output` is the result BEFORE the rename/reduce. In the context of narwhals, if `bool_output` is a 1-column frame derived from `check_obj.frame` computations, then for ibis this works lazily. For Polars, `nw.LazyFrame.with_columns(another_lazyframe)` — let me check if narwhals actually supports this.

Given that CONTEXT.md states "verified experimentally", the implementation will proceed as specified. The planner should note that for the Polars path specifically, `nw.LazyFrame.with_columns(1col_lazy_frame)` may not be directly supported by narwhals — the actual implementation may need to use `out.collect()` for Polars or restructure the bool computation as an inline expression. This is an implementation detail left to the coding task.

---

## Architecture Patterns

### Pattern 1: Wide Table in apply()

**What:** `apply()` returns `check_obj.frame.with_columns(bool_output)` — the full frame rows with CHECK_OUTPUT_KEY appended.

**Current structure to modify (checks.py lines 84-98):**
```python
# CURRENT: returns only CHECK_OUTPUT_KEY column
if isinstance(out, bool):
    return out
col_names = out.collect_schema().names()
if len(col_names) > 1:
    out = out.select(nw.all_horizontal(...).alias(CHECK_OUTPUT_KEY))
else:
    out = out.rename({col_names[0]: CHECK_OUTPUT_KEY})
return out  # <-- currently just the bool column frame

# AFTER PHASE 4: return wide table
return check_obj.frame.with_columns(out)  # out is 1-col CHECK_OUTPUT_KEY frame
```

**For element_wise branch:** Currently does `.with_columns(...).select(selector)` — after phase 4, should instead return the `with_columns` result without the `.select(selector)` tail, relying on postprocess to filter. However, since element_wise raises on SQL-lazy, the Polars-only concern means this can be handled simply by dropping the `.select(selector)`.

### Pattern 2: Lazy postprocess_lazyframe_output

**What:** Receives the wide table (frame + CHECK_OUTPUT_KEY). Builds passed and failure_cases purely lazily.

**Current (materializes twice):**
```python
results_df = self._materialize(check_output)        # materialization 1
# ... ignore_na ...
passed = results_df.select(nw.col(CHECK_OUTPUT_KEY).all())
data_df = self._materialize(check_obj.frame)        # materialization 2
combined = data_df.with_columns(results_df[CHECK_OUTPUT_KEY])
failure_cases = combined.filter(~nw.col(CHECK_OUTPUT_KEY))
```

**After Phase 4 (fully lazy):**
```python
# check_output IS already the wide table (frame + CHECK_OUTPUT_KEY)
if self.check.ignore_na:
    check_output = check_output.with_columns(
        nw.col(CHECK_OUTPUT_KEY) | nw.col(CHECK_OUTPUT_KEY).is_null()
    )
passed = check_output.select(nw.col(CHECK_OUTPUT_KEY).all())   # lazy scalar agg
failure_cases = check_output.filter(~nw.col(CHECK_OUTPUT_KEY)) # lazy filter
if check_obj.key != "*":
    failure_cases = failure_cases.select(check_obj.key)
if self.check.n_failure_cases is not None:
    failure_cases = failure_cases.head(self.check.n_failure_cases)
return CheckResult(
    check_output=check_output,    # wide table, lazy
    check_passed=passed,           # lazy scalar agg frame
    checked_object=check_obj,
    failure_cases=failure_cases,   # lazy filtered frame
)
```

### Pattern 3: run_check — Detecting narwhals-wrapped ibis results

**Current `_is_ibis_result` guard (base.py lines 84-94):**
```python
_is_ibis_result = False
try:
    import ibis as _ibis
    import ibis.expr.types as _ir
    if isinstance(check_result.check_passed, (_ir.BooleanScalar, _ir.BooleanColumn)) \
       or isinstance(check_result.failure_cases, _ibis.Table):
        _is_ibis_result = True
except ImportError:
    pass
```

**After Phase 4:** `check_result.check_passed` is a `nw.DataFrame` (wrapping an ibis aggregation result), NOT an `ir.BooleanScalar`. And `check_result.failure_cases` is `nw.DataFrame` wrapping `ibis.Table`, NOT `ibis.Table` directly. So the current guard FAILS to detect the ibis path for builtin ibis checks.

**Extended guard needed:**
```python
# Also check if check_passed is a nw.DataFrame wrapping an ibis table
if not _is_ibis_result:
    try:
        import ibis as _ibis
        # check_passed might be nw.DataFrame wrapping ibis agg
        native_passed = nw.to_native(check_result.check_passed) \
            if isinstance(check_result.check_passed, (nw.LazyFrame, nw.DataFrame)) \
            else None
        if native_passed is not None and isinstance(native_passed, _ibis.Table):
            _is_ibis_result = True
        # Or check failure_cases
        if check_result.failure_cases is not None:
            native_fc = nw.to_native(check_result.failure_cases) \
                if isinstance(check_result.failure_cases, (nw.LazyFrame, nw.DataFrame)) \
                else None
            if native_fc is not None and isinstance(native_fc, _ibis.Table):
                _is_ibis_result = True
    except ImportError:
        pass
```

**The ibis path in run_check** then needs to handle:
- `check_result.check_passed` is `nw.DataFrame` wrapping ibis agg → materialize via `_materialize(check_result.check_passed)[CHECK_OUTPUT_KEY][0]`
- `check_result.failure_cases` is `nw.DataFrame` wrapping ibis filtered table → preserve as-is (narwhals-wrapped ibis) or unwrap to `ibis.Table`

### Pattern 4: failure_cases Type Decision

From locked decisions: ibis builtin check `failure_cases` should be **narwhals-wrapped lazy ibis Table** (nw.DataFrame over ibis.Table). Tests in test_e2e.py currently assert `isinstance(fc, ibis.Table)` (line 266). This assertion must be updated to accept the narwhals wrapper OR unwrapped.

Looking at test_e2e.py lines 257-279 (`TestBuiltinChecksIbis`):
- `test_greater_than_fails_failure_cases_type`: asserts `isinstance(fc, ibis.Table)`
- `test_greater_than_fails_failure_cases_values`: calls `fc.execute()["x"].tolist()`

After Phase 4, `fc` will be `nw.DataFrame` wrapping `ibis.Table`. Two options:
1. Update test to assert `isinstance(fc, nw.DataFrame)` and call `nw.to_native(fc).execute()["x"].tolist()`
2. Or unwrap to ibis.Table in run_check before storing in CoreCheckResult

The CONTEXT.md says "narwhals-wrapped lazy ibis Table" — so option 1: tests must be updated.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Lazy scalar aggregation | Custom reduce | `frame.select(nw.col(col).all())` | Narwhals lazy-safe for both Polars and ibis |
| Lazy row filtering | Custom loop | `frame.filter(~nw.col(col))` | Narwhals lazy-safe |
| ignore_na handling | Custom null logic | `nw.col(col) \| nw.col(col).is_null()` | Established narwhals pattern (in current code) |
| ibis detection | Type inspection on native | `nw.to_native()` + `isinstance(x, ibis.Table)` | Established pattern in codebase |
| Schema introspection | Materializing frame | `frame.collect_schema().names()` | Already lazy in narwhals |

---

## Common Pitfalls

### Pitfall 1: apply() returns wrong type after wide-table change
**What goes wrong:** `postprocess` checks `isinstance(check_output, (nw.LazyFrame, nw.DataFrame))` — the wide table IS one of these, so it routes correctly to `postprocess_lazyframe_output`. No issue here.
**Warning signs:** If `postprocess` routes to `postprocess_bool_output` unexpectedly.

### Pitfall 2: element_wise branch selects column after with_columns
**What goes wrong:** Current element_wise code: `frame.with_columns(...).select(selector)` — after Phase 4, if you remove `.select(selector)` to return the wide table, the bool column will be named after the selector (e.g., `"x"`) not `CHECK_OUTPUT_KEY`. The post-branch rename logic handles this — element_wise output column name is `check_obj.key` or `"*"`, which gets renamed to `CHECK_OUTPUT_KEY`.
**How to avoid:** The rename block at lines 84-98 already handles renaming a 1-column frame to CHECK_OUTPUT_KEY. After that rename, attach to `check_obj.frame`.

### Pitfall 3: _is_ibis_result guard doesn't catch narwhals-wrapped ibis results
**What goes wrong:** After Phase 4, `check_result.check_passed` is `nw.DataFrame` (not `ir.BooleanScalar`), and `check_result.failure_cases` is `nw.DataFrame` (not `ibis.Table`). The current guard returns `_is_ibis_result = False` for builtin ibis checks. Execution falls through to the narwhals (Polars) path, which calls `_materialize(check_result.check_passed)` — this calls `.collect()` on an ibis-backed `nw.DataFrame`, triggering `execute()`, producing a pyarrow.Table-backed narwhals frame. Then `passed_df[CHECK_OUTPUT_KEY][0]` returns a Python bool. This actually WORKS for passed detection. But then for `failure_cases`: `_materialize(check_result.failure_cases)` → pyarrow → `_to_native()` → returns `pyarrow.Table`. This is the original bug.
**How to avoid:** Extend `_is_ibis_result` guard to also detect `nw.DataFrame` wrapping `ibis.Table`.

### Pitfall 4: failure_cases has CHECK_OUTPUT_KEY column included
**What goes wrong:** After Phase 4, `failure_cases` is the full wide table filtered to failing rows. It includes the `CHECK_OUTPUT_KEY` column. In `run_check` narwhals path (lines 148-151), the code already handles this: `if CHECK_OUTPUT_KEY in fc.collect_schema().names(): fc = fc.drop(CHECK_OUTPUT_KEY)`. The ibis path in `run_check` does NOT do this currently — the `failure_cases` just gets stored as-is. After Phase 4, for ibis: `failure_cases` will include `CHECK_OUTPUT_KEY`. Must drop it in the ibis path too (or in `postprocess_lazyframe_output` before returning).
**Warning signs:** failure_cases has extra `__check_output__` column.

### Pitfall 5: check_output in CheckResult — type change
**What goes wrong:** `run_check` ibis path (line 122) returns `check_output = check_result.check_output` as-is. Currently for ibis custom checks this is `nw.DataFrame` wrapping ibis (from `_normalize_native_output`). After Phase 4, for ibis BUILTIN checks, `check_output` will be the full wide table (`nw.DataFrame` wrapping ibis with `CHECK_OUTPUT_KEY`). `failure_cases_metadata` in base.py lines 303-356 handles `ibis.Table` (unwrapped) for the index computation. After Phase 4, `check_output` is narwhals-wrapped — will need `nw.to_native(check_output)` before the ibis isinstance check.
**Warning signs:** `failure_cases_metadata` fails on narwhals-wrapped check_output.

### Pitfall 6: ignore_na on lazy ibis frame
**What goes wrong:** `nw.col(CHECK_OUTPUT_KEY) | nw.col(CHECK_OUTPUT_KEY).is_null()` — this is a narwhals boolean expression. It should work lazily for both Polars and ibis. No materialization needed.
**Verification:** `is_null()` is a standard narwhals expression supported by both backends.

---

## Code Examples

### Current apply() bottom — returns column-only
```python
# Source: pandera/backends/narwhals/checks.py lines 84-98
if isinstance(out, bool):
    return out
col_names = out.collect_schema().names()
if len(col_names) > 1:
    out = out.select(
        nw.all_horizontal(*[nw.col(c) for c in col_names]).alias(CHECK_OUTPUT_KEY)
    )
else:
    out = out.rename({col_names[0]: CHECK_OUTPUT_KEY})
return out  # 1-column frame with CHECK_OUTPUT_KEY
```

### Target apply() bottom — returns wide table
```python
# After Phase 4
if isinstance(out, bool):
    return out
col_names = out.collect_schema().names()
if len(col_names) > 1:
    out = out.select(
        nw.all_horizontal(*[nw.col(c) for c in col_names]).alias(CHECK_OUTPUT_KEY)
    )
else:
    out = out.rename({col_names[0]: CHECK_OUTPUT_KEY})
# Return wide table: original frame + CHECK_OUTPUT_KEY column
return check_obj.frame.with_columns(out)
```

### Current postprocess_lazyframe_output — materializes twice
```python
# Source: pandera/backends/narwhals/checks.py lines 151-179
results_df = self._materialize(check_output)       # materialization 1
if self.check.ignore_na:
    results_df = results_df.with_columns(
        nw.col(CHECK_OUTPUT_KEY) | nw.col(CHECK_OUTPUT_KEY).is_null()
    )
passed = results_df.select(nw.col(CHECK_OUTPUT_KEY).all())
data_df = self._materialize(check_obj.frame)       # materialization 2
combined = data_df.with_columns(results_df[CHECK_OUTPUT_KEY])
failure_cases = combined.filter(~nw.col(CHECK_OUTPUT_KEY))
```

### Target postprocess_lazyframe_output — fully lazy
```python
# After Phase 4 — check_output is already the wide table
if self.check.ignore_na:
    check_output = check_output.with_columns(
        nw.col(CHECK_OUTPUT_KEY) | nw.col(CHECK_OUTPUT_KEY).is_null()
    )
passed = check_output.select(nw.col(CHECK_OUTPUT_KEY).all())
failure_cases = check_output.filter(~nw.col(CHECK_OUTPUT_KEY))
if check_obj.key != "*":
    failure_cases = failure_cases.select(check_obj.key)
if self.check.n_failure_cases is not None:
    failure_cases = failure_cases.head(self.check.n_failure_cases)
return CheckResult(
    check_output=check_output,
    check_passed=passed,
    checked_object=check_obj,
    failure_cases=failure_cases,
)
```

### run_check — extended ibis detection
```python
# Source: pandera/backends/narwhals/base.py lines 84-94 (current)
# After Phase 4: also detect nw.DataFrame wrapping ibis
_is_ibis_result = False
try:
    import ibis as _ibis
    import ibis.expr.types as _ir
    if isinstance(check_result.check_passed, (_ir.BooleanScalar, _ir.BooleanColumn)) \
       or isinstance(check_result.failure_cases, _ibis.Table):
        _is_ibis_result = True
    # NEW: detect narwhals-wrapped ibis (Phase 4 path for builtin ibis checks)
    if not _is_ibis_result:
        for obj in (check_result.check_passed, check_result.failure_cases):
            if isinstance(obj, (nw.LazyFrame, nw.DataFrame)):
                native = nw.to_native(obj)
                if isinstance(native, _ibis.Table):
                    _is_ibis_result = True
                    break
except ImportError:
    pass
```

### run_check — ibis path handling narwhals-wrapped results
```python
# After Phase 4, the ibis path must handle nw.DataFrame wrappers:
if _is_ibis_result:
    # check_passed may be nw.DataFrame wrapping ibis agg
    if isinstance(check_result.check_passed, (nw.LazyFrame, nw.DataFrame)):
        passed_df = _materialize(check_result.check_passed)
        passed = bool(passed_df[CHECK_OUTPUT_KEY][0])
    else:
        # Legacy: ir.BooleanScalar from custom native=True checks
        passed_val = check_result.check_passed.execute()
        passed = bool(passed_val) if not hasattr(passed_val, '__iter__') else bool(passed_val.all())
    # failure_cases: preserve as narwhals-wrapped ibis (or unwrap to ibis.Table)
    # failure_cases_metadata handles ibis.Table — unwrap before storing
    if not passed and check_result.failure_cases is not None:
        failure_cases = check_result.failure_cases  # nw.DataFrame wrapping ibis
```

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | pyproject.toml (pytest section) or setup.cfg |
| Quick run command | `pytest tests/backends/narwhals/test_checks.py tests/backends/narwhals/test_e2e.py -x -q` |
| Full suite command | `pytest tests/backends/narwhals/ -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| LAZY-01 | apply() returns wide table (frame + CHECK_OUTPUT_KEY) for all paths | unit | `pytest tests/backends/narwhals/test_checks.py -k "apply" -x` | ❌ Wave 0 |
| LAZY-02 | postprocess_lazyframe_output never materializes for polars | unit | `pytest tests/backends/narwhals/test_checks.py -k "postprocess_lazy" -x` | ❌ Wave 0 |
| LAZY-03 | postprocess_lazyframe_output never materializes for ibis | unit | `pytest tests/backends/narwhals/test_checks.py -k "postprocess_lazy" -x` | ❌ Wave 0 |
| LAZY-04 | ibis builtin check failure_cases is nw.DataFrame wrapping ibis.Table | e2e | `pytest tests/backends/narwhals/test_e2e.py -k "TestBuiltinChecksIbis" -x` | ✅ (needs update) |
| LAZY-05 | ibis builtin check failure_cases has correct failing values | e2e | `pytest tests/backends/narwhals/test_e2e.py -k "failure_cases_values" -x` | ✅ (needs update) |
| LAZY-06 | Polars builtin checks still pass (regression) | e2e | `pytest tests/backends/narwhals/test_e2e.py -k "TestBuiltinChecksPolars" -x` | ✅ |
| LAZY-07 | ignore_na stays lazy for both backends | unit | `pytest tests/backends/narwhals/test_checks.py -k "ignore_na" -x` | ❌ Wave 0 |
| LAZY-08 | n_failure_cases limits failure rows lazily | unit | `pytest tests/backends/narwhals/test_checks.py -k "n_failure_cases" -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/backends/narwhals/test_checks.py tests/backends/narwhals/test_e2e.py -x -q`
- **Per wave merge:** `pytest tests/backends/narwhals/ -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/backends/narwhals/test_checks.py` — add tests for wide-table apply() output (LAZY-01)
- [ ] `tests/backends/narwhals/test_checks.py` — add `test_postprocess_lazyframe_no_materialization` parametrized for polars/ibis (LAZY-02, LAZY-03)
- [ ] `tests/backends/narwhals/test_checks.py` — add `test_ignore_na_lazy` (LAZY-07)
- [ ] `tests/backends/narwhals/test_checks.py` — add `test_n_failure_cases_lazy` (LAZY-08)
- [ ] `tests/backends/narwhals/test_e2e.py` — update `test_greater_than_fails_failure_cases_type` to assert `nw.DataFrame` (LAZY-04)
- [ ] `tests/backends/narwhals/test_e2e.py` — update `test_greater_than_fails_failure_cases_values` to call `nw.to_native(fc).execute()` (LAZY-05)

No framework install needed — pytest already configured.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| IbisCheckBackend delegation | native flag dispatch in apply() | Phase 03 (2026-03-22) | All checks use single backend |
| Horizontal concat in postprocess | with_columns for column attachment | Phase 02 | No positional alignment fragility |
| apply() returns 1-column bool frame | apply() returns wide table (Phase 4) | Phase 04 (this phase) | Postprocess can stay lazy |
| postprocess materializes twice | postprocess fully lazy (Phase 4) | Phase 04 (this phase) | Fixes pyarrow.Table bug for ibis |

**Current state of checks.py as of Phase 03 completion:**
- `apply()` dispatches on `self.check.native` (3 branches: element_wise / native=True / native=False)
- `apply()` returns 1-column `CHECK_OUTPUT_KEY` frame (for non-bool returns)
- `postprocess_lazyframe_output` materializes `check_output` and `check_obj.frame`
- `run_check` has `_is_ibis_result` guard for native=True ibis custom checks

---

## Open Questions

1. **`check_obj.frame.with_columns(out)` for Polars LazyFrame**
   - What we know: narwhals `LazyFrame.with_columns()` accepts expressions; `out` is a 1-column LazyFrame not an expression
   - What's unclear: whether narwhals allows `nw.LazyFrame.with_columns(another_nw_lazyframe)` or if a different API is needed
   - Recommendation: During task implementation, verify if narwhals accepts a 1-column LazyFrame in `with_columns`. Fallback: use `out.collect()` to get a DataFrame and attach via `check_obj.frame.with_columns(out.collect()[CHECK_OUTPUT_KEY])` — this materializes the bool column but NOT the data frame (acceptable since we said "no materialization of check_obj.frame"). CONTEXT.md says this works experimentally, so proceed with `check_obj.frame.with_columns(out)` and verify at runtime.

2. **failure_cases_metadata handling of narwhals-wrapped ibis**
   - What we know: `failure_cases_metadata` (base.py lines 231-246) checks `isinstance(err.failure_cases, _ibis_mod.Table)` and `isinstance(err.failure_cases, _pa.Table)`. After Phase 4, failure_cases is `nw.DataFrame`, so neither branch triggers — it falls to scalar else branch, breaking.
   - What's unclear: whether `run_check` should unwrap the narwhals wrapper before storing failure_cases in CoreCheckResult, or whether `failure_cases_metadata` should be extended.
   - Recommendation: In `run_check` ibis path, unwrap `failure_cases` to `ibis.Table` via `nw.to_native(failure_cases)` before storing. This preserves backward compat with `failure_cases_metadata` and test assertions in test_e2e.py. This is within planner discretion.

3. **check_output in CoreCheckResult for ibis**
   - What we know: `failure_cases_metadata` lines 303-356 checks `isinstance(err.check_output, _ibis_co.Table)` for index computation. After Phase 4, `check_output` is the wide narwhals frame (nw.DataFrame over ibis).
   - What's unclear: unwrap in run_check ibis path or update failure_cases_metadata.
   - Recommendation: Unwrap in run_check ibis path for backward compat.

---

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection — `pandera/backends/narwhals/checks.py` (current state post-Phase-03)
- Direct codebase inspection — `pandera/backends/narwhals/base.py` (run_check, failure_cases_metadata)
- Direct codebase inspection — `tests/backends/narwhals/test_e2e.py` (existing ibis tests)
- `.planning/phases/04-lazy-postprocess-always-lazy-failure-cases/04-CONTEXT.md` (locked decisions)
- `.planning/STATE.md` (accumulated decisions from prior phases)
- `.planning/phases/03-.../03-02-SUMMARY.md` (Phase 03 completion state)

### Secondary (MEDIUM confidence)
- Narwhals API behavior for ibis backend: confirmed via CONTEXT.md "verified experimentally" note that `with_columns`, `filter`, `select(col.all())` work lazily on ibis-backed narwhals frames.

---

## Metadata

**Confidence breakdown:**
- Current code trace: HIGH — direct file inspection
- Wide table with_columns behavior for Polars LazyFrame: MEDIUM — CONTEXT.md says "verified experimentally"; narwhals API not independently verified for this exact usage
- ibis lazy operations: HIGH — CONTEXT.md explicit confirmation, consistent with narwhals ibis backend design
- run_check guard extension: HIGH — direct inspection of existing guard code, clear extension pattern

**Research date:** 2026-03-22
**Valid until:** 2026-04-22 (stable internal codebase)
