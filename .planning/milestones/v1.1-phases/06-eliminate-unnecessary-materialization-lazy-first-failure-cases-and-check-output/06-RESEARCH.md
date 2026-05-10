# Phase 6: Eliminate Unnecessary Materialization — Lazy-First failure_cases and check_output

**Researched:** 2026-03-23
**Domain:** Narwhals backend internal architecture — lazy evaluation, CoreCheckResult lifecycle, SchemaError/SchemaErrors construction
**Confidence:** HIGH (all findings from direct codebase inspection; no external libraries to resolve)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**CoreCheckResult.failure_cases type (internal)**
- `CoreCheckResult.failure_cases` is not user-facing — it is internal backend state
- Must hold narwhals wrappers (`nw.LazyFrame` for polars lazy, `nw.DataFrame` for ibis)
- `run_check()` removes `fc.collect()` — failure_cases stays lazy in CoreCheckResult
- `run_check()` removes `_materialize(check_output)` and `_to_native()` — check_output stays lazy
- `run_check()` only materializes once: `select(nw.col(CHECK_OUTPUT_KEY).all())` to evaluate the scalar passed bool
- `_is_ibis_result` guard is fully removed — dead code after Phase 5's uniform expression protocol

**SchemaError.failure_cases (single-error, non-lazy path)**
- Type: **native** (unwrap from narwhals before storing in SchemaError)
- Content: **raw failing rows only** — no enrichment columns (consistent with polars backend)
- Users call `.collect()` / `.execute()` themselves if they want materialized data
- Consistent with polars backend behavior; replaces the ibis backend's forced `.to_pandas()` conversion

**SchemaErrors.failure_cases (lazy mode, aggregated) — failure_cases_metadata() redesign**
- **In scope for Phase 6** — current polars hardcoding violates lazy-first principle for ibis inputs
- Type: **native** (pl.LazyFrame for polars lazy, ibis.Table for ibis, pl.DataFrame for polars eager)
- Content: **enriched with metadata columns** — `failure_case`, `schema_context`, `column`, `check`, `check_number`, `index`
- `failure_cases_metadata()` redesigned to build enriched frame using narwhals / native backend ops instead of always converting to `pl.DataFrame` via polars
- **Row index**: included only for eager inputs (`pl.DataFrame`, `pd.DataFrame`) where row order is well-defined; `None` for lazy/SQL backends — forcing materialization just for index is inconsistent with lazy-first principle
- SQL backends don't guarantee row ordering anyway

**check_nullable() evaluation**
- Fix: only materialize the scalar boolean — `_materialize(combined_lf.select(nw.col(CHECK_OUTPUT_KEY).any()))` — one row, not the full frame
- `failure_cases` stays lazy (`combined_lf.filter(nw.col(CHECK_OUTPUT_KEY)).select(col)`) — narwhals wrapper in CoreCheckResult
- `check_output` stays lazy (the wide `combined_lf` with CHECK_OUTPUT_KEY column)
- Matches the run_check lazy-first pattern

**subsample() lazy strategy**
- `head=`: use `nw.LazyFrame.head(n)` directly — fully lazy for all backends, no materialization
- `tail=` on polars: use `nw.LazyFrame.tail(n)` directly — lazy
- `tail=` on SQL-lazy backends (ibis/DuckDB): raise `NotImplementedError` — SQL has no native TAIL without forced ordering
- Both head + tail: concatenate lazily via `nw.concat([check_obj.head(head), check_obj.tail(tail)]).unique()`
- No full-frame materialization in `subsample()`

**No Native-Type Branching (Hard Constraint)**
- The implementation must NOT introduce new `isinstance(x, ibis.Table)` / `isinstance(x, pl.LazyFrame)` / `isinstance(x, pl.DataFrame)` branches scattered through the backend logic
- The narwhals abstraction should handle the unified path throughout
- Native-type handling is acceptable **only at the boundaries** — when constructing the final SchemaError/SchemaErrors failure_cases, or when a backend-specific API is truly unavoidable (isolated in a helper)
- The goal is to *reduce* native-type branching (remove `_is_ibis_result`), not add new forms

### Claude's Discretion
- Whether `_materialize` helper can be simplified or removed after these changes
- Exact narwhals operations used to build the enriched metadata frame in `failure_cases_metadata()` for ibis (e.g., `mutate()` for literal columns)
- Whether `check_unique()` and `check_dtype()` need similar lazy fixes (investigate during planning)
- How to detect whether the input is eager vs lazy for the row-index decision in `failure_cases_metadata()`

### Deferred Ideas (OUT OF SCOPE)
- Making `postprocess_bool_output` produce the user's backend frame instead of falling back to polars for ibis
- Redesigning `check_unique()` for full laziness (currently calls `_materialize(dup_values)`) — investigate during planning; may be in scope
- Whether `failure_cases_metadata()` enrichment should also apply to `SchemaError.failure_cases` (single-error path)
</user_constraints>

---

## Summary

Phase 6 enforces the lazy-first principle as an architectural invariant throughout the narwhals backend. The core insight is that `.collect()` / `.execute()` should be called **exactly once per check** — to evaluate the scalar boolean "did this check pass?" — and all other data structures (`failure_cases`, `check_output`) should remain as narwhals wrappers until handed back to the user.

The current code has five materialization violations:
1. `run_check()` calls `fc.collect()` and `_materialize(check_output)` / `_to_native()` on the polars path (lines 146, 158, 163)
2. `run_check()` has a dead `_is_ibis_result` bifurcation (lines 76–123) that Phase 5 made unreachable since `apply()` now returns uniform narwhals frames for all backends
3. `subsample()` calls `_materialize(check_obj)` before `.head()` / `.tail()` (lines 54–56) — defeating the point of lazy evaluation before subsampling
4. `check_nullable()` calls `_materialize(combined_lf)` to get the full frame (line 131), then evaluates `.any()` — only the scalar is needed
5. `failure_cases_metadata()` is hardcoded to polars: it calls `_materialize()`, converts via Arrow to `pl.DataFrame`, and builds metadata with `pl.lit()` / `pl.Series` / `pl.concat` — this forces polars materialization even for ibis inputs

Additionally, the existing test `test_ibis_lazy_failure_cases_is_dataframe` in `test_e2e.py` (line 633) asserts `isinstance(fc, pl.DataFrame)` for the ibis lazy path — this test must be updated to assert `isinstance(fc, ibis.Table)` to match the new contract.

**Primary recommendation:** Three focused implementation plans: (1) collapse `run_check()` + fix `subsample()` + fix `check_nullable()`, (2) redesign `failure_cases_metadata()`, (3) update tests and boundary unwrapping in `container.py` / `components.py`.

---

## Architecture Patterns

### Current Code Structure

The five affected locations map to exactly three files:

| File | Location | Problem |
|------|----------|---------|
| `pandera/backends/narwhals/base.py` | `run_check()` lines 76–167 | Dead ibis bifurcation + fc.collect() + _materialize(check_output) + _to_native() |
| `pandera/backends/narwhals/base.py` | `subsample()` lines 54–56 | `_materialize(check_obj)` before .head()/.tail() |
| `pandera/backends/narwhals/base.py` | `failure_cases_metadata()` lines 181–310 | Entire method hardcoded to polars |
| `pandera/backends/narwhals/components.py` | `check_nullable()` lines 131–154 | Full-frame _materialize to evaluate one .any() |
| `pandera/backends/narwhals/container.py` | `validate()` line 114 | `sample_obj.lazy()` — works but forces nw.DataFrame before re-lazifying |

### Pattern 1: Scalar-Only Materialization in run_check()

The new unified `run_check()` has one code path (no ibis branch):

```python
# Source: base.py run_check() — Phase 6 target shape (from CONTEXT.md)
# Only materialization: one scalar bool
passed_lf = check_result.check_passed  # nw.LazyFrame (polars) or nw.DataFrame (ibis)
passed = bool(_materialize(passed_lf)[CHECK_OUTPUT_KEY][0])

failure_cases = None
message = None
if not passed:
    fc = check_result.failure_cases  # stays as nw.LazyFrame or nw.DataFrame
    if fc is None:
        failure_cases = passed
        message = f"Check '{check}' failed — no failure cases captured."
    else:
        if CHECK_OUTPUT_KEY in fc.collect_schema().names():
            fc = fc.drop(CHECK_OUTPUT_KEY)
        failure_cases = fc  # narwhals wrapper — NOT collected here
        message = f"Check '{check}' failed."
    ...

return CoreCheckResult(
    passed=passed,
    check_output=check_result.check_output,  # stays lazy — NOT _materialize() here
    failure_cases=failure_cases,             # stays as nw wrapper — NOT _to_native() here
    ...
)
```

### Pattern 2: Lazy subsample()

```python
# Source: base.py subsample() — Phase 6 target shape (from CONTEXT.md)
def subsample(self, check_obj, head=None, tail=None, sample=None, random_state=None):
    if sample is not None:
        raise NotImplementedError(...)

    if head is None and tail is None:
        return check_obj

    # Detect SQL-lazy (ibis) for tail= guard
    if tail is not None:
        native = nw.to_native(check_obj)
        if hasattr(native, "execute"):  # SQL-lazy: ibis.Table has .execute()
            raise NotImplementedError(
                "tail= is not supported on SQL-lazy backends (Ibis, DuckDB, PySpark) "
                "because SQL has no native TAIL without forced full ordering. "
                "Use head= instead."
            )

    obj_subsample = []
    if head is not None:
        obj_subsample.append(check_obj.head(head))   # lazy — no _materialize()
    if tail is not None:
        obj_subsample.append(check_obj.tail(tail))   # lazy — polars-only branch

    return nw.concat(obj_subsample).unique()
```

Note: `hasattr(native, "execute")` is the established pattern (Phase 04-02 decision) for detecting SQL-lazy vs polars. No new ibis isinstance needed.

### Pattern 3: Scalar-Only check_nullable()

```python
# Source: components.py check_nullable() — Phase 6 target shape (from CONTEXT.md)
combined_lf = check_obj.with_columns(null_expr.alias(CHECK_OUTPUT_KEY))

# Materialize ONE ROW to evaluate the scalar bool — not the full frame
has_nulls_df = _materialize(combined_lf.select(nw.col(CHECK_OUTPUT_KEY).any()))
has_nulls = bool(has_nulls_df[CHECK_OUTPUT_KEY][0])

if not has_nulls:
    return [CoreCheckResult(passed=True, ...)]

# failure_cases and check_output stay lazy
failure_cases = combined_lf.filter(nw.col(CHECK_OUTPUT_KEY)).select(col)  # nw.LazyFrame
check_output_lf = combined_lf                                               # nw.LazyFrame
return [CoreCheckResult(
    passed=False,
    check_output=check_output_lf,    # narwhals wrapper
    failure_cases=failure_cases,     # narwhals wrapper
    ...
)]
```

### Pattern 4: Backend-Agnostic failure_cases_metadata()

The redesigned method must build an enriched frame without touching polars directly. Two sub-paths based on whether the input is lazy/SQL or eager:

**Eager path** (polars eager, pandas): row index is meaningful — can derive it from check_output. Compute index the same way as today but keep using narwhals ops where possible.

**Lazy path** (polars lazy, ibis): row index is always `None` — no forced ordering. Attach metadata columns using narwhals `.with_columns(nw.lit(...).alias(...))` calls. The result stays as a narwhals frame in the user's original type, unwrapped to native only at the final `FailureCaseMetadata` construction.

Eager-vs-lazy detection:
```python
# Source: Phase 01-pr-review-architecture-fixes decision in STATE.md
# hasattr(return_type, "collect") on the CLASS distinguishes lazy from eager
# For each err.failure_cases: check whether it IS a LazyFrame (lazy path) or DataFrame
def _is_lazy(frame) -> bool:
    """True for nw.LazyFrame; False for nw.DataFrame (even SQL-lazy ibis ones)."""
    return isinstance(frame, nw.LazyFrame)
```

Note: ibis inputs arrive as `nw.DataFrame` (not `nw.LazyFrame`) — they are "SQL-lazy" but narwhals wraps them as DataFrame. So `isinstance(fc, nw.LazyFrame)` correctly separates polars-lazy from everything else, and `hasattr(nw.to_native(fc), "execute")` separates ibis from polars-eager.

For index=None path (lazy/SQL), attach literal metadata:
```python
# narwhals .with_columns(nw.lit(value).alias(col_name)) works for both polars and ibis
enriched = fc.with_columns(
    nw.lit(schema_context_str).alias("schema_context"),
    nw.lit(column_str).alias("column"),
    nw.lit(check_str).alias("check"),
    nw.lit(check_number_int).alias("check_number"),
    nw.lit(None).alias("index"),
)
```

For the `failure_case` column (single-column vs multi-column):
```python
if len(fc.collect_schema().names()) > 1:
    # multi-column: JSON-encode rows — ibis has no direct .rows(named=True) API
    # For ibis: use native table.mutate(failure_case=ibis.literal("{complex}"))
    # or defer to polars for the struct→JSON step (acceptable at boundary)
    pass
else:
    fc = fc.rename({fc.collect_schema().names()[0]: "failure_case"})
```

The multi-column case for ibis requires attention. Narwhals has no `.rows(named=True)` cross-backend API. Options:
1. For ibis multi-column failure_cases: serialize each column separately and concatenate as a string (ibis string ops)
2. Accept that multi-column failure_cases in `failure_cases_metadata()` for ibis uses native ibis `.mutate()` — isolated to one helper, not scattered
3. Convert only the failure_cases portion to Arrow for struct encoding (already the established pattern from Phase 04-03)

**Recommendation (Claude's Discretion):** Use narwhals `.with_columns(nw.lit(...))` for all metadata columns, rename the single-column case with narwhals `.rename()`, and for multi-column ibis, convert only `fc` to Arrow via `nw.to_native(fc).to_pyarrow()` then `pl.from_arrow()` just for the struct-encode step. This keeps the breach isolated.

### Pattern 5: Boundary Unwrapping in SchemaError Construction

After `run_check()` returns a `CoreCheckResult` with narwhals-wrapped `failure_cases`, the unwrap to native happens in `run_checks_and_handle_errors()` when constructing `SchemaError`. Currently `failure_cases=result.failure_cases` passes the narwhals wrapper into `SchemaError.__init__`.

The unwrap point must be exactly here (in `components.py` and `container.py`):

```python
# In run_checks_and_handle_errors (components.py) and the for-result loop (container.py):
error = SchemaError(
    schema=schema,
    data=check_obj,
    message=result.message,
    failure_cases=_to_native(result.failure_cases),   # <-- unwrap HERE
    check=result.check,
    check_index=result.check_index,
    check_output=result.check_output,                  # stays as narwhals wrapper for failure_cases_metadata()
    reason_code=result.reason_code,
)
```

The `check_output` field in `SchemaError` must remain as a narwhals wrapper (or None) because `failure_cases_metadata()` uses it to derive the row index for the eager path. `failure_cases` is unwrapped to native since `SchemaError.failure_cases` is user-facing.

### Anti-Patterns to Avoid

- **Materializing check_output in run_check()**: `_materialize(check_result.check_output)` — deleted; check_output must stay lazy for failure_cases_metadata() index derivation
- **isinstance(fc, nw.LazyFrame) as the collect guard**: the old pattern `if isinstance(fc, nw.LazyFrame): fc = fc.collect()` — replaced with lazy-first (no collect)
- **pl.from_arrow() inside the hot path**: only acceptable at the final boundary (FailureCaseMetadata construction), not in per-error processing loops
- **`_is_ibis_result` detection in run_check()**: entirely deleted — Phase 5 made it dead code
- **Full-frame _materialize for a scalar**: `_materialize(combined_lf)` to get `.any()` in check_nullable — replaced with `_materialize(combined_lf.select(...any()))` selecting one row

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Detecting SQL-lazy backend | Custom isinstance chain | `hasattr(nw.to_native(frame), "execute")` | Established Phase 04-02 pattern; polars LazyFrame has no .execute() |
| Detecting eager vs lazy frame | isinstance on native types | `isinstance(frame, nw.LazyFrame)` for polars-lazy; then `hasattr(native, "execute")` for ibis | Already established; no new isinstance needed |
| Adding literal metadata columns | polars pl.lit() | `nw.lit(value).alias(name)` in `.with_columns()` | Works across all narwhals backends including ibis |
| Converting to Arrow for struct JSON encoding | Custom serialization | `nw.to_native(fc).to_pyarrow()` + `pl.from_arrow()` | Established Phase 04-03 pattern; use only at boundary |
| Scalar bool from LazyFrame | Manual index into collected frame | `_materialize(lf.select(expr))[col][0]` | Minimizes materialization to one scalar row |

**Key insight:** The narwhals abstraction (`nw.lit`, `nw.col`, `.with_columns`, `.filter`, `.select`, `.head()`, `.tail()`, `nw.concat`) covers all the lazy-first operations needed. The only places where narwhals is insufficient are struct→JSON encoding for multi-column failure_cases and the final native unwrap at SchemaError/FailureCaseMetadata boundaries.

---

## Common Pitfalls

### Pitfall 1: check_output Narwhals Type in SchemaError

**What goes wrong:** `check_output` is stored in `SchemaError` as a narwhals wrapper (post-Phase 6). If `failure_cases_metadata()` tries to call `.with_row_index()` on it without checking whether it is a `nw.LazyFrame` or `nw.DataFrame`, the ibis path will fail (ibis tables don't have `.with_row_index()` in the narwhals API).

**Why it happens:** `failure_cases_metadata()` currently calls `co.with_row_index("index")` on check_output for the row-index computation. This works for polars but not ibis.

**How to avoid:** Guard the row-index computation: only attempt it for eager frames where row order is well-defined. Use `isinstance(co, nw.LazyFrame) or hasattr(nw.to_native(co), "execute")` to skip index for lazy/SQL inputs.

**Warning signs:** `AttributeError` on `nw.DataFrame.with_row_index` for ibis inputs.

### Pitfall 2: `collect_schema().names()` After DROP

**What goes wrong:** After `run_check()` drops `CHECK_OUTPUT_KEY` from `fc`, the resulting lazy frame still needs to be passed as a narwhals wrapper to `CoreCheckResult.failure_cases`. If the caller later calls `fc.collect_schema().names()` to check schema, this is fine (lazy-safe). But if the caller forgets to `drop(CHECK_OUTPUT_KEY)` before handing to `SchemaError`, the metadata enrichment will include the internal boolean column.

**How to avoid:** The DROP must happen in `run_check()` before assigning to `failure_cases`, or alternatively in the enrichment step of `failure_cases_metadata()` via a filter/select that excludes the boolean column.

### Pitfall 3: subsample() Returns Wrong Type for container.py

**What goes wrong:** The current `container.py` (lines 113–116) normalizes the subsample result:
```python
if isinstance(sample_obj, nw.DataFrame):
    sample_lf = sample_obj.lazy()
else:
    sample_lf = sample_obj  # already nw.LazyFrame
```
After Phase 6, `subsample()` returns `nw.LazyFrame` (from `.head()` / `.tail()` / `nw.concat(...)`). For ibis, `check_obj.head(n)` returns a `nw.DataFrame` (ibis's wrapped DataFrame, not a LazyFrame). The normalization in `container.py` will try to call `.lazy()` on an ibis-backed `nw.DataFrame`, which may not work as expected.

**How to avoid:** Keep the normalization block in `container.py` and verify what `nw.DataFrame.lazy()` does for ibis-backed frames. Alternatively, the normalization is only needed if ibis-backed `nw.DataFrame` from `.head()` can't be used directly by subsequent check methods.

**Confidence:** MEDIUM — needs verification during planning whether `nw.DataFrame.lazy()` is a no-op or raises for ibis.

### Pitfall 4: _to_native() on None

**What goes wrong:** When `failure_cases` is `None` (check passed) or the scalar `False` (bool check failed with no per-row data), calling `_to_native(result.failure_cases)` in the SchemaError construction boundary will fail or return unexpected results.

**How to avoid:** Guard the unwrap: `_to_native(result.failure_cases) if isinstance(result.failure_cases, (nw.LazyFrame, nw.DataFrame)) else result.failure_cases`. The existing `_to_native()` uses `pass_through=True` which safely returns non-narwhals values unchanged — verify this handles None and scalar bool correctly.

**Resolution:** `nw.to_native(None, pass_through=True)` returns `None` ✓ and `nw.to_native(False, pass_through=True)` returns `False` ✓ — `_to_native()` is safe to call unconditionally.

### Pitfall 5: Test Updates — Existing Assertions on pl.DataFrame for Ibis

**What goes wrong:** `test_e2e.py` line 633–645: `test_ibis_lazy_failure_cases_is_dataframe` asserts `isinstance(fc, pl.DataFrame)` for the SchemaErrors path from ibis input. Phase 6 changes this to `ibis.Table`.

**Why it matters:** This test will become RED (assertion error) after the `failure_cases_metadata()` redesign. It must be updated to assert native ibis.Table instead of pl.DataFrame.

**Additional test:** `test_greater_than_fails_failure_cases_type` in `TestBuiltinChecksPolars` (line 188–199) asserts `isinstance(fc, nw.DataFrame)`. Phase 6 changes SchemaError.failure_cases to native — this test must be updated to `isinstance(fc, pl.DataFrame)` for polars-eager inputs.

**And:** `test_greater_than_fails_failure_cases_type` in `TestBuiltinChecksIbis` (line 261–275) asserts `isinstance(fc, nw.DataFrame)` wrapping ibis.Table. Phase 6 changes this to native ibis.Table directly — update assertion.

### Pitfall 6: check_unique() — investigate for lazy fix

**Current state:** `check_unique()` in `components.py` (line 185) calls `_materialize(dup_values)` to get `native_dups`, then checks `len(native_dups) == 0`. This is a scalar-like check (just testing emptiness) but materializes the full duplicates frame.

**Is it fixable?** The group_by approach already avoids full-frame materialization. `_materialize(dup_values)` only materializes the (hopefully small) set of duplicate values — not the entire input frame. The current approach is acceptable since we only materialize the aggregated result, not the original data. However, checking `count > 0` lazily is possible: `.select(nw.len()).collect()[0][0] > 0` would tell us if duplicates exist without materializing all duplicate rows.

**Recommendation (Claude's Discretion):** Investigate during planning. The materialization scope is bounded (only duplicate rows, post-aggregation), so this is lower priority than the other four fixes. May be deferred.

---

## Code Examples

### Existing _materialize() Helper (unchanged)

```python
# Source: pandera/api/narwhals/utils.py
def _materialize(frame) -> nw.DataFrame:
    if isinstance(frame, nw.LazyFrame):
        return frame.collect()
    native = nw.to_native(frame)
    if hasattr(native, "execute"):
        return nw.from_native(native.execute())
    return frame  # already eager DataFrame
```

This helper remains the single authorized materialization point. After Phase 6, it is only called with scalar-select arguments: `.select(nw.col(CHECK_OUTPUT_KEY).all())` or `.select(nw.col(CHECK_OUTPUT_KEY).any())`.

### Existing SQL-Lazy Detection Pattern (established, use unchanged)

```python
# Source: Phase 04-02 decision in STATE.md
native = nw.to_native(check_obj)
if hasattr(native, "execute"):  # ibis.Table has .execute(); pl.LazyFrame does not
    # SQL-lazy path
```

### Narwhals Literal Column Addition

```python
# Narwhals cross-backend literal attach — works for polars AND ibis
enriched = fc.with_columns(
    nw.lit("Column").alias("schema_context"),
    nw.lit("col_name").alias("column"),
    nw.lit("greater_than(0)").alias("check"),
    nw.lit(0).alias("check_number"),
    nw.lit(None).alias("index"),
)
```

Confidence: HIGH for polars. MEDIUM for ibis — narwhals `nw.lit()` maps to ibis `ibis.literal()` internally; verify narwhals stable v1 supports this across backends.

### nw.LazyFrame.head() / .tail() — Lazy for Polars

```python
# Source: narwhals docs — head/tail on LazyFrame stays lazy
lf.head(5)   # returns nw.LazyFrame (polars: pl.LazyFrame.head() is lazy)
lf.tail(5)   # returns nw.LazyFrame (polars: pl.LazyFrame.tail() is lazy)
```

For ibis: `check_obj.head(n)` where `check_obj` is `nw.DataFrame` wrapping ibis.Table calls `ibis.Table.limit(n)` under the hood — stays as ibis query, no execution.

### nw.concat() for Lazy Frames

```python
# Concatenate two lazy frames — stays lazy for polars; returns narwhals wrapper for ibis
result = nw.concat([lf.head(head), lf.tail(tail)]).unique()
```

---

## State of the Art

| Old Approach | Current Approach (Phase 6) | Impact |
|--------------|---------------------------|--------|
| Two branches: `_is_ibis_result` (ibis) + narwhals path | Single unified path — remove ibis branch entirely | Reduces `run_check()` from ~90 lines to ~40 lines |
| `_materialize(combined_lf)` in check_nullable → full frame in memory | `_materialize(combined_lf.select(...any()))` → 1 row in memory | O(N) → O(1) memory for nullable check |
| `_materialize(check_obj)` before .head()/.tail() in subsample() | `check_obj.head(n)` / `check_obj.tail(n)` directly | Subsample stays lazy until actual use |
| `failure_cases_metadata()` always materializes to pl.DataFrame via Arrow | Builds enriched frame lazily in user's backend type | ibis inputs return ibis.Table; polars-lazy returns pl.LazyFrame |
| `fc.collect()` in run_check() polars path | fc stays as nw.LazyFrame in CoreCheckResult | failure_cases never collected inside run_check |
| `_to_native(check_output_df)` on materialized output in run_check() | check_output stays as nw wrapper | check_output available lazily for failure_cases_metadata index derivation |

---

## Open Questions

1. **`nw.DataFrame.lazy()` for ibis-backed frames**
   - What we know: `container.py` line 113 normalizes `nw.DataFrame` to `nw.LazyFrame` via `.lazy()`
   - What's unclear: does `nw.DataFrame.lazy()` on an ibis-backed nw.DataFrame work correctly, or does it fail / return a wrapper that breaks subsequent `.filter()` calls?
   - Recommendation: Test during planning. If `.lazy()` doesn't work for ibis, remove the normalization and let downstream checks accept `nw.DataFrame` directly.

2. **narwhals nw.lit() support for ibis backend**
   - What we know: narwhals translates nw.lit() to ibis.literal() for ibis backends per its translation layer
   - What's unclear: confirmed support in narwhals stable v1 for all literal dtypes (int, str, None)
   - Recommendation: Test `nw.lit(None).alias("index")` on an ibis-backed frame during the failure_cases_metadata implementation.

3. **Multi-column failure_cases struct encoding for ibis**
   - What we know: `pl.Series(fc.rows(named=True))` struct encoding is polars-specific; no equivalent in narwhals
   - What's unclear: best narwhals-agnostic approach for struct→JSON encoding of ibis frames
   - Recommendation: For ibis multi-column failure_cases, use `nw.to_native(fc).to_pyarrow()` then `pl.from_arrow()` at the boundary (same as Phase 04-03 pattern). This materializes only the failure_cases (a small, filtered frame), not the original data.

4. **check_unique() lazy fix — in or out of scope**
   - What we know: current `_materialize(dup_values)` only materializes the (small) duplicate-values aggregation, not the full frame
   - Deferred context says "investigate during planning; may be in scope"
   - Recommendation: Include in Phase 6 only if the fix is trivial (replace `len(native_dups) == 0` with a lazy `.select(nw.len())` check). If it requires restructuring the result-building logic, defer.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | `pyproject.toml` (pytest section) |
| Quick run command | `pytest tests/backends/narwhals/ -x -q` |
| Full suite command | `pytest tests/backends/narwhals/ -q` |

### Phase Requirements → Test Map

No formal REQ-IDs specified. The behaviors to test are derived from the locked decisions:

| Behavior | Test Type | Automated Command | File Exists? |
|----------|-----------|-------------------|-------------|
| run_check() materializes only scalar bool | unit | `pytest tests/backends/narwhals/test_checks.py -x -q` | ✅ (update existing) |
| run_check() failure_cases is narwhals wrapper, not native | unit | `pytest tests/backends/narwhals/test_checks.py -x -q` | ✅ (new assertions) |
| subsample() head= stays lazy (polars) | unit | `pytest tests/backends/narwhals/test_components.py -x -q` | ✅ (new test) |
| subsample() tail= raises NotImplementedError for ibis | unit | `pytest tests/backends/narwhals/test_components.py -x -q` | ✅ (new test) |
| check_nullable() only materializes scalar .any() | unit | `pytest tests/backends/narwhals/test_components.py -x -q` | ✅ (update existing) |
| SchemaError.failure_cases is native for polars | e2e | `pytest tests/backends/narwhals/test_e2e.py -x -q` | ✅ (update assertion) |
| SchemaError.failure_cases is native ibis.Table for ibis | e2e | `pytest tests/backends/narwhals/test_e2e.py -x -q` | ✅ (update assertion) |
| SchemaErrors.failure_cases is ibis.Table for ibis lazy | e2e | `pytest tests/backends/narwhals/test_e2e.py -x -q` | ✅ (update assertion) |
| failure_cases_metadata() builds enriched frame in user's type | unit | `pytest tests/backends/narwhals/ -k failure_cases_metadata -x -q` | ❌ Wave 0 |
| _is_ibis_result block deleted (no dead code) | static | grep / code review | ❌ Wave 0 (verify in review) |

### Sampling Rate

- **Per task commit:** `pytest tests/backends/narwhals/ -x -q`
- **Per wave merge:** `pytest tests/backends/narwhals/ -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] New test for `failure_cases_metadata()` with ibis input — covers lazy-native return type
- [ ] New test for `subsample()` lazy head/tail behavior — one test per case

*(Existing test files cover the rest; assertions need updating not new files.)*

---

## Sources

### Primary (HIGH confidence)
- Direct inspection of `pandera/backends/narwhals/base.py` — full source of `run_check()`, `subsample()`, `failure_cases_metadata()`
- Direct inspection of `pandera/backends/narwhals/components.py` — full source of `check_nullable()`, `check_unique()`
- Direct inspection of `pandera/backends/narwhals/container.py` — container validate() + SchemaError construction
- Direct inspection of `pandera/api/narwhals/utils.py` — `_materialize()` and `_to_native()` implementations
- Direct inspection of `tests/backends/narwhals/test_e2e.py` — existing assertions on failure_cases types
- Direct inspection of `.planning/phases/06-*/06-CONTEXT.md` — all locked decisions
- Direct inspection of `.planning/STATE.md` — accumulated Phase 01–05 decisions

### Secondary (MEDIUM confidence)
- narwhals `.head()` / `.tail()` laziness on LazyFrame: inferred from narwhals design (LazyFrame operations return LazyFrame); confirmed by absence of `.collect()` in narwhals source for these ops
- `nw.lit()` cross-backend support for ibis: inferred from narwhals translation layer design; not directly verified from narwhals docs

### Tertiary (LOW confidence)
- `nw.DataFrame.lazy()` behavior for ibis-backed frames: unknown — identified as Open Question 1

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies; all tools already in use
- Architecture: HIGH — derived directly from current source code and locked CONTEXT.md decisions
- Pitfalls: HIGH (most) / MEDIUM (subsample normalization, nw.lit for ibis)

**Research date:** 2026-03-23
**Valid until:** Stable indefinitely — this is internal codebase research, not external library research
