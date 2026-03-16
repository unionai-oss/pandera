# Phase 3: Column Backend - Research

**Researched:** 2026-03-09
**Domain:** Narwhals column validation backend (check_nullable, check_unique, check_dtype, run_checks)
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| COLUMN-01 | `pandera/backends/narwhals/components.py` with `ColumnBackend` implementing `check_nullable` (float NaN via `is_nan()`), `check_unique`, `check_dtype` (via narwhals engine), and `run_checks` | Verified: narwhals Expr API supports `is_null()`, `is_nan()`, `is_duplicated()`, `collect_schema()` — all core operations confirmed working |
| COLUMN-02 | `check_unique` forces collection via `.collect()` before calling `is_duplicated()`; collect-first pattern documented | Verified: `is_duplicated()` works on both eager `nw.DataFrame` and lazy `nw.LazyFrame`, but Ibis (SQL-lazy) wraps as `nw.DataFrame` with no `.collect()` — `_materialize()` pattern from `checks.py` handles both backends correctly |
</phase_requirements>

---

## Summary

Phase 3 implements `pandera/backends/narwhals/components.py`, containing a `ColumnBackend` class that validates a single narwhals column. The four required methods are `check_nullable`, `check_unique`, `check_dtype`, and `run_checks`. All four are direct narwhals adaptations of the Polars backend in `pandera/backends/polars/components.py`.

The narwhals port replaces Polars-specific APIs (`pl.col`, `pl.concat`, `is_not_null`, `is_not_nan`) with narwhals Expr equivalents (`nw.col`, `nw.concat`, `is_null`, `is_nan`), and replaces schema inspection via `get_lazyframe_schema()` with `lf.collect_schema()`. The critical cross-backend concern is materialization: Polars wraps frames as `nw.LazyFrame`, while Ibis wraps as `nw.DataFrame`. The `_materialize()` helper from `checks.py` covers both cases and must be reused (or inlined) in `components.py`.

Float NaN detection requires checking the column dtype via `collect_schema()` — narwhals `is_nan()` on integer columns returns `null` (not `False`) so the NaN branch must be gated on float dtype. Dtype checking passes the narwhals schema dtype through `narwhals_engine.Engine.dtype()` to produce a pandera `DataType`, then calls `schema.dtype.check(col_pandera_dtype)`. `failure_cases` for dtype errors is the dtype string (scalar), consistent with the Polars backend. The `run_check` method must be implemented directly on `ColumnBackend` because `Phase 4` is where the shared `NarwhalsSchemaBackend` base will be extracted; for now `ColumnBackend` can extend `BaseSchemaBackend` and implement `run_check` inline.

**Primary recommendation:** Mirror `polars/components.py` exactly, replacing Polars APIs with narwhals equivalents. Use `_materialize()` from `checks.py` to handle Polars vs Ibis frame differences. Implement `run_check` inline using the `nw.DataFrame` check_passed access pattern. Use `narwhals_engine.Engine.dtype()` for dtype translation in `check_dtype`. Use `_to_native()` from `api/narwhals/utils.py` at every `failure_cases` construction site.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `narwhals.stable.v1` | >=2.15.0 | Narwhals stable API — all narwhals imports must use this path | Insulates from breaking API changes per project decision |
| `pandera.backends.base` | (internal) | `BaseSchemaBackend`, `CoreCheckResult` — base class and result container | Required interface for backend registration |
| `pandera.engines.narwhals_engine` | (internal) | `Engine.dtype()` — translates narwhals dtypes to pandera `DataType` | Already implemented in Phase 1, used for dtype translation |
| `pandera.api.narwhals.utils` | (internal) | `_to_native()` — convert narwhals frame to native at SchemaError construction sites | Required to prevent narwhals wrappers leaking into error messages |
| `pandera.constants` | (internal) | `CHECK_OUTPUT_KEY = 'check_output'` — column name for boolean check output frame | Used throughout existing backends; consistency required |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pandera.validation_depth` | (internal) | `validate_scope` decorator with `ValidationScope.DATA` / `ValidationScope.SCHEMA` | `check_nullable`, `check_unique`, `check_dtype` must be decorated — exact same pattern as polars backend |
| `pandera.errors` | (internal) | `SchemaError`, `SchemaErrorReason` — error classes used when constructing failure results | `SERIES_CONTAINS_NULLS`, `SERIES_CONTAINS_DUPLICATES`, `WRONG_DATATYPE`, `DATAFRAME_CHECK`, `CHECK_ERROR` |
| `pandera.api.base.error_handler` | (internal) | `ErrorHandler`, `get_error_category` — used in `run_checks_and_handle_errors` | Needed for lazy error collection |

---

## Architecture Patterns

### Recommended Project Structure
```
pandera/backends/narwhals/
├── __init__.py          # already exists
├── checks.py            # Phase 2 — NarwhalsCheckBackend (done)
├── builtin_checks.py    # Phase 2 — 14 built-in check implementations (done)
└── components.py        # Phase 3 — ColumnBackend (NEW)
```

### Pattern 1: ColumnBackend Class Hierarchy

**What:** `ColumnBackend` inherits from `BaseSchemaBackend` directly. It does NOT inherit from `PolarsSchemaBackend` (which has Polars-specific `subsample`, `run_check`, `failure_cases_metadata`, `drop_invalid_rows`). Phase 4 will extract `NarwhalsSchemaBackend` as a shared base — Phase 3 implements `run_check` inline.

**When to use:** Always — this is the only pattern for Phase 3.

```python
# Source: pandera/backends/polars/components.py (adapted)
import narwhals.stable.v1 as nw
from pandera.backends.base import BaseSchemaBackend, CoreCheckResult

class ColumnBackend(BaseSchemaBackend):
    """Column backend for narwhals frames (Polars, Ibis, etc.)."""
```

### Pattern 2: Materialization for Cross-Backend Collection

**What:** Ibis frames arrive as `nw.DataFrame` (not `nw.LazyFrame`). The `_materialize()` helper from `checks.py` handles both. Use it wherever `.collect()` is needed.

**When to use:** `check_unique` (before `is_duplicated()`), `run_check` (to get the boolean from `check_passed`), and anywhere failure_cases must be collected.

```python
# Source: pandera/backends/narwhals/checks.py — _materialize static method
@staticmethod
def _materialize(frame) -> nw.DataFrame:
    if isinstance(frame, nw.LazyFrame):
        return frame.collect()
    native = nw.to_native(frame)
    if hasattr(native, "execute"):
        return nw.from_native(native.execute())
    return frame  # already eager nw.DataFrame
```

**Options for reuse:** Either import `NarwhalsCheckBackend._materialize` as a module-level function, OR duplicate the 6-line helper in `components.py`. Duplication is acceptable since Phase 4 will unify into `NarwhalsSchemaBackend`.

### Pattern 3: Float NaN Detection

**What:** Narwhals `is_nan()` on integer columns returns `null` values (not `False`). The NaN check must be gated on float dtype detection.

**When to use:** Always, inside `check_nullable`.

```python
# Source: verified experimentally against narwhals 2.x
schema_obj = check_obj.collect_schema()
col_dtype = schema_obj[schema.selector]
is_float = isinstance(col_dtype, (nw.Float32, nw.Float64))

if is_float:
    null_mask = nw.col(schema.selector).is_null() | nw.col(schema.selector).is_nan()
else:
    null_mask = nw.col(schema.selector).is_null()
```

**Confidence:** HIGH — confirmed via live test: `is_nan()` on `Int64` column returns `null` per row (not `False`), which would corrupt the null check result if not gated.

### Pattern 4: check_nullable Implementation

**What:** Computes null+nan mask, materializes both frames, horizontally concatenates, filters to get failure cases, wraps in `CoreCheckResult`.

**Key difference from Polars:** Polars uses `pl.concat(..., how="horizontal")` on LazyFrames. Narwhals requires collecting both frames first (narwhals does not support lazy horizontal concat — this is documented in `checks.py`).

```python
# Source: pandera/backends/polars/components.py (adapted for narwhals)
@validate_scope(scope=ValidationScope.DATA)
def check_nullable(self, check_obj, schema) -> list[CoreCheckResult]:
    if schema.nullable:
        return [CoreCheckResult(passed=True, check="not_nullable",
                                reason_code=SchemaErrorReason.SERIES_CONTAINS_NULLS)]

    schema_obj = check_obj.collect_schema()
    col_dtype = schema_obj[schema.selector]
    is_float = isinstance(col_dtype, (nw.Float32, nw.Float64))

    null_expr = nw.col(schema.selector).is_null()
    if is_float:
        null_expr = null_expr | nw.col(schema.selector).is_nan()

    is_null = check_obj.select(null_expr)
    # Materialize both — narwhals does NOT support lazy horizontal concat
    data_df = _materialize(check_obj)
    is_null_df = _materialize(is_null)
    passed = not is_null_df[schema.selector].any()  # True if no nulls

    if passed:
        return [CoreCheckResult(passed=True, ...)]

    combined = nw.concat(
        [data_df, is_null_df.rename({schema.selector: CHECK_OUTPUT_KEY})],
        how="horizontal"
    )
    failure_cases = _to_native(
        combined.filter(nw.col(CHECK_OUTPUT_KEY)).select(schema.selector)
    )
    return [CoreCheckResult(passed=False, check="not_nullable",
                            reason_code=SchemaErrorReason.SERIES_CONTAINS_NULLS,
                            message=f"non-nullable column '{schema.selector}' contains null values",
                            failure_cases=failure_cases)]
```

### Pattern 5: check_unique Implementation (COLUMN-02)

**What:** Force `.collect()` via `_materialize()` before calling `is_duplicated()`. This is the "collect-first" pattern required by COLUMN-02.

**Why the collect-first pattern:** `is_duplicated()` is a window function that requires full data visibility. For Polars `LazyFrame`, narwhals delegates to Polars which can handle this lazily — but for SQL-lazy backends (Ibis), the frame arrives as `nw.DataFrame` wrapping an Ibis table that must be executed first. The `_materialize()` call ensures both backends work.

```python
# Source: pandera/backends/polars/components.py (adapted for narwhals)
@validate_scope(scope=ValidationScope.DATA)
def check_unique(self, check_obj, schema) -> list[CoreCheckResult]:
    check_name = "field_uniqueness"
    if not schema.unique:
        return [CoreCheckResult(passed=True, check=check_name,
                                reason_code=SchemaErrorReason.SERIES_CONTAINS_DUPLICATES)]

    # COLUMN-02: force collection before is_duplicated()
    collected = _materialize(check_obj.select(schema.selector))
    duplicates = collected.select(nw.col(schema.selector).is_duplicated())

    results = []
    for column in duplicates.collect_schema().names():
        if not duplicates[column].any():
            continue
        combined = nw.concat(
            [collected, duplicates.rename({column: "_duplicated"})],
            how="horizontal"
        )
        failure_cases = _to_native(
            combined.filter(nw.col("_duplicated")).select(column)
        )
        results.append(CoreCheckResult(
            passed=False, check=check_name,
            check_output=duplicates.select(~nw.col(column)).rename({column: CHECK_OUTPUT_KEY}),
            reason_code=SchemaErrorReason.SERIES_CONTAINS_DUPLICATES,
            message=f"column '{schema.selector}' not unique:\n{failure_cases}",
            failure_cases=failure_cases,
        ))
    return results
```

### Pattern 6: check_dtype Implementation

**What:** Gets column dtype via `collect_schema()`, translates to pandera dtype via `narwhals_engine.Engine.dtype()`, calls `schema.dtype.check()`. failure_cases is the dtype string (scalar) — consistent with Polars backend behavior. The requirement says "produces a SchemaError with a native frame in failure_cases" but the Polars backend uses `str(obj_dtype)` (scalar). Use scalar string for consistency with Polars backend.

```python
# Source: pandera/backends/polars/components.py (adapted for narwhals)
@validate_scope(scope=ValidationScope.SCHEMA)
def check_dtype(self, check_obj, schema) -> list[CoreCheckResult]:
    if schema.dtype is None:
        return [CoreCheckResult(passed=True, check=f"dtype('{schema.dtype}')",
                                reason_code=SchemaErrorReason.WRONG_DATATYPE)]

    from pandera.engines import narwhals_engine
    results = []
    schema_obj = check_obj.select(schema.selector).collect_schema()
    for column, nw_dtype in zip(schema_obj.names(), schema_obj.dtypes()):
        try:
            col_pandera_dtype = narwhals_engine.Engine.dtype(nw_dtype)
        except TypeError:
            col_pandera_dtype = nw_dtype  # fallback: let .check() return False
        passed = schema.dtype.check(col_pandera_dtype)
        results.append(CoreCheckResult(
            passed=bool(passed),
            check=f"dtype('{schema.dtype}')",
            reason_code=SchemaErrorReason.WRONG_DATATYPE,
            message=(f"expected column '{column}' to have type "
                     f"{schema.dtype}, got {nw_dtype}") if not passed else None,
            failure_cases=str(nw_dtype) if not passed else None,
        ))
    return results
```

### Pattern 7: run_checks and run_check

**What:** `run_checks` iterates schema.checks and calls `self.run_check()` per check. `run_check` calls `check(check_obj, selector)` which invokes `NarwhalsCheckBackend.__call__`, then extracts the boolean from the `CheckResult.check_passed` (which is an `nw.DataFrame` after materialization in `postprocess_lazyframe_output`).

```python
# run_check for narwhals — adapted from PolarsSchemaBackend
def run_check(self, check_obj, schema, check, check_index, *args) -> CoreCheckResult:
    from pandera.constants import CHECK_OUTPUT_KEY
    check_result = check(check_obj, *args)

    # check_passed is nw.DataFrame (materialized in postprocess)
    passed_df = _materialize(check_result.check_passed)
    passed = bool(passed_df[CHECK_OUTPUT_KEY][0])

    failure_cases = None
    message = None
    if not passed:
        if check_result.failure_cases is None:
            failure_cases = passed
            message = (f"{schema.__class__.__name__} '{schema.name}' failed "
                       f"{check_index}: {check}")
        else:
            fc = _materialize(check_result.failure_cases)
            if CHECK_OUTPUT_KEY in fc.collect_schema().names():
                fc = fc.drop(CHECK_OUTPUT_KEY)
            failure_cases = _to_native(fc)
            failure_cases_msg = failure_cases.head().rows(named=True) if hasattr(failure_cases, 'head') else failure_cases
            message = (f"{schema.__class__.__name__} '{schema.name}' failed "
                       f"validator number {check_index}: {check} "
                       f"failure case examples: {failure_cases_msg}")
        if check.raise_warning:
            import warnings
            from pandera.errors import SchemaWarning
            warnings.warn(message, SchemaWarning)
            return CoreCheckResult(passed=True, check=check,
                                   reason_code=SchemaErrorReason.DATAFRAME_CHECK)

    check_output_df = _materialize(check_result.check_output)
    return CoreCheckResult(
        passed=passed, check=check, check_index=check_index,
        check_output=_to_native(check_output_df),
        reason_code=SchemaErrorReason.DATAFRAME_CHECK,
        message=message, failure_cases=failure_cases,
    )
```

### Anti-Patterns to Avoid

- **Using `.collect()` directly on a frame:** Always use `_materialize()` — Ibis frames arrive as `nw.DataFrame` and have no `.collect()` method; calling `.collect()` raises `AttributeError`.
- **Passing narwhals wrappers as `failure_cases` to `SchemaError`:** Always call `_to_native()` before passing frames to `SchemaError` — requirement INFRA-03 mandates native frames in error messages.
- **Calling `is_nan()` on non-float columns without dtype check:** On integer columns, `is_nan()` returns `null` per row (not `False`), which corrupts the null check aggregation.
- **Using `pl.concat` or Polars-specific APIs:** This file must use only `nw.*` APIs.
- **Importing `narwhals_engine` at module top level:** Keep engine import inside `check_dtype` method body to avoid import order issues consistent with existing narwhals backend patterns.
- **Using `nw.horizontal_concat` (does not exist):** Only `nw.concat(..., how="horizontal")` is available, and only on eager `nw.DataFrame`, not `nw.LazyFrame`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Materializing LazyFrame or Ibis DataFrame | Custom isinstance checks + collect calls | `_materialize()` from `checks.py` | Already handles both Polars LazyFrame and Ibis DataFrame-wrapped SQL backends |
| Converting narwhals frame to native | `nw.to_native(frame)` inline | `_to_native()` from `api/narwhals/utils.py` | `pass_through=True` makes it safe to call on already-native frames; prevents double-unwrap errors |
| Float dtype detection | String comparison on dtype names | `isinstance(col_dtype, (nw.Float32, nw.Float64))` | Confirmed working — `isinstance` works on narwhals DTypeClass instances |
| Schema dtype translation | Manual dtype mapping | `narwhals_engine.Engine.dtype(nw_dtype)` | Phase 1 already built all 14 dtype registrations |
| ValidationScope gating | Manual config context checks | `@validate_scope(scope=ValidationScope.DATA/SCHEMA)` decorator | Existing decorator handles `validation_depth` config; exact pattern from Polars backend |

**Key insight:** This phase is almost entirely an adaptation of `polars/components.py`. The only genuinely new logic is the `_materialize()` usage for Ibis compatibility and the float dtype gate for `is_nan()`.

---

## Common Pitfalls

### Pitfall 1: Ibis frames have no `.collect()`

**What goes wrong:** `check_obj.collect()` raises `AttributeError: 'DataFrame' object has no attribute 'collect'` when the frame is an Ibis-backed `nw.DataFrame`.

**Why it happens:** The narwhals Ibis integration wraps Ibis tables as `nw.DataFrame` (eager-looking), not `nw.LazyFrame`. The `.collect()` method only exists on `nw.LazyFrame`.

**How to avoid:** Never call `.collect()` directly. Always route through `_materialize()` which handles the `isinstance(frame, nw.LazyFrame)` branch and the `hasattr(native, 'execute')` branch.

**Warning signs:** `AttributeError` mentioning `.collect` on a `DataFrame` object in tests with `backend="ibis"`.

### Pitfall 2: `nw.concat` requires eager frames

**What goes wrong:** `nw.concat([lazy_frame, other], how="horizontal")` raises or produces unexpected results.

**Why it happens:** Narwhals does not support lazy horizontal concat. This is documented in `checks.py` comment: "Materialize both frames — narwhals does NOT support lazy horizontal concat."

**How to avoid:** Always materialize (via `_materialize()`) both sides before `nw.concat(..., how="horizontal")`.

**Warning signs:** Exception during `nw.concat` in `check_nullable` or `check_unique`.

### Pitfall 3: `is_nan()` on integer columns

**What goes wrong:** `check_nullable` reports incorrect results for integer nullable=False columns because `is_nan()` returns `null` rows (not `False`), which causes `any()` to return unexpected results.

**Why it happens:** `is_nan()` is only meaningful for float types. On integers, narwhals (backed by Polars) returns `null` per row.

**How to avoid:** Gate the `is_nan()` expression behind `isinstance(col_dtype, (nw.Float32, nw.Float64))`. Only add the nan branch for float columns.

**Warning signs:** `check_nullable` failing or giving unexpected pass/fail on integer columns with nulls.

### Pitfall 4: `check_dtype` failure_cases format

**What goes wrong:** Planner or tests expect `failure_cases` to be a native frame (per requirement wording "produces a SchemaError with a native frame in failure_cases") but the Polars backend uses `str(obj_dtype)` (scalar string).

**Why it happens:** The requirement says "native frame" but the actual Polars implementation stores the dtype as a string scalar for dtype errors. Frame-based failure_cases are for row-level violations; dtype mismatches are schema-level and don't have row-level failure cases.

**How to avoid:** Use `str(nw_dtype)` (scalar string) as `failure_cases` for `check_dtype`, matching polars backend behavior. The "native frame" wording in the requirement refers to run-time checks (`run_checks`), not `check_dtype`.

**Warning signs:** Tests failing on `failure_cases` type assertions for dtype check results.

### Pitfall 5: Module-level import of `narwhals_engine`

**What goes wrong:** Circular import error when `components.py` is loaded.

**Why it happens:** `narwhals_engine.py` imports from `pandera.dtypes` and `pandera.engines`, which may trigger import chains back through backends.

**How to avoid:** Import `narwhals_engine` inside the `check_dtype` method body, not at the top of the module. Consistent with how `pandera.api.narwhals.types` is imported inside methods in `narwhals_engine.py`.

---

## Code Examples

Verified patterns from official sources:

### Narwhals collect_schema (schema inspection without materializing)
```python
# Source: verified against narwhals.stable.v1 — collect_schema() is available on both
# nw.LazyFrame and nw.DataFrame; returns a Schema object with .names() and .dtypes()
schema = lf.collect_schema()
names = schema.names()       # list[str]
dtypes = schema.dtypes()     # list[nw dtype]
col_dtype = schema["x"]      # direct column dtype lookup
```

### Narwhals horizontal concat (eager only)
```python
# Source: verified experimentally — must materialize first
data_df = _materialize(check_obj)    # nw.DataFrame
mask_df = _materialize(mask_lf)     # nw.DataFrame
combined = nw.concat([data_df, mask_df.rename({"col": CHECK_OUTPUT_KEY})], how="horizontal")
failure_cases = combined.filter(nw.col(CHECK_OUTPUT_KEY))
```

### Float dtype detection
```python
# Source: verified against narwhals.stable.v1 — isinstance works on DTypeClass instances
schema_obj = check_obj.collect_schema()
col_dtype = schema_obj[column_name]
is_float = isinstance(col_dtype, (nw.Float32, nw.Float64))
```

### Dtype translation for check_dtype
```python
# Source: verified against narwhals_engine.Engine.dtype()
from pandera.engines import narwhals_engine
nw_dtype = check_obj.collect_schema()["x"]   # e.g. Float64
pandera_dtype = narwhals_engine.Engine.dtype(nw_dtype)  # e.g. narwhals_engine.Float64
passed = schema.dtype.check(pandera_dtype)   # bool
```

### Access boolean value from nw.DataFrame check_passed
```python
# Source: verified experimentally — after _materialize(), check_passed is nw.DataFrame
passed_df = _materialize(check_result.check_passed)  # nw.DataFrame
passed = bool(passed_df[CHECK_OUTPUT_KEY][0])
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `pl.col`, `pl.concat`, `pl.LazyFrame` directly | `nw.col`, `nw.concat`, `nw.LazyFrame` via `narwhals.stable.v1` | Phase 1 decision | Enables Ibis, Polars, and future backends through single code path |
| `get_lazyframe_schema(lf)` (polars utils) | `lf.collect_schema()` (narwhals native) | narwhals 2.x | Narwhals provides schema inspection without materializing |
| `is_float_dtype()` from `polars/base.py` | `isinstance(col_dtype, (nw.Float32, nw.Float64))` | narwhals 2.x | narwhals DTypeClass supports isinstance checks |
| `.collect().item()` on single-cell polars frame | `df[CHECK_OUTPUT_KEY][0]` on nw.DataFrame | narwhals 2.x | narwhals does not expose `.item()` — use index access |

**Deprecated/outdated:**
- `get_lazyframe_schema()`, `get_lazyframe_column_names()` from `pandera.api.polars.utils`: Polars-specific; use `collect_schema()` directly on narwhals frames.
- `is_float_dtype()` from `pandera.backends.polars.base`: Polars-specific; use narwhals isinstance pattern.
- `PolarsSchemaBackend` as base class for narwhals ColumnBackend: Will be replaced by `NarwhalsSchemaBackend` in Phase 4.

---

## Open Questions

1. **Should `_materialize()` be imported from `checks.py` or duplicated?**
   - What we know: `NarwhalsCheckBackend._materialize` is a static method; importing it creates coupling between `components.py` and `checks.py`.
   - What's unclear: Whether the planner will prefer extraction to a module-level utility vs inline duplication.
   - Recommendation: Duplicate the 6-line helper in `components.py` as a module-level function. Phase 4 will extract it into `NarwhalsSchemaBackend` base class anyway.

2. **Should ColumnBackend receive a full `validate()` method in Phase 3?**
   - What we know: The Phase 3 success criteria only require the four core check methods. The Polars `ColumnBackend.validate()` method exists but orchestrates coercion and lazy error collection — this belongs to Phase 4 container work.
   - What's unclear: Whether tests will call `check_nullable` etc. directly or go through `validate()`.
   - Recommendation: Implement the four check methods and `run_checks`/`run_check` only. Do NOT implement `validate()` — keep it scoped per requirements. Tests should call methods directly.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | `pyproject.toml` (pytest config section) |
| Quick run command | `pytest tests/backends/narwhals/test_components.py -x` |
| Full suite command | `pytest tests/backends/narwhals/ -x` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| COLUMN-01 | `check_nullable` rejects null values for `nullable=False` columns | unit | `pytest tests/backends/narwhals/test_components.py::test_check_nullable_fails -x` | ❌ Wave 0 |
| COLUMN-01 | `check_nullable` passes for `nullable=True` columns | unit | `pytest tests/backends/narwhals/test_components.py::test_check_nullable_passes -x` | ❌ Wave 0 |
| COLUMN-01 | `check_nullable` catches float NaN values (not just null) | unit | `pytest tests/backends/narwhals/test_components.py::test_check_nullable_catches_nan -x` | ❌ Wave 0 |
| COLUMN-01 | `check_unique` detects duplicate values | unit | `pytest tests/backends/narwhals/test_components.py::test_check_unique_fails -x` | ❌ Wave 0 |
| COLUMN-01 | `check_unique` passes on unique data | unit | `pytest tests/backends/narwhals/test_components.py::test_check_unique_passes -x` | ❌ Wave 0 |
| COLUMN-01 | `check_dtype` returns False for wrong dtype | unit | `pytest tests/backends/narwhals/test_components.py::test_check_dtype_wrong -x` | ❌ Wave 0 |
| COLUMN-01 | `check_dtype` returns True for correct dtype | unit | `pytest tests/backends/narwhals/test_components.py::test_check_dtype_correct -x` | ❌ Wave 0 |
| COLUMN-01 | `run_checks` executes Check objects and collects results | unit | `pytest tests/backends/narwhals/test_components.py::test_run_checks -x` | ❌ Wave 0 |
| COLUMN-02 | `check_unique` materializes before `is_duplicated()` (Ibis compat) | unit | `pytest tests/backends/narwhals/test_components.py -x -k ibis` | ❌ Wave 0 |
| COLUMN-01+02 | All methods work on both polars and ibis backends (parameterized) | unit | `pytest tests/backends/narwhals/test_components.py -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/backends/narwhals/test_components.py -x`
- **Per wave merge:** `pytest tests/backends/narwhals/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/backends/narwhals/test_components.py` — covers COLUMN-01, COLUMN-02 (parameterized with polars and ibis fixtures from existing conftest)
- [ ] No new framework or fixture infrastructure needed — `make_narwhals_frame` fixture in `conftest.py` already handles both polars and ibis frame creation

---

## Sources

### Primary (HIGH confidence)
- Live code execution against `narwhals.stable.v1` (version installed in project `.pixi/envs/default`) — verified `is_nan`, `is_null`, `is_duplicated`, `collect_schema`, `isinstance(dtype, nw.Float32/Float64)`, `nw.concat` behavior
- `pandera/backends/polars/components.py` — authoritative reference implementation; narwhals port follows this exactly
- `pandera/backends/narwhals/checks.py` — `_materialize()` helper, materialization patterns for Ibis vs Polars
- `pandera/engines/narwhals_engine.py` — `Engine.dtype()` verified working for dtype translation
- `pandera/api/narwhals/utils.py` — `_to_native()` with `pass_through=True`
- `pandera/api/narwhals/types.py` — `NarwhalsData` NamedTuple definition

### Secondary (MEDIUM confidence)
- `.planning/STATE.md` accumulated decisions — `narwhals.stable.v1` import requirement, Ibis materializes as `nw.DataFrame` not `nw.LazyFrame`
- `.planning/REQUIREMENTS.md` — COLUMN-01, COLUMN-02 exact specifications

### Tertiary (LOW confidence)
- None

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all imports and APIs verified against live project code
- Architecture: HIGH — Polars backend is the exact template; adaptations verified experimentally
- Pitfalls: HIGH — all four pitfalls confirmed via live test execution (is_nan on int, concat on lazy, collect on Ibis DataFrame, dtype format)

**Research date:** 2026-03-09
**Valid until:** 2026-04-09 (narwhals stable.v1 API is stable; 30-day window appropriate)
