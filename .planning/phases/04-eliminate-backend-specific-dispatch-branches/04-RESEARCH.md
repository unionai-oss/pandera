# Phase 4: Eliminate Backend-Specific Dispatch Branches - Research

**Researched:** 2026-05-25
**Domain:** Narwhals backend internal refactoring — PySpark dispatch branches in base.py, components.py, container.py
**Confidence:** HIGH

## Summary

Phase 4 removes or properly abstracts the four `is_pyspark` / `Implementation.PYSPARK` dispatch violations introduced during Phase 2 (02-04) to make PySpark tests pass. These branches were pragmatic bug-fixes under time pressure; Phase 4 addresses the architectural debt.

There are four distinct dispatch sites, each with a different character:

1. **`base.py:run_check` lines 143-154** — PySpark branch avoids `_materialize()` (which calls `.execute()`, absent on `pyspark.sql.DataFrame`) by using native `.first()`. This branch is *removable* by fixing `_materialize()` in `pandera/api/narwhals/utils.py` to use `.collect()` for PySpark instead of `.execute()`. Once `_materialize()` is fixed, the `run_check` branch becomes dead code and can be deleted.

2. **`base.py:_concat_failure_cases` lines 53-68** — PySpark branch uses module-string sniffing (`type(item).__module__.startswith("pyspark")`) to route PySpark frames to `.union()`. Items in `failure_case_collection` are *native* frames (returned by `nw.to_native()` inside `_build_lazy_failure_case`). There is no purely narwhals-native way to eliminate the special-casing here without keeping items wrapped in narwhals through the full pipeline. The *meaningful improvement* is to keep items as narwhals-wrapped objects and use `nw.concat()` or type-dispatch on `nw.Implementation` (not module-string sniffing) for the concatenation step.

3. **`components.py:check_dtype` lines 276-309** — PySpark branch uses `str(pyspark_dtype) == str(schema.dtype)` because narwhals cross-engine dtype comparison (`narwhals_engine.Engine.dtype(pyspark_engine.Int)`) fails — there is no narwhals API that maps `T.IntegerType()` to `nw.Int32` without wrapping a DataFrame. The string comparison IS functionally correct. The improvement is to replace the frame-implementation probe (`check_obj.implementation in (PYSPARK, PYSPARK_CONNECT)`) with a *schema-driven* check (`isinstance(schema.dtype, pyspark_engine.DataType)`) so the branch is triggered by what the user configured rather than what backend is present.

4. **`container.py:validate()` lines 102-106, 231-246** — PySpark branch sets `check_obj.pandera.errors` and returns the original frame (the PySpark contract) instead of raising `SchemaErrors` (the standard narwhals contract). This protocol difference is *genuine* and cannot be eliminated. The feedback memory explicitly marks this as "arguably necessary." The success criterion (SC4) asks only that the inline `is_pyspark` block be extracted to a method on `DataFrameSchemaBackend` for readability.

**Primary recommendation:** Fix `_materialize()` first (enables SC1), then address `_concat_failure_cases` via narwhals-native concat (SC2), explain/restructure `check_dtype` dispatch (SC3), and extract the container method (SC4). Each is independent and can be planned as a separate task.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| PySpark scalar pass/fail extraction | Narwhals utils (`_materialize`) | base.py run_check | _materialize owns the backend-specific execution protocol; run_check should be generic |
| Failure case frame concatenation | base.py `_concat_failure_cases` | narwhals concat API | Must handle heterogeneous frame types (PySpark + polars scalars) |
| Dtype comparison for cross-engine schemas | components.py `check_dtype` | pyspark_engine | Schema configured with PySpark types requires PySpark-aware comparison |
| Error protocol dispatch (raise vs. return) | container.py `validate()` | — | PySpark contract is set-errors-return; all others raise |

## Standard Stack

This phase is pure internal refactoring. No new external packages are introduced.

### Core (existing)
| Library | Version | Purpose |
|---------|---------|---------|
| narwhals.stable.v1 | (installed) | All narwhals operations, `nw.Implementation` enum, `nw.concat` |
| pyspark.sql.types | (installed) | PySpark native type access via `.type` attribute on pyspark_engine dtypes |

### Relevant internal modules
| Module | Purpose |
|--------|---------|
| `pandera/api/narwhals/utils.py` | `_materialize()`, `_is_sql_lazy()`, `_SQL_LAZY_IMPLEMENTATIONS` |
| `pandera/backends/narwhals/base.py` | `NarwhalsSchemaBackend`, `_concat_failure_cases`, `run_check` |
| `pandera/backends/narwhals/components.py` | `ColumnBackend.check_dtype` |
| `pandera/backends/narwhals/container.py` | `DataFrameSchemaBackend.validate` |
| `pandera/engines/pyspark_engine.py` | `pyspark_engine.DataType` (for `isinstance` check in check_dtype) |

## Package Legitimacy Audit

> No new packages are installed in this phase. Audit not applicable.

## Architecture Patterns

### SC1: Fix `_materialize()` for PySpark

**Current state:** `_materialize()` in `pandera/api/narwhals/utils.py` handles two cases:
- `nw.LazyFrame` (polars) → `.collect()`
- `nw.DataFrame` (SQL-lazy, e.g. ibis) → `nw.to_native(frame).execute()`

PySpark `nw.DataFrame` is SQL-lazy (`_is_sql_lazy()` returns True) but its native object is `pyspark.sql.DataFrame` which has `.collect()`, not `.execute()`. The existing SQL-lazy branch calls `.execute()` which raises `AttributeError`.

**Fix:** Add a PySpark sub-branch inside `_materialize()`:

```python
# Source: direct codebase analysis of pandera/api/narwhals/utils.py
def _materialize(frame) -> nw.DataFrame:
    if isinstance(frame, nw.LazyFrame):
        return frame.collect()
    if isinstance(frame, nw.DataFrame) and _is_sql_lazy(frame):
        native = nw.to_native(frame)
        if frame.implementation in (
            nw.Implementation.PYSPARK,
            nw.Implementation.PYSPARK_CONNECT,
        ):
            # PySpark: .collect() returns a list of Row objects;
            # re-wrap via nw.from_native after collecting via pyarrow
            import pyarrow as pa
            rows = native.collect()
            arrow_table = pa.Table.from_pylist([r.asDict() for r in rows])
            return nw.from_native(arrow_table)
        return nw.from_native(native.execute())
    return frame
```

**Alternative (simpler):** Use `.toPandas()` + `nw.from_native` for the PySpark collect path. But pyarrow is already a transitive dependency. The simplest path is:

```python
if frame.implementation in (
    nw.Implementation.PYSPARK,
    nw.Implementation.PYSPARK_CONNECT,
):
    return nw.from_native(nw.to_native(frame).toPandas())
```

However, `.toPandas()` requires the `pandas` extra. The cleaner path uses the `.first()` approach only for the single-row scalar case (which is all `run_check` needs). For the full-frame case, `.collect()` + pyarrow is needed.

**Simpler still:** For `run_check`, the only materialization needed is a single scalar `bool` from a 1-row, 1-column aggregated frame. The existing `.first()` pattern achieves this correctly. The fix is to extend `_materialize()` to handle PySpark for the single-row scalar case and remove the `run_check` branch:

```python
# Extend _materialize in utils.py for PySpark single-row scalar case:
if frame.implementation in (
    nw.Implementation.PYSPARK,
    nw.Implementation.PYSPARK_CONNECT,
):
    native = nw.to_native(frame)
    row = native.first()
    if row is None:
        # Empty result — treat as "passed" (no failing rows)
        return nw.from_dict({col: [] for col in frame.collect_schema().names()},
                            backend="pyarrow")
    # Wrap single row as pyarrow for narwhals
    import pyarrow as pa
    return nw.from_native(pa.table({k: [v] for k, v in row.asDict().items()}))
```

This is the recommended approach: fix `_materialize()` to handle PySpark via `.first()` for the bounded single-row case, then `run_check` uses `_materialize(passed_lf)[CHECK_OUTPUT_KEY][0]` as before.

**Impact:** Once `_materialize()` handles PySpark, the `run_check` branch (lines 143-154) becomes dead code and can be deleted. `check_nullable` also uses `_materialize()` and would benefit automatically.

### SC2: Fix `_concat_failure_cases` — narwhals-native concat

**Current state:** Module-string sniffing detects PySpark items, drops scalar polars frames when PySpark frames are present.

**Root cause:** `_build_lazy_failure_case` returns `nw.to_native(enriched)` — a native frame. Once unwrapped, the type is invisible to narwhals APIs. The items in `failure_case_collection` are a mix of native PySpark DataFrames and `pl.DataFrame` objects.

**The mixing problem:** Scalar/schema-level failures go through `_build_scalar_failure_case` which always returns `pl.DataFrame`. Data-level failures go through `_build_lazy_failure_case` which returns `nw.to_native(enriched)` — for PySpark, this is a native `pyspark.sql.DataFrame`. Converting a polars scalar frame to PySpark requires a `SparkSession`, which `_concat_failure_cases` does not have access to.

**Approach A (keep items wrapped in narwhals):** Change `_build_lazy_failure_case` to return the narwhals-wrapped frame instead of calling `nw.to_native(enriched)`. Then `_concat_failure_cases` receives all items as `nw.DataFrame` / `nw.LazyFrame` objects and can dispatch on `item.implementation` rather than module strings. This is the architecturally clean approach.

```python
# _build_lazy_failure_case: remove nw.to_native(enriched), return narwhals frame
@staticmethod
def _build_lazy_failure_case(fc, err, check_identifier):
    # ... same enrichment logic ...
    return enriched  # nw.LazyFrame or nw.DataFrame (narwhals-wrapped)

# _concat_failure_cases: dispatch on nw.Implementation
def _concat_failure_cases(items: list) -> Any:
    if not items:
        return pl.DataFrame() if pl is not None else None
    # Separate narwhals-wrapped items from native items
    # All items from _build_*_failure_case are now narwhals-wrapped or pl.DataFrame (scalar)
    nw_items = [i for i in items if isinstance(i, (nw.DataFrame, nw.LazyFrame))]
    pl_items = [i for i in items if not isinstance(i, (nw.DataFrame, nw.LazyFrame))]
    # ...
```

**Approach B (pyarrow roundtrip for scalar frames):** When PySpark frames are present, convert scalar `pl.DataFrame` items to pyarrow, then use PySpark's `.createDataFrame(arrow_table)` to create PySpark-native frames for all items, then union. This requires a `SparkSession` reference from one of the PySpark items.

**Recommendation:** Approach A is cleaner and aligns with the narwhals philosophy. It requires adjusting `_build_lazy_failure_case` to return narwhals-wrapped frames and adjusting `failure_cases_metadata()` which currently expects native frames in `failure_case_collection`.

**Note on the silent-drop issue:** The success criterion says scalar frames should "no longer be silently dropped when PySpark frames are present." Under Approach A, all items are narwhals-wrapped and `nw.concat()` can combine them — but only if they share the same backend. A `nw.DataFrame` (PySpark) cannot be `nw.concat`'d with a `nw.DataFrame` (pyarrow/polars). So Approach B is needed for the mixed case. The pragmatic resolution: accept that scalar frames from schema-level checks are represented differently for PySpark (document this) OR implement Approach B with a SparkSession reference from the PySpark item.

**Alternative pragmatic fix:** Remove the `if not pyspark_items:` fallback (which currently returns `None` when there are only polars scalar frames but no PySpark frames — this is a bug since it should concat the polars items). Replace module-string sniffing with `nw.Implementation` check by keeping the lazy result wrapped in narwhals.

### SC3: Fix `check_dtype` — schema-driven dispatch

**Current state:** `check_dtype` in `components.py` detects PySpark by probing the *frame*:
```python
is_pyspark = check_obj.implementation in (nw.Implementation.PYSPARK, ...)
```
Then uses `str(pyspark_dtype) == str(schema.dtype)` for PySpark frames.

**Why narwhals-native dtype comparison fails:** `narwhals_engine.Engine.dtype(pyspark_engine.Int)` raises `TypeError` because the narwhals engine does not have `pyspark_engine.Int` in its equivalents registry. Cross-engine dtype comparison is not supported in the narwhals engine. This is a genuine narwhals/pandera limitation — not something fixable within the backend alone.

**Verified:**
- `narwhals_engine.Engine.dtype(nw.Int32)` returns `narwhals_engine.Int32` [VERIFIED: codebase]
- `narwhals_engine.Engine.dtype(pyspark_engine.Engine.dtype(T.IntegerType()))` raises `TypeError` [VERIFIED: codebase]
- `str(T.IntegerType()) == str(pyspark_engine.Engine.dtype(T.IntegerType()))` is `True` (both stringify to `'IntegerType()'`) [VERIFIED: codebase]
- `pyspark_engine.Int.check(narwhals_engine.Int32)` returns `False` [VERIFIED: codebase]

**The improvement (without narwhals changes):** Replace frame-implementation probe with schema-driven probe:

```python
# Instead of: is_pyspark = check_obj.implementation in (PYSPARK, PYSPARK_CONNECT)
# Use:
from pandera.engines import pyspark_engine as _pyspark_engine
uses_pyspark_dtype = isinstance(schema.dtype, _pyspark_engine.DataType)
```

This dispatches on *what the user configured* (a PySpark dtype in the schema) rather than *what backend is present* (the frame implementation). The two are correlated in practice but semantically different: a schema configured with PySpark types needs PySpark-native string comparison regardless of which backend executes it.

**The test workaround (lines 61-64 in test_pyspark_dtypes.py):** Creates an empty single-column DataFrame to avoid `STRUCT_ARRAY_LENGTH_MISMATCH`. This error occurs because `conftest.spark_df()` uses `verifySchema=False` to create DataFrames with 2-value rows against a 1-column schema. The narwhals backend triggers Spark execution (via `.first()` in `run_check`), which causes PySpark to validate row structure and fail. The workaround is a valid *test fixture correction*, not a backend workaround. It should be documented with a comment explaining the `verifySchema=False` mismatch. Removing it would require fixing `conftest.spark_df()` to not use `verifySchema=False`, but that fixture is shared across many tests and may break others.

**Recommended action:** Add a comment to lines 61-64 explaining:
- Why the empty df is created
- That the source df uses `verifySchema=False` with mismatched column count
- That Spark execution triggers schema validation which fails on mismatched data

### SC4: Extract PySpark error-setting to a method

**Current state:** `container.py:validate()` has inline `is_pyspark` blocks at two points:

```python
# Lines 231-236: error path
elif is_pyspark:
    error_dicts = error_handler.summarize(schema_name=schema.name)
    check_obj.pandera.errors = error_dicts
    return check_obj

# Lines 244-246: success path
if is_pyspark:
    check_obj.pandera.errors = {}
    return check_obj
```

**Why `is_pyspark` cannot be eliminated:** The PySpark backend contract (set `.pandera.errors` and return original frame) is genuinely different from the narwhals standard (raise `SchemaErrors`). `_is_sql_lazy()` would incorrectly catch ibis, which uses the raise protocol. There is no narwhals API to detect "uses pandera accessor pattern" vs "raises exception pattern."

**The feedback memory** explicitly flags this as "arguably necessary since it's a genuine protocol difference."

**Recommended refactoring (extract method):**

```python
# New method on DataFrameSchemaBackend:
def _handle_pyspark_validation_result(
    self,
    check_obj,
    error_handler,
    schema,
    has_errors: bool,
):
    """Set pandera.errors on a PySpark DataFrame and return it.
    
    Mirrors the native PySpark backend contract: validation errors are
    recorded on check_obj.pandera.errors rather than raised as SchemaErrors.
    This is a protocol difference specific to PySpark — all other backends
    raise SchemaErrors on failure.
    """
    if has_errors:
        error_dicts = error_handler.summarize(schema_name=schema.name)
        check_obj.pandera.errors = error_dicts
    else:
        check_obj.pandera.errors = {}
    return check_obj
```

Then `validate()` becomes:
```python
if is_pyspark:
    return self._handle_pyspark_validation_result(
        check_obj, error_handler, schema, has_errors=bool(error_handler.collected_errors)
    )
```

This satisfies SC4 (inline block extracted to method) while keeping `is_pyspark` as a boolean flag in `validate()`.

### Anti-Patterns to Avoid

- **Removing `is_pyspark` from `container.py`:** The protocol difference is real. Ibis does not use the accessor pattern; only PySpark does. Merging into `_is_sql_lazy()` would break ibis.
- **Using module-string sniffing in new code:** Already identified as the current violation. New code should use `nw.Implementation` enum checks or schema-driven type checks.
- **Calling `_materialize()` on full PySpark frames:** `_materialize()` for PySpark should only be used on single-aggregated-row results. The full-frame materialize path (`.collect()` on a large distributed frame) is never safe in the narwhals backend hot path.
- **Requiring SparkSession in `_concat_failure_cases`:** Getting `SparkSession` from a native frame in a utility function is a layering violation. Keep `_concat_failure_cases` free of SparkSession dependencies.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PySpark type-to-narwhals mapping | Manual type dispatch table | `str()` comparison on pyspark_engine dtypes | Narwhals internal mapping API is private/unstable; str comparison is semantically equivalent |
| DataFrame concatenation | Custom reduce loop | narwhals `nw.concat()` for same-backend items | Already handles same-backend concat; only diverge when backends differ |
| PySpark execution | Custom `.collect()` logic | Fix `_materialize()` in utils.py | Centralizes the execution protocol in one place |

## Runtime State Inventory

> Not applicable — this phase is a pure code refactoring with no rename/migration.

## Common Pitfalls

### Pitfall 1: `_materialize()` returns pyarrow-backed narwhals frame for PySpark
**What goes wrong:** After fixing `_materialize()` to return a pyarrow-backed `nw.DataFrame` for PySpark, code that does `_materialize(lf)[CHECK_OUTPUT_KEY][0]` now works. But code that does `_materialize(lf).to_native()` will get a `pyarrow.Table`, not a `pyspark.sql.DataFrame`.
**How to avoid:** Only use `_materialize()` for scalar extraction (`[col][0]`). Never use it when you need a native PySpark frame back.

### Pitfall 2: `nw.concat()` with mixed-backend items
**What goes wrong:** If `_build_lazy_failure_case` returns narwhals-wrapped items but some are PySpark-backed `nw.DataFrame` and others are pyarrow-backed `nw.DataFrame`, `nw.concat()` will fail with a backend mismatch error.
**How to avoid:** Ensure all items passed to `nw.concat()` are the same backend. Keep polars scalar frames as polars; concat PySpark frames separately.

### Pitfall 3: `isinstance(schema.dtype, pyspark_engine.DataType)` with None dtype
**What goes wrong:** If `schema.dtype is None`, `isinstance(None, pyspark_engine.DataType)` returns False, which is correct — but `check_dtype` already guards with `if schema.dtype is None: return [...]` at the top.
**How to avoid:** The existing None guard is in place; the isinstance check only runs after it.

### Pitfall 4: The `check_obj.implementation` check vs. `schema.dtype` isinstance check
**What goes wrong:** If a user configures a schema with `Column(T.IntegerType())` but validates a *pandas* frame (hypothetically via narwhals), the schema-driven check (`isinstance(schema.dtype, pyspark_engine.DataType)`) would trigger but `nw.to_native(check_obj).schema` would be a pandas schema, not a PySpark schema.
**Why it's OK:** PySpark schemas (`Column(T.IntegerType())`) are only used with PySpark DataFrames in practice. The correlation between `isinstance(schema.dtype, pyspark_engine.DataType)` and a PySpark frame is 100% in real usage.

### Pitfall 5: Silent scalar frame drop in `_concat_failure_cases`
**What goes wrong:** The current code drops non-PySpark items (scalar `pl.DataFrame` from `_build_scalar_failure_case`) when PySpark frames are present, meaning schema-level errors (dtype mismatches, wrong columns) have no data in the final `failure_cases`.
**Current behavior documented in the code.** SC2 says this should be fixed.
**The constraint:** Creating a PySpark DataFrame from scalar data requires `SparkSession`. The only available reference is from a PySpark item in the list (`.sparkSession` attribute on native frame). Getting SparkSession from `nw.to_native(pyspark_item)` is feasible but is a layering concern.

## Code Examples

### Current `_materialize()` for PySpark (broken)
```python
# Source: pandera/api/narwhals/utils.py (current state)
def _materialize(frame) -> nw.DataFrame:
    if isinstance(frame, nw.LazyFrame):
        return frame.collect()
    if isinstance(frame, nw.DataFrame) and _is_sql_lazy(frame):
        # BUG: PySpark nw.DataFrame native is pyspark.sql.DataFrame
        # which has .collect() not .execute() — this raises AttributeError
        return nw.from_native(nw.to_native(frame).execute())
    return frame
```

### Proposed `_materialize()` fix
```python
# Source: direct codebase analysis
def _materialize(frame) -> nw.DataFrame:
    if isinstance(frame, nw.LazyFrame):
        return frame.collect()
    if isinstance(frame, nw.DataFrame) and _is_sql_lazy(frame):
        if frame.implementation in (
            nw.Implementation.PYSPARK,
            nw.Implementation.PYSPARK_CONNECT,
        ):
            # PySpark: use .first() for the single-row scalar case
            # (the only context _materialize is used for PySpark in this backend).
            # .first() triggers bounded execution — only one row is collected.
            native = nw.to_native(frame)
            row = native.first()
            if row is None:
                # Empty result frame — build from schema with 0 rows
                import pyarrow as pa
                schema_names = frame.collect_schema().names()
                return nw.from_native(pa.table({k: [] for k in schema_names}))
            import pyarrow as pa
            return nw.from_native(pa.table({k: [v] for k, v in row.asDict().items()}))
        return nw.from_native(nw.to_native(frame).execute())
    return frame
```

### Current `check_dtype` PySpark branch (module-string sniff)
```python
# Source: pandera/backends/narwhals/components.py lines 276-309
is_pyspark = check_obj.implementation in (
    nw.Implementation.PYSPARK,
    nw.Implementation.PYSPARK_CONNECT,
)
native_pyspark_schema = (
    nw.to_native(check_obj).schema if is_pyspark else None
)
for column, nw_dtype in zip(schema_obj.names(), schema_obj.dtypes()):
    if is_pyspark:
        pyspark_dtype = native_pyspark_schema[column].dataType
        pyspark_dtype_str = str(pyspark_dtype)
        passed = pyspark_dtype_str == str(schema.dtype)
        # ...
```

### Proposed `check_dtype` schema-driven dispatch
```python
# Replace frame-implementation probe with schema-driven probe:
from pandera.engines import pyspark_engine as _pyspark_engine
uses_pyspark_dtype = isinstance(schema.dtype, _pyspark_engine.DataType)

native_obj = nw.to_native(check_obj) if uses_pyspark_dtype else None
native_schema = native_obj.schema if uses_pyspark_dtype else None

for column, nw_dtype in zip(schema_obj.names(), schema_obj.dtypes()):
    if uses_pyspark_dtype:
        pyspark_dtype = native_schema[column].dataType
        pyspark_dtype_str = str(pyspark_dtype)
        passed = pyspark_dtype_str == str(schema.dtype)
        # ... rest of PySpark path unchanged
```

### Extract container.py PySpark error-setting to method
```python
# Source: pandera/backends/narwhals/container.py proposed refactoring
def _handle_pyspark_validation_result(
    self,
    check_obj,
    error_handler,
    schema,
    has_errors: bool,
):
    """Record validation outcome on PySpark DataFrame via pandera accessor.

    PySpark uses a different validation contract from other backends:
    errors are set on check_obj.pandera.errors and the original frame is
    returned, rather than raising SchemaErrors. This matches the native
    PySpark backend contract and is required for the existing PySpark test
    suite to pass.
    """
    if has_errors:
        error_dicts = error_handler.summarize(schema_name=schema.name)
        check_obj.pandera.errors = error_dicts
    else:
        check_obj.pandera.errors = {}
    return check_obj
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `pl.concat()` for all backends | Module-string-sniffing PySpark branch | Phase 02-04 (commit b848d8bd) | Fixed crash but introduced dispatch violation |
| `_materialize()` (ibis path) | Native `.first()` branch in `run_check` | Phase 02-04 | Fixed STRUCT_ARRAY_LENGTH_MISMATCH but bypassed `_materialize()` |
| Narwhals dtype comparison | `str(pyspark_dtype) == str(schema.dtype)` | Phase 02-04 | Fixed dtype comparison failures for PySpark |
| Inline PySpark error-setting | Still inline (no change yet) | Phase 02-04 | 4th dispatch violation identified in PR review |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `pyspark.sql.DataFrame.first()` returns a `Row` or `None` (never raises on empty) | SC1 fix | `_materialize()` fix may not work as expected |
| A2 | `row.asDict()` on a PySpark `Row` always produces a dict of the named columns | SC1 fix | pyarrow table construction fails |
| A3 | `isinstance(schema.dtype, pyspark_engine.DataType)` correctly identifies PySpark-configured schemas in all test cases | SC3 fix | check_dtype silently misroutes for edge cases |

## Open Questions

1. **SC2 — scalar frame silent drop**
   - What we know: scalar `pl.DataFrame` items are dropped when PySpark frames are present because we can't concat across backends
   - What's unclear: whether any test actually catches this silent drop (i.e., does any PySpark test assert on `failure_cases` content for schema-level errors?)
   - Recommendation: Check test corpus; if no test catches the drop, document the behavior and add a warning comment. If tests do catch it, a SparkSession-based pyarrow roundtrip is needed.

2. **SC1 — pyarrow vs. toPandas() for PySpark materialize**
   - What we know: both approaches work; toPandas() requires pandas; pyarrow requires constructing a table from Row.asDict()
   - What's unclear: whether pyarrow is always available in PySpark environments
   - Recommendation: Use pyarrow (already a narwhals dependency); avoid toPandas() since pandas is optional.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| narwhals | All backend code | Yes | (installed) | — |
| pyspark | PySpark path | Yes (test env) | (installed) | Tests skip if not available |
| pyarrow | `_materialize()` PySpark fix | Yes | (narwhals dep) | — |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | pyproject.toml (pytest section) |
| Quick run command | `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/test_pyspark_error.py tests/pyspark/test_pyspark_dtypes.py -x -q` |
| Full suite command | `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/ --ignore=tests/pyspark/test_schemas_on_pyspark_pandas.py -k "spark and not spark_connect" -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ARCH-01 | `run_check` no longer has `Implementation in (PYSPARK, ...)` check | unit | `pytest tests/narwhals/test_container.py tests/pyspark/test_pyspark_error.py -x` | Yes |
| ARCH-02 | `_concat_failure_cases` uses narwhals-native concat | unit | `pytest tests/pyspark/test_pyspark_error.py::test_pyspark_check_eq -x` | Yes |
| ARCH-03 | `check_dtype` uses schema-driven (not frame-driven) dispatch | unit | `pytest tests/pyspark/test_pyspark_dtypes.py -x` | Yes |
| ARCH-04 | PySpark error-setting extracted to method | unit | `pytest tests/pyspark/test_pyspark_error.py tests/pyspark/test_pyspark_check.py -x` | Yes |

### Wave 0 Gaps
- [ ] `tests/narwhals/test_container.py` — add a test asserting `_materialize()` works for a PySpark-backed 1-row `nw.DataFrame` (may require mocking or a PySpark fixture)
- [ ] `tests/narwhals/test_components.py` — add a test for `check_dtype` with a schema using `pyspark_engine.DataType` that the narwhals engine cannot directly resolve

*(If PySpark tests are only run in the `pyspark` nox session, both of these can be in `tests/pyspark/` instead.)*

## Security Domain

> Omitted — this is an internal refactoring with no user-facing inputs, authentication, or data storage changes.

## Sources

### Primary (HIGH confidence)
- `pandera/backends/narwhals/base.py` — direct codebase read (all four dispatch violations)
- `pandera/backends/narwhals/components.py` — direct codebase read (check_dtype)
- `pandera/backends/narwhals/container.py` — direct codebase read (validate error protocol)
- `pandera/api/narwhals/utils.py` — direct codebase read (_materialize, _is_sql_lazy)
- `pandera/engines/narwhals_engine.py` — direct codebase read (Engine.dtype fallback chain)
- `pandera/engines/pyspark_engine.py` — runtime exploration (pyspark_engine.Int.check behavior)
- `.planning/phases/02-test-coverage-and-ci/02-03-TRIAGE.md` — Phase 2 triage (root cause documentation)
- `.planning/phases/02-test-coverage-and-ci/02-04-SUMMARY.md` — Phase 2 fix summary
- `.claude/.../memory/feedback_backend_specific_code.md` — feedback (avoid is_pyspark branches)

### Secondary (MEDIUM confidence)
- narwhals._spark_like.utils source — confirms PySpark→narwhals type mapping exists internally but is not public API
- Runtime exploration of `narwhals_engine.Engine.dtype()` with various inputs — confirms cross-engine dtype comparison fails

## Metadata

**Confidence breakdown:**
- SC1 (run_check / _materialize): HIGH — the failure mode and fix are both verified in the codebase
- SC2 (_concat_failure_cases): MEDIUM — the backend-mixing constraint is verified; the exact fix approach requires a design decision
- SC3 (check_dtype): HIGH — the str comparison is verified correct; schema-driven dispatch is a straightforward refactoring
- SC4 (container.py extract): HIGH — mechanical extract-method refactoring with no logic changes

**Research date:** 2026-05-25
**Valid until:** 2026-06-25 (stable internal codebase; narwhals API changes are the main risk)
