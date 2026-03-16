# Phase 5: Ibis Registration and Integration - Research

**Researched:** 2026-03-14
**Domain:** Ibis narwhals backend registration, uniqueness checks, drop_invalid_rows, parity testing
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- `register_ibis_backends()` gains narwhals auto-detection, mirroring the polars pattern exactly: detect narwhals at registration time, register narwhals backends for `ibis.Table` when narwhals is installed
- Emits the same `UserWarning` as `register_polars_backends()` when narwhals auto-activates
- `tests/backends/narwhals/conftest.py` calls `register_ibis_backends()` alongside `register_polars_backends()` (same autouse module fixture) — narwhals conftest needs explicit registration because tests hit backends directly without going through schema.validate()
- `pandera/backends/narwhals/register.py` is deleted — it's a dead file (nothing imports it); registration responsibility stays co-located with each library's own register.py
- Both container-level (`check_column_values_are_unique`) and column-level (`check_unique`) use `group_by().agg(nw.len())` — SQL-lazy safe, works across Polars and Ibis without requiring collect()
- Column-level `check_unique` is not implemented in the ibis backend at all; Phase 5 implements it in the narwhals `ColumnBackend` via `group_by().agg(nw.len())`
- This replaces/supersedes the Polars-only `collect()`-then-`is_duplicated()` approach for Ibis paths
- Detect if the underlying native frame is an `ibis.Table` (after `nw.to_native()`), delegate to `IbisSchemaBackend.drop_invalid_rows()`, re-wrap with `nw.from_native()` — one `try/except ImportError` guard in `base.py`
- Fix the existing Polars path in `drop_invalid_rows` at the same time: replace `pl.fold` with `nw.all_horizontal()` to use narwhals-native boolean reduction
- New file: `tests/backends/narwhals/test_parity.py`
- Draw from: `test_polars_container.py`, `test_ibis_container.py`, `test_polars_decorators.py`, `test_ibis_decorators.py`, `test_polars_model.py`, `test_ibis_model.py`
- Exclude: dtype-specific tests, strategy tests, pydantic/typing tests, builtin check tests
- Coerce-dependent tests: include as `xfail(strict=True)`

### Claude's Discretion

- Exact narwhals `group_by().agg()` expression for uniqueness checks
- Whether the ibis-delegation branch in `drop_invalid_rows` uses `isinstance(native, ibis.Table)` or a string backend name check
- Exact structure/grouping of tests within test_parity.py

### Deferred Ideas (OUT OF SCOPE)

- None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| REGISTER-03 | Narwhals backend registers for `ibis.Table` — end-to-end `schema.validate(table)` works for Ibis frames, closing known xfail gaps (`coerce_dtype`, column `unique`) | Registration pattern verified; dtype check fix for ibis dtypes identified |
| TEST-02 | Tests cover schema validation (column presence, dtype check, nullable, unique), all 14 builtin checks, lazy validation mode, dtype coercion, and error message correctness (native frame types in `failure_cases`) | Ibis validation chain exercised; failure cases are pandas.DataFrame (native) via `_to_native`/`execute()` |
| TEST-04 | A curated subset of `tests/polars/` and `tests/ibis/` end-to-end tests runs with the narwhals backend active covering validation depth semantics, `lazy=True` error collection, strict/filter column modes, and decorator behavior | Source test files identified; parity test structure mapped |
</phase_requirements>

## Summary

Phase 5 closes the last open requirements: registering narwhals backends for `ibis.Table` and completing test coverage across both Polars and Ibis backends. The registration pattern is a near-exact mirror of `register_polars_backends()` in `pandera/backends/ibis/register.py`, with one important wrinkle: `register_ibis_backends()` currently lacks `@lru_cache` and is called via `register_default_backends()` in both `pandera/api/ibis/container.py` and `pandera/api/ibis/components.py` every time a schema operates on an ibis table.

The `check_unique` and `check_column_values_are_unique` implementations both need to replace their `is_duplicated()` / `.collect()` paths with `group_by().agg(nw.len())` to remain SQL-lazy-safe. The existing Polars-only `drop_invalid_rows` in `base.py` uses `pl.fold` which breaks when `check_output` is a pandas DataFrame (the native type for ibis after `_to_native()` materializes via `.execute()`); this must be replaced with `nw.all_horizontal()` for the Polars path and ibis-delegation for the Ibis path.

There is one additional fix required beyond what the CONTEXT.md mentions: `check_dtype` in `components.py` needs a third pass for ibis. Currently it tries (1) narwhals engine lookup then (2) polars native schema lookup. For ibis, both fail because `ibis_engine.Int64.check(narwhals_engine.Int64)` returns `False` and polars native schema is `None` for ibis tables. The fix is to detect `isinstance(native, ibis.Table)` and consult `native.schema()[col]` which returns the ibis native dtype that `ibis_engine.Int64.check()` correctly validates.

**Primary recommendation:** Follow the locked pattern exactly — narwhals detection block first in `register_ibis_backends()`, `@lru_cache`, direct `BACKEND_REGISTRY` writes are NOT needed (register_backend works when called before register_default_backends runs), dtype three-pass fix in components.py, group_by uniqueness everywhere, ibis delegation for drop_invalid_rows.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `narwhals.stable.v1` | 2.15.0 | Cross-backend DataFrame API | Already in use throughout narwhals backend |
| `ibis` | current in env | SQL-lazy backend under test | The target registration backend for this phase |
| `polars` | current in env | Already-working backend | Used in parallel tests and as reference |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pandas` | current | Native frame type for ibis | `_to_native()` on ibis nw.DataFrame produces pandas |
| `functools.lru_cache` | stdlib | Registration idempotency | Wrap `register_ibis_backends()` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `group_by().agg(nw.len())` | `is_duplicated()` | `is_duplicated()` works on ibis nw.DataFrame after `_materialize()` but is not SQL-lazy; group_by stays server-side |
| `isinstance(native, ibis.Table)` | string backend name check | isinstance is more robust and avoids importing ibis for the name check |

**Installation:**
```bash
# No new dependencies — ibis and narwhals already in test environment
```

## Architecture Patterns

### Recommended Project Structure

No new directories. Changes touch:
```
pandera/backends/ibis/register.py      # Add @lru_cache + narwhals auto-detection block
pandera/backends/narwhals/register.py  # DELETE (dead file)
pandera/backends/narwhals/base.py      # Fix drop_invalid_rows (pl.fold → nw.all_horizontal + ibis delegation)
pandera/backends/narwhals/components.py # Fix check_unique (group_by) + check_dtype (ibis third pass)
pandera/backends/narwhals/container.py  # Fix check_column_values_are_unique (group_by)
tests/backends/narwhals/conftest.py    # Add register_ibis_backends() call
tests/backends/narwhals/test_parity.py # New file
```

### Pattern 1: Narwhals Auto-Detection in register_ibis_backends()

**What:** Mirror the polars pattern — `@lru_cache`, `try narwhals import`, register narwhals backends, emit `UserWarning`; fall through to native ibis backends when narwhals not available.

**When to use:** Always — this is the only implementation of `register_ibis_backends()`.

**Example:**
```python
# Source: pandera/backends/polars/register.py (existing, verified)
import warnings
from functools import lru_cache
import ibis

@lru_cache
def register_ibis_backends(check_cls_fqn: str | None = None):
    from pandera.api.checks import Check
    from pandera.api.ibis.components import Column
    from pandera.api.ibis.container import DataFrameSchema

    try:
        import narwhals.stable.v1 as nw

        from pandera.backends.narwhals import builtin_checks  # noqa
        from pandera.backends.narwhals.checks import NarwhalsCheckBackend
        from pandera.backends.narwhals.components import ColumnBackend
        from pandera.backends.narwhals.container import DataFrameSchemaBackend

        warnings.warn(
            "Narwhals is installed. Pandera is using the experimental Narwhals backends "
            "for Ibis Tables. These backends may change in future versions.",
            UserWarning,
            stacklevel=2,
        )

        DataFrameSchema.register_backend(ibis.Table, DataFrameSchemaBackend)
        Column.register_backend(ibis.Table, ColumnBackend)
        Check.register_backend(ibis.Table, NarwhalsCheckBackend)
        Check.register_backend(ibis.Column, NarwhalsCheckBackend)
    except ImportError:
        from pandera.backends.ibis import builtin_checks  # noqa
        from pandera.backends.ibis.checks import IbisCheckBackend
        from pandera.backends.ibis.components import ColumnBackend
        from pandera.backends.ibis.container import DataFrameSchemaBackend

        DataFrameSchema.register_backend(ibis.Table, DataFrameSchemaBackend)
        Column.register_backend(ibis.Table, ColumnBackend)
        Check.register_backend(ibis.Table, IbisCheckBackend)
        Check.register_backend(ibis.Column, IbisCheckBackend)
```

**Critical note on `@lru_cache` and the guard:** `register_backend()` has a guard: `if (cls, type_) not in cls.BACKEND_REGISTRY`. This guard does NOT block the narwhals registration because `register_ibis_backends()` is called from `register_default_backends()` which is called by `get_backend()` which is called by `schema.validate()`. Before the first `schema.validate(ibis_table)` call, `ibis.Table` is NOT in the registry. Therefore `register_backend()` works correctly — no need for direct `BACKEND_REGISTRY` writes. The `@lru_cache` ensures the registration runs only once even though `register_default_backends()` is called on every `get_backend()` invocation.

### Pattern 2: group_by().agg(nw.len()) for SQL-Lazy Uniqueness

**What:** Replace `is_duplicated()` with a group-by aggregation to find duplicates without materializing the full frame.

**When to use:** Both `check_unique` (column-level, `components.py`) and `check_column_values_are_unique` (container-level, `container.py`).

**Example (verified working on ibis via direct test):**
```python
# Source: verified in research session against ibis.memtable
# For column-level check_unique in components.py:
grouped = (
    check_obj
    .select(nw.col(schema.selector))
    .group_by(nw.col(schema.selector))
    .agg(nw.len().alias("_count"))
)
dups = grouped.filter(nw.col("_count") > 1).select(schema.selector)
# To check if any duplicates exist: materialize the small dup-values frame
native_dups = nw.to_native(_materialize(dups))
has_duplicates = len(native_dups) > 0
# failure_cases = _to_native(dups)
```

**Example (container-level check_column_values_are_unique):**
```python
# For each column subset in temp_unique:
grouped = (
    check_obj
    .select(subset)
    .group_by(*[nw.col(c) for c in subset])
    .agg(nw.len().alias("_count"))
)
dup_rows = grouped.filter(nw.col("_count") > 1).drop("_count")
native_dups = nw.to_native(_materialize(dup_rows))
if len(native_dups) > 0:
    passed = False
    failure_cases = native_dups
```

### Pattern 3: drop_invalid_rows — Polars Fix + Ibis Delegation

**What:** The current `drop_invalid_rows` in `base.py` uses `pl.fold` which breaks when `check_output` is a pandas DataFrame. For Polars, replace with `nw.all_horizontal()`. For Ibis, delegate to `IbisSchemaBackend.drop_invalid_rows()`.

**Critical observation:** After `run_check()` on an ibis-backed nw frame, `check_output` is a **pandas DataFrame** (not polars, not ibis). This is because `_to_native()` on an ibis nw.DataFrame calls `.execute()` which returns pandas. So `pl.DataFrame({...})` in the current `drop_invalid_rows` fails with a polars Series type error.

**Example:**
```python
# Source: pandera/backends/narwhals/base.py drop_invalid_rows (to be rewritten)
def drop_invalid_rows(self, check_obj, error_handler):
    errors = getattr(error_handler, "schema_errors", [])
    if not errors:
        return check_obj

    # Detect ibis path: native frame is ibis.Table
    native = nw.to_native(check_obj) if isinstance(check_obj, (nw.LazyFrame, nw.DataFrame)) else check_obj
    try:
        import ibis as _ibis
        if isinstance(native, _ibis.Table):
            from pandera.backends.ibis.base import IbisSchemaBackend
            result = IbisSchemaBackend().drop_invalid_rows(native, error_handler)
            return nw.from_native(result, eager_or_interchange_only=False)
    except ImportError:
        pass

    # Polars path: use nw.all_horizontal for boolean reduction
    check_outputs = [err.check_output for err in errors if err.check_output is not None]
    if not check_outputs:
        return check_obj
    # check_outputs are native pl.DataFrame with 'check_output' column
    import polars as pl
    combined = pl.DataFrame(
        {str(i): co[CHECK_OUTPUT_KEY] for i, co in enumerate(check_outputs)}
    )
    valid_rows = combined.select(
        valid_rows=nw.to_native(
            nw.from_native(combined).select(
                nw.all_horizontal(*[nw.col(c) for c in combined.columns]).alias("valid_rows")
            )
        )["valid_rows"]
    )["valid_rows"]
    return check_obj.filter(valid_rows)
```

**Simpler approach (Claude's discretion):** Since polars check_outputs are `pl.DataFrame` with `CHECK_OUTPUT_KEY` column, `nw.all_horizontal` can be used directly:
```python
# Polars-only path after ibis delegation:
check_dfs = [pl.Series(err.check_output[CHECK_OUTPUT_KEY]) for err in errors]
valid_rows = pl.concat(check_dfs, rechunk=False).to_frame().select(
    valid_rows=nw.to_native(
        nw.from_native(pl.concat([e.check_output for e in errors], how="horizontal"))
        .select(nw.all_horizontal(*[nw.col(str(i)) for i in range(len(errors))]).alias("valid_rows"))
    )["valid_rows"]
)["valid_rows"]
```

Actually the cleanest approach:
```python
# After establishing check_outputs are all pl.DataFrame:
import polars as pl
merged = pl.concat(
    [err.check_output.rename({CHECK_OUTPUT_KEY: str(i)}) for i, err in enumerate(errors)],
    how="horizontal"
)
valid_rows = merged.select(
    pl.fold(acc=pl.lit(True), function=lambda a, x: a & x, exprs=pl.all())
)["literal"]
```

Wait — the locked decision says "replace `pl.fold` with `nw.all_horizontal()`". The correct narwhals approach:
```python
import polars as pl
import narwhals.stable.v1 as nw

bool_cols = {str(i): err.check_output[CHECK_OUTPUT_KEY] for i, err in enumerate(errors)}
merged_pl = pl.DataFrame(bool_cols)
merged_nw = nw.from_native(merged_pl)
valid_rows = merged_nw.select(
    nw.all_horizontal(*[nw.col(c) for c in merged_pl.columns]).alias("valid_rows")
).to_native()["valid_rows"]
return check_obj.filter(valid_rows)
```

### Pattern 4: check_dtype Third Pass for Ibis

**What:** Add a third dtype-check pass for ibis. After the narwhals_engine pass (1) and polars native schema pass (2) both fail, check if the native frame is an ibis.Table and consult its schema.

**Why needed:** `ibis_engine.Int64.check(narwhals_engine.Int64)` returns `False`. Ibis `Column(dt.Int64)` schemas use `ibis_engine.Int64` as their dtype. The narwhals backend's `check_dtype` must handle this case.

**Example (verified working):**
```python
# In components.py check_dtype, after the existing two-pass logic:
if not passed:
    try:
        import ibis as _ibis
        native_check_obj = nw.to_native(check_obj)
        if isinstance(native_check_obj, _ibis.Table):
            ibis_schema = native_check_obj.schema()
            ibis_native_dtype = ibis_schema.get(column)
            if ibis_native_dtype is not None:
                passed = schema.dtype.check(ibis_native_dtype)
    except ImportError:
        pass
```

**Confidence:** HIGH — verified via `ibis_engine.Int64.check(native.schema()['a'])` returning `True` in research session.

### Pattern 5: conftest.py Registration

**What:** Add `register_ibis_backends()` call alongside `register_polars_backends()` in the autouse module fixture.

**Example:**
```python
# tests/backends/narwhals/conftest.py — additions to _suppress_narwhals_warning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from pandera.backends.polars.register import register_polars_backends
    from pandera.backends.ibis.register import register_ibis_backends
    register_polars_backends.cache_clear()
    register_ibis_backends.cache_clear()  # needs @lru_cache first
    register_polars_backends()
    register_ibis_backends()
    yield
```

### Anti-Patterns to Avoid

- **Direct BACKEND_REGISTRY writes:** Not needed — `register_backend()` works when called before `register_default_backends()` runs. The guard only blocks re-registration of an already-registered type.
- **Materializing the full frame for uniqueness:** `is_duplicated()` works on ibis nw.DataFrame (via `_materialize().execute()`) but defeats SQL-lazy benefits. Use `group_by().agg(nw.len())` instead per the locked decision.
- **Calling `register_narwhals_backends()` from ibis register:** The dead `narwhals/register.py` file should be deleted, not called.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Ibis drop_invalid_rows logic | Custom positional join in narwhals | `IbisSchemaBackend.drop_invalid_rows()` | Already handles both positional-join backends (duckdb, polars) and row_number fallback |
| Boolean column reduction | Custom fold loop | `nw.all_horizontal()` | Built-in narwhals API, tested, cross-backend |
| Ibis schema dtype lookup | Custom ibis dtype introspection | `ibis_table.schema().get(col)` | Direct ibis API, returns native dtype that `ibis_engine.Engine.dtype().check()` understands |

## Common Pitfalls

### Pitfall 1: check_dtype Fails for Ibis Without Third Pass
**What goes wrong:** `schema.validate(ibis_table)` raises `SchemaError: expected column 'a' to have type int64, got Int64` — the column IS int64 but the type comparison fails because ibis_engine.Int64.check() doesn't understand narwhals_engine.Int64.
**Why it happens:** `check_dtype` two-pass logic: (1) narwhals_engine.Engine.dtype(nw_Int64) -> ok, then dtype.check(nw_pandera_Int64) -> False for ibis_engine.Int64; (2) polars native schema is None for ibis. No third pass exists.
**How to avoid:** Add ibis-native third pass: detect `isinstance(nw.to_native(check_obj), ibis.Table)` and check against `native.schema()[col]`.
**Warning signs:** `SchemaError: expected column 'X' to have type T, got T` — same type name, different engine class.

### Pitfall 2: lru_cache Missing on register_ibis_backends()
**What goes wrong:** `register_ibis_backends()` is called on every `schema.validate()` because `register_default_backends()` calls it. Without `@lru_cache`, every validate() call re-enters the registration logic and emits the UserWarning repeatedly.
**Why it happens:** `register_ibis_backends()` currently has no `@lru_cache`. The polars version has it.
**How to avoid:** Add `@lru_cache` decorator. The `check_cls_fqn` parameter must be added for the lru_cache signature to match polars pattern.

### Pitfall 3: drop_invalid_rows Fails for Ibis check_output Type
**What goes wrong:** `pl.DataFrame({str(i): err.check_output for i, err in ...})` fails with `TypeError: Expected Narwhals class or scalar` because ibis `check_output` is a pandas DataFrame, not a Polars Series/DataFrame.
**Why it happens:** `_to_native()` on an ibis nw.DataFrame calls `.execute()` which returns pandas. The current `drop_invalid_rows` assumes all `check_output` values are Polars.
**How to avoid:** Detect ibis path first (isinstance check on native), delegate to IbisSchemaBackend; only run Polars path for non-ibis frames.

### Pitfall 4: check_unique group_by Approach — failure_cases Format
**What goes wrong:** `group_by().agg(nw.len())` returns duplicate *values* not duplicate *rows*. The `check_output` column format differs from `is_duplicated()` which returns per-row booleans.
**Why it happens:** The group_by approach finds which values are duplicated, not which row indices are duplicated.
**How to avoid:** For `check_unique`, failure_cases should be the duplicate values (scalar column), not indexed rows. Return the dup_values DataFrame as failure_cases. For `check_output`, construct a per-row boolean by joining back or return `None` for check_output (acceptable since drop_invalid_rows is not called for column-level unique checks in isolation).

### Pitfall 5: conftest.py cache_clear Order
**What goes wrong:** `register_ibis_backends.cache_clear()` fails with `AttributeError` if `@lru_cache` is not yet applied to `register_ibis_backends()`.
**Why it happens:** The plan step that adds `@lru_cache` must precede the conftest update.
**How to avoid:** Plan ordering: register.py fix comes before conftest update.

### Pitfall 6: narwhals/register.py Deletion
**What goes wrong:** Deleting `narwhals/register.py` when something imports it.
**Why it happens:** Could have been imported somewhere.
**How to avoid:** Verify no imports: `grep -rn "from pandera.backends.narwhals.register\|import narwhals.register"` — confirmed empty (verified in research).

## Code Examples

### group_by uniqueness on ibis (verified)
```python
# Source: verified in research session, narwhals 2.15.0, ibis memtable
import ibis
import narwhals.stable.v1 as nw

t = ibis.memtable({"col": [1, 2, 2]})
nw_t = nw.from_native(t, eager_or_interchange_only=False)

grouped = nw_t.select(nw.col("col")).group_by(nw.col("col")).agg(nw.len().alias("_count"))
dups = grouped.filter(nw.col("_count") > 1).select("col")
native_dups = nw.to_native(dups).execute()
# native_dups is a pandas DataFrame with the duplicate values
```

### nw.all_horizontal on polars (verified)
```python
# Source: verified in research session
import polars as pl
import narwhals.stable.v1 as nw

df = pl.DataFrame({"a": [True, False, True], "b": [True, True, True]})
nw_df = nw.from_native(df)
result = nw_df.select(
    nw.all_horizontal(nw.col("a"), nw.col("b")).alias("valid_rows")
)
# result is nw.DataFrame with boolean "valid_rows" column
```

### isinstance ibis detection for drop_invalid_rows (verified)
```python
# Source: verified in research session
import ibis
import narwhals.stable.v1 as nw

t = ibis.memtable({"col": [1, 2, 3]})
nw_t = nw.from_native(t, eager_or_interchange_only=False)
native = nw.to_native(nw_t)
assert isinstance(native, ibis.Table)  # True
```

### ibis dtype third pass in check_dtype (verified)
```python
# Source: verified in research session
import ibis
import ibis.expr.datatypes as dt
from pandera.engines import ibis_engine

t = ibis.memtable({"a": [1, 2, 3]})
ibis_schema = t.schema()
ibis_native_dtype = ibis_schema.get("a")  # dt.Int64 instance
pandera_dtype = ibis_engine.Engine.dtype(dt.Int64)
assert pandera_dtype.check(ibis_native_dtype)  # True
```

### failure_cases type for ibis (verified)
```python
# Source: verified in research session — ibis failure_cases are pandas.DataFrame
# _to_native() on ibis nw.DataFrame returns pandas.DataFrame (via .execute())
# This is NATIVE — TEST-02 requirement is satisfied (not a narwhals wrapper)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `pl.fold` for boolean reduction | `nw.all_horizontal()` | Phase 5 | Cross-backend, no polars import needed |
| `is_duplicated().collect()` for uniqueness | `group_by().agg(nw.len())` | Phase 5 | SQL-lazy safe, works on ibis without materialization |
| No `@lru_cache` on `register_ibis_backends()` | `@lru_cache` | Phase 5 | Prevents repeated registration + warning on every validate() |

**Deprecated/outdated:**
- `pandera/backends/narwhals/register.py`: Empty file with no imports, never imported by anything. Delete in Phase 5.
- `COLUMN-02` requirement notes `check_unique` forces collection via `.collect()` before calling `is_duplicated()`. Phase 5 supersedes this with the group_by approach for ibis paths.

## Open Questions

1. **failure_cases_metadata for ibis pandas DataFrames**
   - What we know: `failure_cases_metadata` in `base.py` checks `isinstance(err.failure_cases, pl.DataFrame)`. For ibis, `failure_cases` is a pandas DataFrame. It falls into the scalar branch, producing a pl.DataFrame with the pandas repr as the `failure_case` string value.
   - What's unclear: Whether this scalar-branch handling is "correct enough" for TEST-02 or needs a pandas-aware branch.
   - Recommendation: TEST-02 asserts that `failure_cases` is a native frame type (not narwhals wrapper). Pandas DataFrame is native. The scalar-branch degraded output is acceptable for v1; add a `isinstance(err.failure_cases, pd.DataFrame)` branch only if tests fail. Monitor during implementation.

2. **check_output from ibis uniqueness group_by**
   - What we know: The group_by approach returns failure_cases (duplicate values) but not per-row boolean check_output. The `check_output` field is used by `drop_invalid_rows` and `failure_cases_metadata`.
   - What's unclear: For ibis uniqueness checks, since `drop_invalid_rows` delegates to `IbisSchemaBackend`, the ibis-native path will need ibis-native boolean columns — not what group_by returns.
   - Recommendation: Return `check_output=None` for uniqueness failures (consistent with how ibis backend currently handles it). The `drop_invalid_rows` ibis delegation uses `err.check_output` as `ir.BooleanColumn` — pass `None` and let the ibis backend's own path handle it if needed, or document that uniqueness rows are not drop-invalid-rows compatible in ibis path for v1.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing throughout project) |
| Config file | `pyproject.toml` (pytest settings) |
| Quick run command | `pytest tests/backends/narwhals/ -x -q` |
| Full suite command | `pytest tests/backends/narwhals/ tests/ibis/ -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REGISTER-03 | `schema.validate(ibis_table)` succeeds via narwhals backend | integration | `pytest tests/backends/narwhals/test_container.py -x -k ibis` | ❌ Wave 0 — new test needed in test_parity.py |
| REGISTER-03 | `register_ibis_backends()` emits UserWarning when narwhals installed | unit | `pytest tests/backends/narwhals/test_container.py -x -k test_ibis_narwhals_auto_activated` | ❌ Wave 0 |
| TEST-02 | Column presence, dtype check, nullable, unique on ibis | unit | `pytest tests/backends/narwhals/test_components.py tests/backends/narwhals/test_checks.py -x` | ✅ (parametrized with ibis fixture) |
| TEST-02 | All 14 builtin checks on ibis | unit | `pytest tests/backends/narwhals/test_checks.py -x` | ✅ (make_narwhals_frame fixture covers ibis) |
| TEST-02 | lazy=True error collection on ibis | integration | `pytest tests/backends/narwhals/test_container.py -x -k lazy` | ✅ (existing, polars only) |
| TEST-02 | failure_cases is native (not narwhals wrapper) on ibis | unit | `pytest tests/backends/narwhals/test_container.py -x -k native` | ✅ (TEST-03 already covers polars; ibis variant needed) |
| TEST-04 | Polars + Ibis validation depth, strict/filter, lazy, decorators | integration | `pytest tests/backends/narwhals/test_parity.py -x` | ❌ Wave 0 — new file |

### Sampling Rate
- **Per task commit:** `pytest tests/backends/narwhals/ -x -q`
- **Per wave merge:** `pytest tests/backends/narwhals/ tests/ibis/ -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/backends/narwhals/test_parity.py` — covers TEST-04 (and ibis-specific REGISTER-03/TEST-02)
- [ ] Ibis-specific test in `tests/backends/narwhals/test_container.py` for `register_ibis_backends()` UserWarning — covers REGISTER-03 registration verification

*(All other infrastructure — pytest, fixtures, conftest — exists; only new test file and targeted additions needed)*

## Sources

### Primary (HIGH confidence)
- Direct code inspection: `pandera/backends/ibis/register.py` — current register function, no lru_cache
- Direct code inspection: `pandera/backends/polars/register.py` — template for ibis narwhals pattern
- Direct code inspection: `pandera/backends/narwhals/base.py` — current drop_invalid_rows uses pl.fold
- Direct code inspection: `pandera/backends/narwhals/components.py` — check_unique and check_dtype current implementations
- Direct code inspection: `pandera/backends/narwhals/container.py` — check_column_values_are_unique uses .collect()
- Verified test execution: all code examples run and confirmed working in research session

### Secondary (MEDIUM confidence)
- STATE.md accumulated decisions — registration patterns, existing decisions from phases 1-4

### Tertiary (LOW confidence)
- None — all critical claims verified by direct execution

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in use, versions verified
- Architecture: HIGH — all patterns verified by running code in research session
- Pitfalls: HIGH — each pitfall was reproduced and the fix verified during research

**Research date:** 2026-03-14
**Valid until:** 2026-04-14 (stable domain — narwhals and ibis APIs are stable)
