# Stack Research

**Domain:** Narwhals-backed dataframe validation backend for pandera
**Researched:** 2026-03-09
**Confidence:** HIGH (Narwhals 2.15.0 installed and inspected directly from source)

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| narwhals | 2.15.0 (installed) | Unified dataframe API layer bridging pandas, Polars, Ibis, PySpark, PyArrow, Dask, DuckDB | Zero-dependency compatibility layer using a subset of the Polars API; explicitly supports all backends pandera needs; MIT licensed; production-stable (Development Status: 5) |
| narwhals.stable.v1 | same | Backward-compatible namespace for older API guarantees | Pandera should import from `narwhals.stable.v1` rather than `narwhals` directly, to avoid churn from main-API changes. The `v1` namespace provides stability guarantees equivalent to semantic versioning. |
| narwhals.stable.v2 | same | Newer stable namespace, exists in 2.15.0 | Monitor but do not depend on — v2 is present in 2.15.0 but has not yet announced full stability guarantees |

### Narwhals API Surface Relevant to Pandera

#### What Narwhals Provides (HIGH confidence — inspected from source)

**Entry points:**

| Function | Signature | What it does |
|----------|-----------|--------------|
| `nw.from_native(obj)` | Accepts pd.DataFrame, pl.DataFrame, pl.LazyFrame, ibis.Table, pyspark.DataFrame, duckdb.PyRelation, pa.Table, modin.DataFrame, dask.DataFrame | Returns narwhals `DataFrame` or `LazyFrame` wrapper |
| `nw.to_native(obj)` | Narwhals wrapper | Returns the original backend object |
| `nw.narwhalify` | Decorator | Wraps a function so its dataframe args are auto-converted from/to native; useful for check functions |
| `nw.get_native_namespace(obj)` | Narwhals or native object | Returns the native module (e.g., `pandas`, `polars`) |

**DataFrame/LazyFrame shared API (works on both eager and lazy):**

| Method | Notes |
|--------|-------|
| `.schema` | Property, returns `nw.Schema` (dict-like: `{col: nw.DType}`) |
| `.collect_schema()` | Method, same as `.schema` for lazy frames; prefer this for lazy backends |
| `.columns` | List of column names |
| `.select(*exprs)` | Column projection |
| `.with_columns(*exprs)` | Add/replace columns |
| `.filter(*predicates)` | Row filtering |
| `.rename(mapping)` | Rename columns |
| `.drop(*columns, strict=bool)` | Drop columns |
| `.sort(by, descending, nulls_last)` | Sort rows |
| `.group_by(keys)` | Group-by (returns GroupBy, not frame) |
| `.join(other, on, how, ...)` | Joins: inner, left, full, cross, anti, semi |
| `.unique(subset, keep, order_by)` | Deduplication |
| `.drop_nulls(subset)` | Drop rows with nulls |
| `.head(n)` / `.tail(n)` | Subsample |
| `.pipe(fn, *args, **kwargs)` | Function chaining |
| `.explode(columns)` | Unnest list columns |
| `.unpivot(on, index, ...)` | Melt / wide-to-long |
| `.gather_every(n, offset)` | Periodic sampling |

**LazyFrame-only:**

| Method | Notes |
|--------|-------|
| `.collect(backend=None)` | Materializes to DataFrame. `backend` can be `"pandas"`, `"pyarrow"`, or `"polars"`. For Ibis, default collects to PyArrow; for Dask/PySpark also defaults to PyArrow. |
| `.lazy()` | No-op on lazy frames, but exists for API uniformity |

**DataFrame-only (eager):**

| Method | Notes |
|--------|-------|
| `.lazy()` | Converts eager DataFrame to LazyFrame |
| `.to_pandas()`, `.to_arrow()` | Eager materialization |
| `.__getitem__(col)` | Column access by name |
| `.shape` | `(nrows, ncols)` tuple |
| `.row(i)` | Get a row by index |
| `.sample(n, fraction, ...)` | Sampling |

**Expr API (works inside `select`/`with_columns`/`filter`):**

| Expr method | Notes |
|-------------|-------|
| `nw.col("name")` | Column reference |
| `.cast(dtype)` | Type coercion |
| `.is_null()` / `.is_nan()` | Null/NaN detection |
| `.is_in(collection)` | Membership check |
| `.is_between(lo, hi)` | Range check |
| `.is_duplicated()` / `.is_unique()` | Duplicate detection |
| `.is_first_distinct()` / `.is_last_distinct()` | First/last occurrence flags |
| `.n_unique()` / `.null_count()` | Aggregations |
| `.eq(v)` / `.ne(v)` / `.gt(v)` / `.ge(v)` / `.lt(v)` / `.le(v)` | Comparisons |
| `.fill_null(value)` | Null filling |
| `.over(partition_by)` | Window expressions |
| `.any()` / `.all()` | Boolean reductions |
| `nw.all_horizontal(...)` | Cross-column AND |
| `nw.any_horizontal(...)` | Cross-column OR |
| `nw.when(pred).then(val).otherwise(val)` | Conditional |
| `.map_batches(fn, return_dtype)` | Custom function — EAGER ONLY (see Gaps below) |

**Dtype system (all verified from `narwhals/dtypes.py`):**

Narwhals provides a unified dtype hierarchy independent of any backend:
- Integers: `Int8`, `Int16`, `Int32`, `Int64`, `Int128`, `UInt8`, `UInt16`, `UInt32`, `UInt64`, `UInt128`
- Floats: `Float32`, `Float64`
- Other scalars: `String`, `Boolean`, `Date`, `Datetime(time_unit, time_zone)`, `Duration(time_unit)`, `Time`, `Binary`, `Decimal`, `Categorical`, `Enum(categories)`, `Object`, `Unknown`
- Nested: `List(inner)`, `Array(inner, shape)`, `Struct(fields)`
- Introspection: `dtype.is_numeric()`, `dtype.is_integer()`, `dtype.is_float()`, `dtype.is_temporal()`, `dtype.is_nested()`
- Equality: `nw.Int64() == nw.Int64`, `nw.Datetime("us") == nw.Datetime("us", "UTC")` (exact match with params)

**Schema conversion utilities (verified from `narwhals/schema.py`):**

- `nw.Schema.from_native(native_schema)` — converts polars, pyarrow, or pandas dtype dict to `nw.Schema`
- `nw.Schema.from_polars(schema)`, `from_arrow(schema)`, `from_pandas_like(schema)` — per-backend converters
- `schema.to_polars()`, `schema.to_arrow()`, `schema.to_pandas(dtype_backend=...)` — export back to native

**Backend detection:**

- `df.implementation` — returns `nw.Implementation` enum: `PANDAS`, `POLARS`, `IBIS`, `PYSPARK`, `PYSPARK_CONNECT`, `DASK`, `DUCKDB`, `PYARROW`, `MODIN`
- `df.implementation.is_pandas()`, `.is_polars()`, `.is_pandas_like()`, `.is_spark_like()` — boolean checks
- Useful for edge-case special-casing that Narwhals does not abstract

**Supported backends by category (verified from `narwhals/_utils.py` `Implementation` enum and `_from_native_impl`):**

| Category | Backends | API Level |
|----------|----------|-----------|
| Full eager (DataFrame + Series) | pandas, Polars (eager), PyArrow, Modin, cuDF | All Expr/Series ops |
| Full lazy (LazyFrame) | Polars (lazy), Dask | Most Expr ops incl. `map_batches` |
| SQL lazy (LazyFrame, no Series) | Ibis, PySpark, DuckDB | Limited Expr ops; no `map_batches` |

#### Narwhals `stable.v1` namespace

The `stable.v1` namespace re-exports all the same classes and functions as `narwhals` main, but with a stability guarantee. Pandera should use `import narwhals.stable.v1 as nw` to insulate itself from breaking changes in the main namespace.

A `stable.v2` namespace also exists in 2.15.0.

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pyarrow` | >=13 (already in pandera) | Default collection target for Ibis/DuckDB/PySpark lazy frames via `LazyFrame.collect()` | When narwhals collects an Ibis or PySpark frame by default; also for error reporting from lazy backends |
| `polars` | >=1.0 (already in pandera) | Reference eager backend for narwhals; Polars lazy is a first-class narwhals citizen | As the primary test target for the Narwhals backend |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `nox` | Test session runner | Already used; add a `narwhals` session to `noxfile.py` parameterized per backend |
| `pixi` / `uv` | Environment management | Already used; add `narwhals` to the `[pypi-dependencies]` section in `pixi.toml` or `pyproject.toml` optional extras |
| `pytest` markers | Per-backend test parameterization | Use `@pytest.mark.parametrize` over backend fixtures for `tests/backends/narwhals/` |

## Installation

```bash
# As a pandera optional extra (add to pyproject.toml)
pip install pandera[narwhals]

# narwhals itself has no required dependencies — it uses whatever the user has installed
pip install narwhals>=2.15.0

# For development, narwhals is already in the pixi.lock; add to pixi.toml:
# [pypi-dependencies]
# narwhals = ">=2.15.0, <3"
```

## Gaps: Pandera Operations That Do Not Map Cleanly to Narwhals

These are the critical findings for roadmap planning.

### 1. `map_batches` (element-wise and custom checks) — NOT available for lazy backends

**Severity: HIGH**

`map_batches` is marked `not_implemented()` on the `LazyExpr` base class in narwhals (verified in `narwhals/_compliant/expr.py` line 880). This means:
- `element_wise=True` checks (which use `map_elements` / `map_batches` internally) **cannot execute via narwhals** against Ibis, PySpark, or DuckDB.
- Custom check functions that operate element-wise will fail at the narwhals boundary for these backends.
- Mitigation: For lazy SQL backends, element-wise checks must either be skipped, converted to set-based filter expressions, or executed post-`collect()`. Pandera's existing Ibis backend already sidesteps this by disabling element-wise checks.

### 2. Ibis series is "interchange-only" — not a full Series

**Severity: HIGH for column-level checks**

Narwhals wraps Ibis columns as `IbisInterchangeSeries` (verified in `narwhals/_ibis/series.py`), not a full `Series`. This class raises `NotImplementedError` for every attribute except `.dtype`. Pandera column-level checks that operate on `Series` objects (e.g., `is_null()`, `is_in()`, `unique()`) **cannot use narwhals Series API against Ibis** — they must use Expr-based LazyFrame operations instead. The narwhals approach for Ibis is always: build Expr chains on the LazyFrame, never pull a column as a Series.

### 3. No unified dtype coerce API

**Severity: MEDIUM**

Narwhals provides `col.cast(dtype)` which works for all backends. However, pandera's `DataType.coerce()` method is expected to return a LazyFrame/DataFrame with the column cast and produce structured `ParserError` when coercion fails. Narwhals does not expose a "try-cast-report-failures" pattern — pandera must implement its own coercion wrapper using `nw.col(...).cast(dtype)` and catch the backend-specific exceptions (e.g., `narwhals.exceptions.InvalidOperationError`, or native backend errors surfaced through narwhals). The existing polars engine's `polars_object_coercible` pattern (check nulls after cast) is the right model.

### 4. No narwhals dtype engine — dtype resolution stays per-library

**Severity: MEDIUM (key architectural decision)**

Narwhals has its own `nw.DType` hierarchy (e.g., `nw.Int64`, `nw.String`, `nw.Datetime`) that can be read off any frame via `df.collect_schema()`. However, narwhals does NOT provide a `dtype()` factory that maps user-specified pandera `DataType` objects back to narwhals dtypes for coercion. Pandera's Engine metaclass pattern (`pandera/engines/polars_engine.py`) where each `DataType` subclass knows how to resolve and coerce itself is **not replaceable** by narwhals — it is pandera-specific logic.

**Recommendation:** Do NOT create a `narwhals_engine.py`. Instead, map narwhals dtypes to existing pandera `DataType` instances at check time. The narwhals backend should:
1. Use `df.collect_schema()` to get a `nw.Schema` with `nw.DType` values.
2. Map `nw.DType` to pandera `DataType` using a narwhals-dtype-to-pandera lookup table (per the Polars engine pattern, but using narwhals dtypes as input keys rather than polars dtypes).
3. Coerce using `nw.col(name).cast(nw_dtype)` inside `with_columns`.

This avoids duplicating the Engine metaclass pattern while staying within narwhals.

### 5. `over()` (window expressions) — not available for Ibis

**Severity: LOW-MEDIUM**

Narwhals `_evaluate_window_expr` is `not_implemented()` in the Ibis backend (`narwhals/_ibis/dataframe.py` line 427). Window expressions like `over(partition_by=...)` for grouped checks will not work against Ibis. Group-by checks for Ibis must use `group_by(...).agg(...)` chains on the LazyFrame, not window expressions.

### 6. `join_asof` — not available for PySpark

**Severity: LOW**

`join_asof` is `not_implemented()` for the PySpark backend (`narwhals/_spark_like/dataframe.py` line 607). Pandera does not currently use `join_asof` in any backend, so this is not blocking.

### 7. PySpark Series access — no Series, lazy only

**Severity: SAME as Ibis above**

PySpark in narwhals is SQL-lazy-only. No `Series` API, no `map_batches`. All operations must be expressed via Expr on the LazyFrame.

### 8. pandas `index` is abstracted away

**Severity: LOW**

Narwhals treats pandas DataFrames as if they had no index (it uses positional integer index internally). For pandera, this is actually a feature — the narwhals backend does not need to handle pandas' multi-index or non-default index edge cases, unlike the native pandas backend. However, it does mean the narwhals backend cannot replicate `Index`/`MultiIndex` schema validation that pandera's pandas backend supports for pandas schemas. This is expected: the narwhals backend targets DataFrame-level columns, not index-level.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `narwhals.stable.v1` namespace | `narwhals` main namespace | Never for production code — main namespace can change between patch releases |
| Expression-based checks on LazyFrame | Series-based checks per column | Eager-only backends (pandas, Polars eager, PyArrow); use Series API there when simpler, but prefer Expr for code reuse |
| Delegate dtype resolution to existing per-library engines | Create a `narwhals_engine.py` | If a future Narwhals version provides a stable dtype-to-cast mapping API; re-evaluate at narwhals 3.x |
| `nw.from_native(df)` + work in narwhals, then `nw.to_native()` | Work in native frame and use narwhals only for introspection | Never — the whole point is a unified implementation path |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `narwhals` main namespace (non-stable) | Can receive breaking changes between releases without semantic versioning; Narwhals devs explicitly recommend `stable.v1` for library authors | `narwhals.stable.v1` |
| `map_batches` / `map_elements` in lazy SQL check contexts | Not implemented for Ibis, PySpark, DuckDB backends — will raise `NotImplementedError` at runtime | Narwhals Expr chains (`col().is_null()`, `col().is_in()`, etc.) that express the same check in SQL-compatible terms |
| Accessing `df[col]` as a Series for Ibis/PySpark frames | Ibis columns return `IbisInterchangeSeries` which raises `NotImplementedError` on every method except `.dtype` | `df.select(nw.col(name).some_method())` — operate via Expr on the LazyFrame |
| Creating a `narwhals_engine.py` to replicate the Engine metaclass | Narwhals does not expose a dtype resolution/coercion registry; trying to build one duplicates pandera's existing per-library engines with no benefit | Use existing per-library engines for dtype resolution; use `nw.col(name).cast(nw_dtype)` for coercion in the narwhals backend |
| Narwhals Interchange Protocol (v1 interchange-level DataFrames) | Removed in `narwhals` main namespace (only in `stable.v1`); very limited API — only dtype inspection, no operations | `nw.from_native(obj)` which returns a proper LazyFrame for Ibis |
| Depending on `df.implementation` for main logic branches | Defeats the purpose of narwhals as a unifier | Only use `df.implementation` for known narrow edge cases (e.g., pandas Period dtype) that narwhals explicitly cannot abstract |

## Stack Patterns by Variant

**For eager backends (pandas, Polars DataFrame, PyArrow, Modin):**
- Use `nw.from_native(df, eager_only=True)` to get a `DataFrame`
- Series API is fully available: `df[col]` returns a narwhals `Series` with all methods
- `map_batches` works for element-wise custom checks
- `.collect()` is not needed — data is already materialized

**For Polars LazyFrame:**
- Use `nw.from_native(lf)` to get a `LazyFrame`
- All Expr-based operations work including window expressions (`over()`)
- `map_batches` works (Polars lazy supports it)
- Call `.collect()` at the end of validation to materialize errors

**For Ibis Table:**
- Use `nw.from_native(ibis_table)` to get a `LazyFrame`
- Only Expr-based operations; no Series API, no `map_batches`, no window expressions
- Call `.collect()` (defaults to PyArrow) to materialize for error reporting
- Element-wise checks must be converted to set-based filter expressions or skipped

**For PySpark DataFrame:**
- Use `nw.from_native(spark_df)` to get a `LazyFrame`
- Same constraints as Ibis: Expr-only, no Series, no `map_batches`
- Call `.collect()` (defaults to PyArrow) to materialize
- `join_asof` is not available; pandera does not use it, so not a blocker

**For DuckDB PyRelation:**
- Use `nw.from_native(rel)` to get a `LazyFrame`
- SQL-lazy backend; Expr-only, no `map_batches`
- Not a current pandera target but narwhals supports it transparently

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| narwhals 2.15.0 | polars >=0.20.4 | narwhals imposes minimum polars version; pandera already requires polars >=0.20.0, compatible |
| narwhals 2.15.0 | pandas >=1.1.3 | narwhals imposes minimum pandas version; pandera requires pandas >=2.1.1, no conflict |
| narwhals 2.15.0 | ibis-framework >=6.0.0 | narwhals imposes minimum ibis version; pandera requires ibis >=9.0.0, no conflict |
| narwhals 2.15.0 | pyspark >=3.5.0 | narwhals imposes minimum PySpark version; pandera supports PySpark >=3.2.0 — narwhals raises the floor to 3.5.0 for the narwhals backend |
| narwhals 2.15.0 | pyarrow >=13.0.0 | narwhals imposes minimum pyarrow version; pandera already requires pyarrow >=13, compatible |
| narwhals 2.15.0 | Python >=3.9 | narwhals requires Python 3.9+; pandera requires 3.10+, no conflict |

## Sources

- Narwhals 2.15.0 source code, inspected directly from `.pixi/envs/default/lib/python3.12/site-packages/narwhals/` — HIGH confidence for all API findings
- `narwhals/__init__.py` — public API surface
- `narwhals/dtypes.py` — complete dtype hierarchy
- `narwhals/translate.py` — `from_native`, `to_native`, `narwhalify`
- `narwhals/dataframe.py` — `DataFrame`, `LazyFrame`, `BaseFrame` methods
- `narwhals/schema.py` — `Schema`, `from_native`, `from_polars`, `from_arrow`, `from_pandas_like`
- `narwhals/_ibis/dataframe.py` — Ibis lazy frame implementation; confirmed `_evaluate_window_expr = not_implemented()`
- `narwhals/_ibis/series.py` — `IbisInterchangeSeries`; confirmed raises `NotImplementedError` on all methods except `.dtype`
- `narwhals/_spark_like/dataframe.py` — PySpark lazy frame; confirmed `join_asof = not_implemented()`
- `narwhals/_compliant/expr.py` — confirmed `map_batches = not_implemented()` on `LazyExpr` base
- `narwhals/_utils.py` — `Implementation` enum; all supported backend values
- `narwhals-2.15.0.dist-info/METADATA` — version, backend support matrix, library classification

---
*Stack research for: Narwhals-backed pandera validation backend*
*Researched: 2026-03-09*
