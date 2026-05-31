# Feature Landscape

**Domain:** Narwhals-backed pandera validation backend
**Researched:** 2026-03-09

---

## Summary

The Narwhals backend must implement every validation feature that the reference backends (Polars, Ibis) support, expressed in terms of the Narwhals expression API instead of native library APIs. The feature surface splits cleanly into three layers: schema-level structural checks (column presence, dtype, ordering), column-level value checks (nullable, unique, user checks), and the cross-cutting built-in check suite (14 check functions registered per backend). Narwhals provides direct expression equivalents for nearly all built-in checks; the hard cases are `unique_values_eq` (requires materialisation), dtype coercion (needs a `narwhals_engine.py`), and set-default on float columns (requires float-aware null/nan fill). The main differentiator is lazy-mode pandas validation, which the native pandas backend cannot provide at all.

---

## Table Stakes

Features users expect. Missing = backend is useless.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Column presence check | Every schema must verify its declared columns exist | Low | `nw.LazyFrame.collect_schema().names()` gives column list; straightforward membership test |
| Column dtype check | Core purpose of pandera | Medium | Requires mapping narwhals dtypes (`nw.Int64`, `nw.String`, etc.) to pandera `DataType` instances; blocked until dtype engine decision is resolved |
| Nullable check | `nullable=False` is the default on every column | Medium | `nw.col(name).is_null()` exists. Float columns also need `.is_nan()` — narwhals `Expr.is_nan()` is present; float detection uses `nw.Schema` dtype inspection |
| Schema-level unique check | `schema.unique=["a","b"]` cross-column uniqueness | Medium | Narwhals `DataFrame.is_duplicated()` available but eager-only; must `.collect()` first for lazy frames |
| Column-level unique check | `Column(unique=True)` per-column uniqueness | Medium | Same materialise-then-check pattern; absent from Ibis backend (xfail); must not repeat that gap |
| Strict / filter column mode | `strict=True` rejects extra columns; `strict="filter"` drops them | Low | `nw.LazyFrame.drop()` available; column list from `collect_schema().names()` |
| Ordered columns | `ordered=True` validates column ordering | Low | Pure Python comparison over `collect_schema().names()` list |
| User-defined element-wise checks | `Check(lambda s: s > 0, element_wise=True)` | Medium | Narwhals `Expr.map_elements` exists but behaviour varies across backends for lazy frames; confirm element-wise path works for all target libraries before committing |
| User-defined series checks | `Check(lambda s: (s > 0).all())` returning bool | Low | Pass `NarwhalsData(frame, key)` to check fn; postprocess bool or frame output |
| Dataframe-level checks | `schema.checks=[Check(fn)]` over full frame | Low | Same dispatch; key is `None` or `"*"` |
| Lazy validation mode | `schema.validate(df, lazy=True)` collects all errors | Medium | `ErrorHandler(lazy=True)` pattern already exists in Polars/Ibis; replicate exactly |
| Eager validation mode | `schema.validate(df)` raises on first error | Low | Default; simpler than lazy |
| `add_missing_columns` parser | Adds schema columns absent from the frame | Medium | `nw.LazyFrame.with_columns()` available; default literal value + `cast` using narwhals dtype |
| `coerce=True` dtype coercion | Casts column to declared dtype | High | Narwhals `Expr.cast()` exists; but pandera's `DataType.coerce()` / `try_coerce()` methods must be wired to a narwhals engine — this is the biggest open question (see Pitfalls) |
| Error collection and `SchemaError` / `SchemaErrors` | Users rely on structured error output | Low | Reuse existing `ErrorHandler`, `SchemaError`, `SchemaErrorReason` — no new infrastructure needed |
| Backend registration | `BACKEND_REGISTRY[(DataFrameSchema, pl.DataFrame)]` maps to narwhals backend | Low | Follow `pandera/backends/polars/register.py` pattern exactly |
| `drop_invalid_rows=True` | Filters failing rows and returns cleaned frame | Medium | Must materialise (`collect`), filter by error row indices, return to original frame type |

---

## Built-in Checks (column-level value checks)

All 14 built-in checks must be implemented in `pandera/backends/narwhals/builtin_checks.py`. The table maps each check to its Narwhals expression and notes difficulty.

| Check | Narwhals Expression | Difficulty | Notes |
|-------|---------------------|------------|-------|
| `equal_to(value)` | `nw.col(key) == value` | Low | Direct `__eq__` on `Expr` |
| `not_equal_to(value)` | `nw.col(key) != value` | Low | Direct `__ne__` |
| `greater_than(min_value)` | `nw.col(key) > min_value` | Low | Direct `__gt__` |
| `greater_than_or_equal_to(min_value)` | `nw.col(key) >= min_value` | Low | Direct `__ge__` |
| `less_than(max_value)` | `nw.col(key) < max_value` | Low | Direct `__lt__` |
| `less_than_or_equal_to(max_value)` | `nw.col(key) <= max_value` | Low | Direct `__le__` |
| `in_range(min, max, include_min, include_max)` | `nw.col(key).is_between(min, max, closed=...)` | Low | `is_between` has a `closed` param with `"both"/"left"/"right"/"none"` — maps cleanly to `include_min`/`include_max` booleans |
| `isin(allowed_values)` | `nw.col(key).is_in(allowed_values)` | Low | `is_in` accepts any iterable; narwhals explicitly rejects `Expr` argument (unlike Polars), which is fine here since pandera passes a Python list |
| `notin(forbidden_values)` | `~nw.col(key).is_in(forbidden_values)` | Low | Invert `is_in` result |
| `str_matches(pattern)` | `nw.col(key).str.contains(pattern)` with `^` prefix | Low | Narwhals `str.contains` supports regex; anchor with `^` for match-from-start semantics matching Polars behaviour |
| `str_contains(pattern)` | `nw.col(key).str.contains(pattern)` | Low | Same method, no anchor |
| `str_startswith(string)` | `nw.col(key).str.starts_with(string)` | Low | Direct mapping |
| `str_endswith(string)` | `nw.col(key).str.ends_with(string)` | Low | Direct mapping |
| `str_length(min, max, exact)` | `nw.col(key).str.len_chars()` + comparison | Low | `len_chars()` exists; combine with `is_between` or `==` for exact |
| `unique_values_eq(values)` | Must `.collect()` then `Series.unique()` | Medium | No lazy equivalent; forces materialisation; return `bool` (same as Polars backend does) |

**Check return type contract:** Narwhals builtin checks must return either a `nw.LazyFrame` with a single boolean column (column-level checks) or a `bool` (whole-frame aggregation checks like `unique_values_eq`). The check backend's `postprocess` must handle both, mirroring `PolarsCheckBackend`.

---

## Differentiators

Features that Narwhals enables beyond what any single native backend provides today.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Lazy-mode pandas validation | `pd.DataFrame` currently validates eagerly only; Narwhals wraps it in a lazy graph via `nw.from_native(df).lazy()` | Medium | Users get deferred validation and full error collection on pandas without changing their schema definitions |
| Single backend for Polars + Ibis + pandas | One code path to maintain instead of three | High (upfront) | Reduces long-term maintenance; individual library quirks surface as narwhals version requirements |
| DuckDB / PyArrow table validation for free | Narwhals already supports `duckdb.PyRelation` and `pyarrow.Table`; the backend inherits support once registered | Low (incremental) | Register `(DataFrameSchema, pa.Table)` and `(DataFrameSchema, duckdb.DuckDBPyRelation)` after core is working |
| Future Dask support without new backend | Narwhals includes a `_dask` backend; if Narwhals adds Dask support to its stable API, pandera inherits it | Low (incremental) | Conditional on narwhals stabilising Dask support |
| Fills Ibis backend gaps without Ibis-specific code | Ibis backend is missing `coerce_dtype`, column `unique`, `set_default`; narwhals backend reimplements these generically | Medium | Closes 5 xfail tests in the ibis test suite through a unified path |

---

## Anti-Features

Features to deliberately NOT try to implement in v1.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Series / Index / MultiIndex validation | The pandas backend has `SeriesSchema`, `Index`, `MultiIndex` backends; narwhals has no index concept | Scope narwhals backend to `DataFrameSchema` + `Column` only; document that Index validation requires native pandas backend |
| Hypothesis strategies | `pandera/strategies/` is pandas-only and tightly coupled to `hypothesis`; narwhals has no synthesis path | PROJECT.md explicitly defers to future milestone |
| Schema IO (YAML/JSON) | `pandera/io/` is pandas-only; narwhals dtypes would need their own serialisation format | PROJECT.md explicitly defers to future milestone |
| `element_wise=True` checks on Ibis | Ibis does not support arbitrary Python UDFs via narwhals; `map_elements` on the narwhals ibis backend raises `NotImplementedError` | Raise a clear `UnsupportedCheckError` early when `element_wise=True` and an Ibis input are combined, rather than silently failing at execution time |
| Groupby-based checks (`Check(fn, groupby="col")`) | `groupby` / `query` / `aggregate` are not implemented in the Polars check backend either (all `raise NotImplementedError`); narwhals groupby is a separate code path | Keep `raise NotImplementedError` for groupby in the narwhals check backend for v1; document the gap |
| `report_duplicates` / `keep_setting` for column uniqueness | The `convert_uniquesettings("first"/"last"/"all")` logic in the Ibis backend has no clean narwhals equivalent because row-number window functions are not universally supported across narwhals backends | Always report all duplicates (equivalent to `keep_setting="all"`) for v1; make `report_duplicates` a no-op with a warning |
| Subsampling (`head`/`tail`/`sample` params) | The Ibis and Polars backends both silently ignore these; narwhals `DataFrame.head()` / `DataFrame.tail()` / `DataFrame.sample()` exist but require materialisation for lazy frames | Skip for v1 with a warning; easier to add post-MVP since narwhals exposes the methods |
| pyspark.pandas inline special-casing | CONCERNS.md documents 55+ inline branches in the pandas backend; narwhals backend should not replicate this | Route pyspark.pandas users to the native pandas backend; do not register a narwhals backend for pyspark.pandas types in v1 |

---

## Feature Dependencies

```
Backend registration
  -> validate() method (container)
      -> collect_column_info()
      -> strict_filter_columns()     [parser]
      -> add_missing_columns()       [parser; depends on: dtype coercion]
      -> coerce_dtype()              [parser; depends on: narwhals_engine.py OR existing engine delegation]
      -> set_default()               [parser; depends on: float-aware null/nan detection]
      -> check_column_presence()     [check; depends on: collect_column_info]
      -> check_column_values_are_unique() [check; requires: materialise]
      -> run_schema_component_checks()    [check; depends on: ColumnBackend]
          -> ColumnBackend.validate()
              -> coerce_dtype()           [column-level; same dependency as container]
              -> set_default()            [column-level]
              -> check_nullable()         [check; requires: float dtype detection]
              -> check_unique()           [check; requires: materialise]
              -> check_dtype()            [check; depends on: dtype engine]
              -> run_checks()             [check; depends on: CheckBackend]
                  -> NarwhalsCheckBackend.apply()
                      -> builtin_checks.py    [depends on: NarwhalsData type]
      -> run_checks()                [dataframe-level checks; depends on: NarwhalsCheckBackend]

narwhals_engine.py (or engine delegation)
  -> dtype.check()
  -> dtype.coerce()
  -> dtype.try_coerce()

NarwhalsData named tuple
  -> builtin_checks.py
  -> NarwhalsCheckBackend
```

**Critical path for a minimal working backend:**

1. `NarwhalsData` named tuple (analogous to `PolarsData`)
2. `NarwhalsCheckBackend` — `checks.py` with `apply` / `postprocess`
3. `builtin_checks.py` — all 14 checks, all low complexity
4. `ColumnBackend` — `check_nullable`, `check_unique`, `check_dtype`, `run_checks`
5. `DataFrameSchemaBackend` — `validate`, `collect_column_info`, `strict_filter_columns`, `check_column_presence`, `check_column_values_are_unique`, `run_schema_component_checks`, `run_checks`
6. `register.py` — Polars first, then Ibis, then pandas
7. Dtype engine + `coerce_dtype` — highest complexity, separate research required
8. `add_missing_columns` + `set_default` — medium; depend on dtype engine being available

---

## MVP Recommendation

Prioritise in this order:

1. `NarwhalsData` type + `NarwhalsCheckBackend` — enables the whole check dispatch chain
2. All 14 builtin checks — all map directly to narwhals expressions; low risk, high coverage
3. `ColumnBackend`: nullable, unique, dtype, run_checks — core column validation
4. `DataFrameSchemaBackend`: validate, column presence, uniqueness, strict mode, run_checks — end-to-end frame validation without coercion
5. Register for Polars — first target library; enables running tests
6. Register for Ibis — second target; immediately closes Ibis xfail gaps
7. Dtype coercion + engine — defer until core validation is green; this is the hardest piece and blocks nothing else except `coerce=True` schemas

Defer:
- `add_missing_columns` and `set_default`: Depend on dtype coercion being available. Mark as `NotImplementedError` until the engine is resolved.
- pandas registration: Register after Polars and Ibis are passing; the narwhals pandas lazy path adds complexity.
- `drop_invalid_rows`: Requires full materialisation + row-filtering; implement after core is stable.
- Subsampling: Non-trivial on lazy frames; add after MVP.

---

## Sources

- Narwhals `Expr` class — `narwhals/expr.py` (installed locally at `.pixi/envs/default/lib/python3.12/site-packages/narwhals/`; HIGH confidence)
- Narwhals `ExprStringNamespace` — `narwhals/expr_str.py` (installed locally; HIGH confidence)
- Narwhals `DType` hierarchy — `narwhals/dtypes.py` (installed locally; HIGH confidence)
- Narwhals `DataFrame` / `LazyFrame` — `narwhals/dataframe.py` (installed locally; HIGH confidence)
- Polars backend reference implementation — `pandera/backends/polars/` (source; HIGH confidence)
- Ibis backend reference implementation — `pandera/backends/ibis/` (source; HIGH confidence)
- Built-in checks (Polars, Ibis, pandas) — `pandera/backends/*/builtin_checks.py` (source; HIGH confidence)
- Known gaps and xfail tests — `.planning/codebase/CONCERNS.md` (source; HIGH confidence)
