# Architecture Patterns

**Domain:** Narwhals-backed validation backend for pandera
**Researched:** 2026-03-09

---

## Recommended Architecture

The Narwhals backend adds a new backend package (`pandera/backends/narwhals/`) and a new engine (`pandera/engines/narwhals_engine.py`). It does NOT introduce a new `pandera/api/narwhals/` API layer. The Narwhals backend is transparent infrastructure: users continue writing schemas with `pandera.polars.DataFrameSchema`, `pandera.pandas.DataFrameSchema`, etc., and the narwhals backend registers itself against those same existing schema classes for those same native data types.

The registration key is `(ExistingSchemaClass, NativeDataType)` — identical to how the polars backend registers `(polars.DataFrameSchema, pl.LazyFrame) -> DataFrameSchemaBackend`. The narwhals backend simply registers an alternative backend class for the same keys, which can be activated by calling `register_narwhals_backends()` after the native backend has already been registered. Because `BaseSchema.register_backend()` skips if a key is already present (first-writer-wins), the activation mechanism needs to set entries directly or the narwhals registration must run before native backends are registered.

### No pandera/api/narwhals/ Layer

The existing per-library API classes (`pandera/api/polars/container.py:DataFrameSchema`, `pandera/api/ibis/container.py:DataFrameSchema`, etc.) already define `register_default_backends()` pointing to their respective native backends. The narwhals backend overrides the backend selection without touching the API layer. Users define schemas with the same API classes they already use; the narwhals backend handles validation execution.

This is the correct reading of the project goal: "users continue to pass their native dataframes and pandera validates them internally via Narwhals." There is no `pandera.narwhals.DataFrameSchema` for users to import.

### Critical Design Decision: Registration Against Native Types

The registry key is `(SchemaClass, DataObjectType)` where `DataObjectType` is the actual runtime `type()` of the object passed to `schema.validate(df)`. Users pass `pl.DataFrame`, `pd.DataFrame`, `ibis.Table` — never `nw.DataFrame`. The backend wraps with `nw.from_native()` internally after receiving the native object.

Correct registration:

```python
# pandera/backends/narwhals/register.py
import polars as pl

polars_DataFrameSchema.register_backend(pl.DataFrame, NarwhalsDataFrameSchemaBackend)
polars_DataFrameSchema.register_backend(pl.LazyFrame, NarwhalsDataFrameSchemaBackend)
polars_Column.register_backend(pl.LazyFrame, NarwhalsColumnBackend)
Check.register_backend(pl.LazyFrame, NarwhalsCheckBackend)
```

The same pattern applies to pandas, ibis, and pyspark — registering against each native type while pointing to the shared narwhals backend classes.

### Custom Check Compatibility: Native Syntax Preserved

Users writing custom checks for native libraries must not be broken. The polars `Check.register_backend(pl.LazyFrame, PolarsCheckBackend)` registration is what makes `Check(lambda data: data.lazyframe.select(...))` work — `data` is `PolarsData` because `PolarsCheckBackend.preprocess()` returns a `PolarsData` wrapping the `pl.LazyFrame`.

When the narwhals backend registers `Check.register_backend(pl.LazyFrame, NarwhalsCheckBackend)`, it replaces the polars check backend. The `NarwhalsCheckBackend` must therefore:

1. For **built-in checks**: pass `NarwhalsData` (narwhals-wrapped) to the check function so the `Dispatcher` routes to `pandera/backends/narwhals/builtin_checks.py`
2. For **user-defined checks**: pass the native data container (e.g. `PolarsData` with the original `pl.LazyFrame`) so user lambdas written in native syntax continue to work

The `Dispatcher` in `pandera/api/function_dispatch.py` dispatches on `type(args[0])`, keyed by the first argument's type annotation of the registered function. Built-in checks registered in `pandera/backends/narwhals/builtin_checks.py` annotate their first argument as `NarwhalsData`, so the dispatcher routes to them when `NarwhalsData` is passed. User-defined check functions are raw callables (not `Dispatcher` instances), so `NarwhalsCheckBackend.apply()` can detect this and pass the native container instead.

```python
# NarwhalsCheckBackend.apply() pseudo-logic
from pandera.api.function_dispatch import Dispatcher

if isinstance(self.check_fn, Dispatcher):
    # built-in: pass NarwhalsData so Dispatcher routes to narwhals builtin_checks
    out = self.check_fn(NarwhalsData(nw_frame, key))
else:
    # user-defined: pass native container so user lambdas work unchanged
    native_container = build_native_container(original_native_obj, key)
    out = self.check_fn(native_container)
    # normalize out to nw.DataFrame boolean mask for postprocessing
```

This means `NarwhalsCheckBackend` must retain a reference to the original native object alongside the narwhals-wrapped frame throughout the validation call.

### Component Boundaries

| Component | File | Responsibility | Communicates With |
|-----------|------|---------------|-------------------|
| NarwhalsSchemaBackend | `pandera/backends/narwhals/base.py` | Shared helpers: subsample, run_check, failure_cases_metadata, drop_invalid_rows | Called by container and column backends |
| DataFrameSchemaBackend | `pandera/backends/narwhals/container.py` | Main validation pipeline: collect_column_info, run parsers, run checks | Calls ColumnBackend; uses ErrorHandler; wraps/unwraps native frames |
| ColumnBackend | `pandera/backends/narwhals/components.py` | Per-column validation: check_nullable, check_unique, check_dtype, run_checks | Called by DataFrameSchemaBackend.run_schema_component_checks |
| NarwhalsCheckBackend | `pandera/backends/narwhals/checks.py` | Execute check functions; routes built-ins to NarwhalsData, user checks to native containers | Called by ColumnBackend.run_check and DataFrameSchemaBackend.run_check |
| Built-in checks | `pandera/backends/narwhals/builtin_checks.py` | Narwhals implementations of equal_to, greater_than, etc., typed on NarwhalsData | Registered via @register_builtin_check; dispatched by Dispatcher on NarwhalsData type |
| Registration | `pandera/backends/narwhals/register.py` | Map (ExistingSchemaClass, NativeType) -> NarwhalsBackendClass | Called explicitly or as part of opt-in mechanism |
| Narwhals engine | `pandera/engines/narwhals_engine.py` | Map nw.Dtype to pandera DataType; implement coerce/check/try_coerce using narwhals API | Used by ColumnBackend.check_dtype and DataFrameSchemaBackend.coerce_dtype |

### Data Flow

**Registration flow:**

```
pandera.use_backend("narwhals")  [or direct register_narwhals_backends() call]
  -> register_narwhals_backends()   [lru_cache, runs once per config]
     -> polars_DataFrameSchema.register_backend(pl.DataFrame, NarwhalsDataFrameSchemaBackend)
     -> polars_DataFrameSchema.register_backend(pl.LazyFrame, NarwhalsDataFrameSchemaBackend)
     -> polars_Column.register_backend(pl.LazyFrame, NarwhalsColumnBackend)
     -> Check.register_backend(pl.LazyFrame, NarwhalsCheckBackend)
     -> [same pattern for pandas, ibis, pyspark — guarded by try/except ImportError]
```

Note: `BACKEND_REGISTRY` is a class-level dict on `BaseSchema`. `register_backend()` skips if the key exists. The narwhals registration must either run before native backends or force-write entries. The activation mechanism (`pandera.use_backend("narwhals")`) needs to handle this ordering or directly write into the registry.

**Validation flow (per validate() call):**

```
schema.validate(native_df)   [e.g. polars DataFrameSchema.validate(pl.DataFrame)]
  -> get_backend(native_df)  -> NarwhalsDataFrameSchemaBackend()
  -> NarwhalsDataFrameSchemaBackend.validate(native_df, schema)
       -> nw_frame = nw.from_native(native_df)   [wrap once; keep original for check dispatch]
       -> collect_column_info(nw_frame, schema)
       -> core parsers loop (all operate on nw_frame):
            add_missing_columns
            strict_filter_columns
            coerce_dtype   -> narwhals_engine.DataType.coerce(nw_frame)
            set_default
       -> collect_schema_components
       -> subsample(nw_frame, head, tail, sample)
       -> core checks loop:
            check_column_presence(nw_frame, schema, column_info)
            check_column_values_are_unique(nw_frame, schema)
            run_schema_component_checks -> NarwhalsColumnBackend.validate(nw_frame, col)
              -> check_nullable, check_unique, check_dtype [all use nw_frame]
              -> run_checks -> NarwhalsCheckBackend(nw_frame, key)
                   if built-in: check_fn(NarwhalsData(nw_frame, key)) -> nw bool mask
                   if user:     check_fn(native_container(original_native, key)) -> normalize
            run_checks(nw_frame, schema)  [dataframe-level checks, same routing]
       -> error handling: raise SchemaErrors or drop_invalid_rows
       -> nw.to_native(nw_frame)   [unwrap back; result matches input library type]
       -> return native result
```

**Dtype checking flow:**

```
NarwhalsColumnBackend.check_dtype(nw_frame, schema)
  -> actual_nw_dtype = nw_frame.schema[col_name]   [e.g. nw.Int64]
  -> schema.dtype.check(actual_nw_dtype)
     [schema.dtype is a narwhals_engine.DataType, e.g. narwhals_engine.Int64]
     -> compare actual_nw_dtype == self.type (nw.Int64)
```

---

## Files to Create

### New Files (complete implementations needed)

```
pandera/backends/narwhals/__init__.py
pandera/backends/narwhals/base.py          # NarwhalsSchemaBackend(BaseSchemaBackend)
pandera/backends/narwhals/container.py     # DataFrameSchemaBackend(NarwhalsSchemaBackend)
pandera/backends/narwhals/components.py    # ColumnBackend(NarwhalsSchemaBackend)
pandera/backends/narwhals/checks.py        # NarwhalsCheckBackend(BaseCheckBackend)
pandera/backends/narwhals/builtin_checks.py  # @register_builtin_check typed on NarwhalsData
pandera/backends/narwhals/register.py      # register_narwhals_backends() with lru_cache

pandera/engines/narwhals_engine.py         # Engine(metaclass=engine.Engine) + DataType subclasses
```

### Modified Files (additions required)

```
pyproject.toml                             # Add narwhals optional dependency group
pandera/__init__.py                        # pandera.use_backend("narwhals") mechanism
tests/backends/narwhals/                   # New test directory
```

### Files That Do NOT Need to Be Created

- `pandera/api/narwhals/` — no new API layer; narwhals is backend-only
- `pandera/narwhals.py` — no user-facing entry module; users import pandera.polars / pandera.pandas as before
- `pandera/typing/narwhals.py` — not a type annotation target
- `pandera/accessors/narwhals_accessor.py` — narwhals wraps native types; accessors remain on native frames

---

## Patterns to Follow

### Pattern 1: NarwhalsSchemaBackend extends BaseSchemaBackend

Mirror `PolarsSchemaBackend` in `pandera/backends/polars/base.py`. Override `subsample()`, `run_check()`, `failure_cases_metadata()`, and `drop_invalid_rows()` using Narwhals API.

Key differences from Polars:
- `nw.concat()` for subsampling (works for both eager and lazy)
- `frame.is_duplicated()` for uniqueness check
- `isinstance(frame, nw.LazyFrame)` to branch lazy-specific behavior
- `frame.columns` instead of `df.collect_schema().names()`

### Pattern 2: DataFrameSchemaBackend mirrors polars/container.py

Replicate `pandera/backends/polars/container.py` step-for-step, substituting Polars API:

| Polars | Narwhals equivalent |
|--------|---------------------|
| `pl.LazyFrame` | `nw.LazyFrame` (or `nw.DataFrame \| nw.LazyFrame`) |
| `df.collect_schema().names()` | `frame.columns` |
| `pl.col(name).is_not_null()` | `nw.col(name).is_not_null()` |
| `df.with_columns(...)` | `frame.with_columns(...)` |
| `df.drop(cols)` | `frame.drop(cols)` |
| `df.cast({col: dtype})` | `frame.cast({col: dtype})` |
| `df.lazy()` | `frame.lazy()` |
| `df.collect()` | `frame.collect()` |

`validate()` wraps with `nw.from_native()` as first step and unwraps with `nw.to_native()` as last step. The original native object is retained for the check dispatch path.

### Pattern 3: register_narwhals_backends uses lru_cache with per-library guards

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def register_narwhals_backends(check_cls_fqn: str | None = None):
    from pandera.api.checks import Check
    from pandera.backends.narwhals import builtin_checks  # noqa: F401
    from pandera.backends.narwhals.checks import NarwhalsCheckBackend
    from pandera.backends.narwhals.components import ColumnBackend
    from pandera.backends.narwhals.container import DataFrameSchemaBackend

    try:
        import polars as pl
        from pandera.api.polars.components import Column as PolarsColumn
        from pandera.api.polars.container import DataFrameSchema as PolarsDataFrameSchema
        PolarsDataFrameSchema.BACKEND_REGISTRY[
            (PolarsDataFrameSchema, pl.DataFrame)
        ] = DataFrameSchemaBackend
        PolarsDataFrameSchema.BACKEND_REGISTRY[
            (PolarsDataFrameSchema, pl.LazyFrame)
        ] = DataFrameSchemaBackend
        PolarsColumn.BACKEND_REGISTRY[(PolarsColumn, pl.LazyFrame)] = ColumnBackend
        Check.BACKEND_REGISTRY[(Check, pl.LazyFrame)] = NarwhalsCheckBackend
    except ImportError:
        pass

    # Same pattern for pandas, ibis, pyspark
```

Direct registry writes (rather than `register_backend()`) are required because `register_backend()` skips existing keys. The `builtin_checks` import is a side-effect-only import to trigger `@register_builtin_check` decorators.

### Pattern 4: narwhals_engine.py uses the metaclass pattern

```python
class Engine(metaclass=engine.Engine, base_pandera_dtypes=DataType):
    """Narwhals data type engine."""

    @classmethod
    def dtype(cls, data_type: Any) -> dtypes.DataType:
        try:
            return engine.Engine.dtype(cls, data_type)
        except TypeError:
            return DataType(data_type)


@Engine.register_dtype(equivalents=["int64", int, nw.Int64, dtypes.Int64, dtypes.Int64()])
@immutable
class Int64(DataType, dtypes.Int64):
    type = nw.Int64
```

`DataType.check()` compares `nw.Dtype` instances using `==`. `DataType.coerce()` calls `frame.cast({key: self.type})`. `DataType.try_coerce()` wraps coerce in error handling that raises `ParserError` with failure cases.

### Pattern 5: NarwhalsCheckBackend routes built-ins vs user checks

```python
from pandera.api.function_dispatch import Dispatcher

class NarwhalsCheckBackend(BaseCheckBackend):

    def apply(self, check_obj: NarwhalsData, original_native_obj):
        if self.check.element_wise:
            # element-wise: apply scalar fn to each element via narwhals map
            ...
        elif isinstance(self.check_fn, Dispatcher):
            # built-in check: pass NarwhalsData so Dispatcher routes to narwhals builtin
            out = self.check_fn(check_obj)
        else:
            # user-defined check: pass native container so user lambdas work unchanged
            native_container = _make_native_container(original_native_obj, check_obj.key)
            raw_out = self.check_fn(native_container)
            # normalize raw_out to nw.DataFrame boolean mask
            out = _normalize_to_nw(raw_out, check_obj)
        return out
```

The `original_native_obj` (before `nw.from_native()`) must be threaded through the call from `DataFrameSchemaBackend.validate()` into `NarwhalsCheckBackend.apply()`.

### Pattern 6: Built-in checks annotate first arg as NarwhalsData

```python
# pandera/backends/narwhals/builtin_checks.py
from pandera.api.narwhals.types import NarwhalsData

@register_builtin_check(aliases=["gt"], error="greater_than({min_value})")
def greater_than(data: NarwhalsData, min_value: Any) -> nw.LazyFrame:
    return data.frame.select(nw.col(data.key).gt(min_value))
```

The `Dispatcher` in `CHECK_FUNCTION_REGISTRY` keys on `type(args[0])`. When `NarwhalsData` is passed, it routes here. When `PolarsData` is passed (native polars check backend still active for polars checks), it routes to `pandera/backends/polars/builtin_checks.py` instead.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Introducing pandera/api/narwhals/ as a user-facing API layer

**What goes wrong:** Creates a new schema class (`pandera.api.narwhals.DataFrameSchema`) that users would need to import, contradicting the project goal of transparent backend selection.

**Prevention:** The narwhals backend registers against existing per-library schema classes. No new API layer.

### Anti-Pattern 2: Registering nw.DataFrame as the data type

**What goes wrong:** `BaseSchema.get_backend()` calls `type(check_obj)`. Users pass `pl.DataFrame`, not `nw.DataFrame`. Registering narwhals wrapper types means `BackendNotFoundError` on every real validate call.

**Prevention:** Register against native library types (`pl.DataFrame`, `pd.DataFrame`, `ibis.Table`). The backend wraps internally with `nw.from_native()` after receiving the native object.

**Detection:** `BackendNotFoundError` despite `register_narwhals_backends()` running.

### Anti-Pattern 3: Using register_backend() for narwhals activation (first-writer-wins prevents override)

**What goes wrong:** `BaseSchema.register_backend()` skips if `(cls, type_)` is already in `BACKEND_REGISTRY`. The polars backend registers first. Calling `register_backend(pl.DataFrame, NarwhalsBackend)` silently does nothing.

**Prevention:** For narwhals activation (overriding existing registrations), write directly to `BACKEND_REGISTRY` rather than using the `register_backend()` helper.

**Detection:** Narwhals backend never gets called despite registration; native polars backend still executes.

### Anti-Pattern 4: Wrapping to nw.DataFrame before get_backend() sees the object

**What goes wrong:** If `nw.from_native()` happens before `get_backend()`, the registry lookup sees `nw.DataFrame` instead of `pl.DataFrame` and raises `BackendNotFoundError`.

**Prevention:** `nw.from_native()` is the first line inside `DataFrameSchemaBackend.validate()`, after the dispatch has already resolved.

### Anti-Pattern 5: Passing NarwhalsData to user-defined check functions

**What goes wrong:** A user writing `Check(lambda data: data.lazyframe.select(pl.col("x").gt(0)))` expects `data` to be `PolarsData` with a `.lazyframe` attribute. Passing `NarwhalsData` breaks with `AttributeError: 'NarwhalsData' has no attribute 'lazyframe'`.

**Prevention:** `NarwhalsCheckBackend.apply()` detects whether `self.check_fn` is a `Dispatcher` (built-in) or raw callable (user-defined) and passes the appropriate container type.

### Anti-Pattern 6: narwhals_engine.Engine without the metaclass pattern

**What goes wrong:** Without `metaclass=engine.Engine`, `@Engine.register_dtype()` silently does nothing and `Engine.dtype()` dispatch is unavailable. Dtype coercion fails with `AttributeError`.

**Prevention:** `class Engine(metaclass=engine.Engine, base_pandera_dtypes=DataType):` — copy verbatim from `polars_engine.py`.

### Anti-Pattern 7: Importing all libraries unconditionally at module level in register.py

**What goes wrong:** A user with only Polars installed gets `ImportError` for pandas/ibis when narwhals registration activates.

**Prevention:** Wrap each library import in `try/except ImportError: pass` inside the registration function body.

---

## Build Order

```
Step 1 — Narwhals data types and utilities (no runtime library coupling):
  pandera/api/narwhals/types.py     # NarwhalsData, NarwhalsFrame, NarwhalsCheckObjects
  pandera/api/narwhals/utils.py     # get_frame_column_names, get_frame_schema

Step 2 — Engine (depends on types, pandera dtypes):
  pandera/engines/narwhals_engine.py
    - DataType base class + Engine metaclass
    - Register nw.Int64, nw.Float64, nw.String, nw.Boolean, nw.Date, nw.Datetime
    - coerce(), try_coerce(), check() must work before container backend can be built

Step 3 — Backend base (depends on types, pandera errors):
  pandera/backends/narwhals/base.py    # NarwhalsSchemaBackend(BaseSchemaBackend)

Step 4 — Check backend (depends on base, types):
  pandera/backends/narwhals/checks.py  # NarwhalsCheckBackend(BaseCheckBackend)

Step 5 — Built-in checks (depends on check backend, API extensions):
  pandera/backends/narwhals/builtin_checks.py

Step 6 — Column backend (depends on base, check backend, engine):
  pandera/backends/narwhals/components.py  # ColumnBackend(NarwhalsSchemaBackend)

Step 7 — Container backend (depends on base, column backend, check backend):
  pandera/backends/narwhals/container.py  # DataFrameSchemaBackend(NarwhalsSchemaBackend)

Step 8 — Registration (depends on all backends):
  pandera/backends/narwhals/register.py   # register_narwhals_backends()

Step 9 — Integration hooks:
  pyproject.toml        # narwhals optional extra
  pandera/__init__.py   # pandera.use_backend("narwhals") activation
```

**Rationale for this order:**
- Engine must exist before any backend calls `schema.dtype.coerce()` or `schema.dtype.check()`
- `register.py` imports from all backend modules; all must exist before registration is attempted
- `builtin_checks.py` side-effect import in `register.py` requires check backend to already be importable
- `pandera/api/narwhals/types.py` and `utils.py` are light utility modules with no circular risk; build first

**Note on pandera/api/narwhals/:** These two utility files (`types.py`, `utils.py`) contain no schema classes — only type aliases and helper functions used by backends and the engine. They belong under `pandera/api/narwhals/` by naming convention but do not constitute a user-facing API layer. No `container.py`, `components.py`, or `model.py` is created there.

---

## Scalability Considerations

| Concern | Approach |
|---------|----------|
| Adding PySpark support | Add `pyspark.sql.DataFrame` to the library-guarded block in `register.py`; no other files change |
| Adding PyArrow support | Add `pyarrow.Table` to `register.py`; add pyarrow dtypes to `narwhals_engine.py` |
| Narwhals dtype coverage | Add `@Engine.register_dtype` entries to `narwhals_engine.py`; no other files change |
| Coexistence with native backends | Native backends remain as defaults; narwhals backend activates only when `register_narwhals_backends()` is called |
| Backend precedence | Direct `BACKEND_REGISTRY` writes in `register_narwhals_backends()` override the native backend for the opted-in scope |

---

## Sources

- `pandera/backends/polars/register.py` — canonical registration pattern and lru_cache convention
- `pandera/backends/polars/base.py` — PolarsSchemaBackend helpers to mirror
- `pandera/backends/polars/container.py` — complete validation pipeline to replicate
- `pandera/backends/polars/components.py` — ColumnBackend pattern to mirror
- `pandera/backends/polars/checks.py` — PolarsCheckBackend apply/postprocess pattern; source of built-in vs user check distinction
- `pandera/backends/polars/builtin_checks.py` — PolarsData-typed check function pattern
- `pandera/api/base/schema.py` — BaseSchema.get_backend() confirms native type keying via type(check_obj); register_backend() first-writer-wins semantics
- `pandera/api/function_dispatch.py` — Dispatcher.register() and __call__() confirm type(args[0]) dispatch; explains NarwhalsData annotation requirement for builtin_checks
- `pandera/engines/polars_engine.py` — Engine metaclass and DataType pattern; direct model for narwhals_engine.py
- `pandera/backends/ibis/register.py` — secondary confirmation of per-type registration pattern
