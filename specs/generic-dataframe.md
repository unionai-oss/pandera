# Generic DataFrame Schema API Spec

> **Status:** Draft
> **Author:** pandera maintainers
> **Related work:**
> [`docs/source/narwhals_backend.md`](../docs/source/narwhals_backend.md),
> [narwhals](https://narwhals-dev.github.io/narwhals/),
> [specs/xarray.md](./xarray.md)

---

## 1. Motivation

Pandera today maintains **one schema API implementation per dataframe
library**: `pandera.pandas`, `pandera.polars`, `pandera.pyspark`,
`pandera.ibis`, plus geopandas/pyarrow/modin/dask variants layered on the
pandas API. Each implementation subclasses the shared generic layer in
`pandera/api/dataframe/`, but still duplicates:

- a concrete `DataFrameSchema` / `Column` / `DataFrameModel` triple,
- a backend package (`pandera/backends/{lib}/`) re-implementing the same
  core checks in the library's native expression language,
- an engine (`pandera/engines/{lib}_engine.py`) redeclaring the same dtypes,
- a typing module (`pandera/typing/{lib}.py`),
- registration glue and docs.

The [narwhals backend](../docs/source/narwhals_backend.md) already proves
that a single validation implementation can serve polars, ibis, and pyspark:
`pandera/backends/narwhals/` re-implements the container, component, and
check backends once in narwhals expressions, and is registered *underneath*
the existing per-library schema classes via
`CONFIG.use_narwhals_backend`.

What's missing is the top half: **a dataframe-agnostic schema API** —
`DataFrameSchema` / `DataFrameModel` and their components — that users can
apply directly to *any* narwhals-compatible dataframe without picking a
per-library module first. This spec defines that API, its typing story, its
extensibility model, and the roadmap by which it eventually **replaces** the
per-library schema implementations.

### Design goals

1. **Narwhals is the default backend** of the generic
   `DataFrameSchema`/`DataFrameModel`, but — as with every other pandera
   schema class — the backend is customizable via the existing
   `BACKEND_REGISTRY` mechanism.
2. **The typing system must work** under both mypy (plugin) and
   pyright/Pylance (no plugin support — careful generics only).
3. **Custom checks can be written against any DF type** (native polars,
   pandas, duckdb, ... signatures), while the orchestrating validator remains
   the narwhals backend, so long as the DF type is narwhals-compatible.
4. **The generic API eventually replaces the per-framework schema APIs**
   (pandas, polars, pyspark, ibis), with a developer-friendly way to
   register additional dataframe types.
5. **The xarray API grows a generic `NDArray` typing construct** so
   xarray-like libraries can reuse the xarray schema API, mirroring how
   narwhals-compatible libraries reuse the generic dataframe API.

---

## 2. Current State of the Codebase

This section records the facts the design builds on (as of `main` @
`e870571`).

### 2.1 What exists

| Layer | Status |
|---|---|
| `pandera/backends/narwhals/` | **Complete backend**: `DataFrameSchemaBackend`, `ColumnBackend`, `NarwhalsCheckBackend`, built-in checks as `nw.Expr` functions, lazy end-to-end validation, SQL-lazy (ibis/duckdb/pyspark/sqlframe) awareness via `pandera/api/narwhals/utils.py`. |
| `pandera/engines/narwhals_engine.py` | **Complete engine**: all core dtypes registered against `nw.*` types; `Engine.dtype()` falls back to the abstract `pandera.dtypes.*` base of foreign-engine dtypes, so cross-engine dtype comparison works. |
| `pandera/api/narwhals/` | **Support code only**: `NarwhalsData`, `NarwhalsCheckResult` (types.py), `ErrorHandler`, lazy/materialize helpers (utils.py). No schema/model classes. |
| `pandera/api/dataframe/` | The shared generic layer: `DataFrameSchema(Generic[TDataObject], BaseSchema)`, `ComponentSchema(Generic[TDataObject])`, `DataFrameModel(Generic[TDataFrame, TSchema], BaseModel)`, `Field`/`FieldInfo`, `BaseConfig`. All per-library APIs subclass these. |
| Backend registry | `BaseSchema.BACKEND_REGISTRY: dict[(schema_cls, frame_type), backend_cls]`; `get_backend()` walks the check object's MRO; `register_default_backends()` is the per-class lazy-registration hook. |
| Typing | `pandera/typing/common.py` (`DataFrameBase[T]`, `SeriesBase`, `AnnotationInfo`, patched `_GenericAlias.__call__` for `DataFrame[Model](data)` validation); per-library `typing/{pandas,polars,ibis}.py`; mypy plugin in `pandera/mypy.py`. Branch `nielsb/fix-series-typing` adds `DataFrame(DataFrameBase[T], pd.DataFrame)` proper parameterization, mypy `__getitem__`/attribute hooks for column-level inference, and mypy + pyright CI test suites. |

### 2.2 What's missing (this spec)

- A narwhals-native/generic `DataFrameSchema`, `Column`, `DataFrameModel`,
  and `BaseConfig` public API.
- A `pandera/typing/narwhals.py` (or generic) typing module.
- Automatic backend resolution for *any* narwhals-compatible frame type
  (today the narwhals backend is only registered against an enumerated set:
  `pl.DataFrame/LazyFrame`, `nw.DataFrame/LazyFrame`, `ibis.Table`,
  `pyspark.sql.DataFrame` — always under the per-library schema classes).
- A public registration API + entry point for third-party frame types.
- A migration path that lets the per-library APIs become thin shims.

### 2.3 Known gaps in the narwhals backend (inherited)

The generic API sits on the narwhals backend, so it inherits the gaps
documented in `docs/source/narwhals_backend.md` § Known gaps: column-level
`coerce=True` no-op, no `add_missing_columns`/`set_default`, no
`group_by` checks, no element-wise checks on SQL-lazy backends, no schema
IO, no synthesis strategies, `sample=`/`tail=` limits on SQL-lazy, and
`drop_invalid_rows` not filtering uniqueness failures. Closing these is a
**prerequisite for the replacement milestone** (§8), not for the alpha.

---

## 3. Proposed Architecture

### 3.1 Module layout

```
pandera/
├── api/
│   └── narwhals/                  # grows from support-code to full API
│       ├── types.py               # (exists) NarwhalsData, NarwhalsCheckResult
│       ├── utils.py               # (exists) lazy/materialize helpers
│       ├── error_handler.py       # (exists)
│       ├── container.py           # NEW: DataFrameSchema
│       ├── components.py          # NEW: Column
│       ├── model.py               # NEW: DataFrameModel
│       ├── model_config.py        # NEW: BaseConfig
│       └── registry.py            # NEW: frame-type registration (§6)
├── backends/narwhals/             # (exists, reused as-is)
├── engines/narwhals_engine.py     # (exists, reused as-is)
├── typing/narwhals.py             # NEW: DataFrame[T], LazyFrame[T], Series[T]
└── narwhals.py                    # NEW: public accessor module
```

**Public import surface** (mirrors `pandera.polars`, `pandera.ibis`):

```python
import pandera.narwhals as pa

class Model(pa.DataFrameModel):
    a: int
    b: str = pa.Field(isin=["x", "y"])

schema = pa.DataFrameSchema({"a": pa.Column(int)})
```

**Naming decision — `pandera.narwhals`, not `pandera.generic`.** The module
is named for its default backend, consistent with every existing accessor
module, and because "generic" already means something else in the codebase
(`pandera/api/dataframe/` is internally called the generic layer, and
`GENERIC_SCHEMA_CACHE` refers to `Generic[...]` model subscripting). When
the replacement milestone lands (§8), the *top-level* `pandera.DataFrameSchema`
becomes the alias for this API — at that point users never type "narwhals"
at all, which is the real dataframe-agnostic spelling.

### 3.2 Class hierarchy

All classes slot into the existing generic layer exactly like polars/ibis
do today:

```python
# pandera/api/narwhals/container.py
from narwhals.typing import IntoFrame
import narwhals as nw

# Everything the narwhals backend can ingest. Kept as a runtime-checkable
# alias; the *authoritative* test is nw.from_native (see §3.3).
NarwhalsCheckObjects = Union[nw.DataFrame, nw.LazyFrame, IntoFrame]

class DataFrameSchema(_DataFrameSchema[NarwhalsCheckObjects]):
    """Dataframe-agnostic schema; validates any narwhals-compatible frame."""

    @staticmethod
    def register_default_backends(check_obj_cls: type) -> None:
        register_narwhals_native_backends(check_obj_cls)   # §3.3

    def validate(
        self,
        check_obj: TFrame,          # TypeVar("TFrame", bound=NarwhalsCheckObjects)
        head=None, tail=None, sample=None, random_state=None,
        lazy=False, inplace=False,
    ) -> TFrame: ...
```

```python
# pandera/api/narwhals/components.py
class Column(ComponentSchema[NarwhalsCheckObjects]): ...

# pandera/api/narwhals/model.py
class DataFrameModel(_DataFrameModel[nw.LazyFrame, DataFrameSchema]):
    Config: type[BaseConfig]

    @classmethod
    def validate(cls, check_obj: TFrame, ...) -> DataFrame[Self]: ...
```

Notes:

- `validate()` is **input-type preserving** at runtime, exactly like the
  narwhals backend today: eager in → eager out, lazy in → lazy out, native
  in → same native type out (`_to_frame_kind_nw`). The `TFrame` TypeVar
  expresses this statically (§5).
- `Column` supports the same kwargs as the polars `Column` (no
  `Index`/`MultiIndex` — see §8.3 for the pandas question).
- `BaseConfig` gains one new option, `validation_backend: str = "narwhals"`
  (§4), and inherits everything else from
  `pandera/api/dataframe/model_config.py`.
- The engine for dtype resolution is `narwhals_engine.Engine`. Users can
  annotate model fields with python builtins (`int`, `str`), pandera dtypes
  (`pa.Int64`), narwhals dtypes (`nw.Int64`), **or foreign-engine dtypes**
  (`pl.Int64`, `"int64[pyarrow]"`) — the existing cross-engine fallback in
  `narwhals_engine.Engine.dtype()` normalizes them.

### 3.3 Backend resolution: dynamic, not enumerated

Today `register_polars_backends`/`register_ibis_backends`/... enumerate
concrete frame types. The generic schema instead resolves the backend
**dynamically from narwhals compatibility**:

```python
# pandera/api/narwhals/registry.py
@lru_cache
def register_narwhals_native_backends(check_obj_cls: type) -> None:
    if not is_narwhals_compatible(check_obj_cls):
        raise BackendNotFoundError(
            f"{check_obj_cls} is not narwhals-compatible. Install a "
            "narwhals-supported library or register a custom backend via "
            "pandera.narwhals.register_frame_type (see docs)."
        )
    from pandera.backends.narwhals import (
        DataFrameSchemaBackend, ColumnBackend, NarwhalsCheckBackend,
    )
    DataFrameSchema.register_backend(check_obj_cls, DataFrameSchemaBackend)
    Column.register_backend(check_obj_cls, ColumnBackend)
    Check.register_backend(check_obj_cls, NarwhalsCheckBackend)


def is_narwhals_compatible(check_obj_cls: type) -> bool:
    """True if narwhals can wrap instances of this class.

    Checks nw.DataFrame/nw.LazyFrame subclasses, then narwhals'
    dependency-based type predicates (is_into_frame et al.) — without
    instantiating anything.
    """
```

Consequences:

- **Zero-config support** for every library narwhals supports now or later
  (pandas, polars, modin, cuDF, pyarrow, dask, duckdb, ibis, pyspark,
  sqlframe, ...). New narwhals releases extend pandera for free.
- Registration remains keyed per concrete class in `BACKEND_REGISTRY`, so
  the MRO-walk in `BaseSchema.get_backend` and `force=True` overrides keep
  working — goal 1 (customizable backend) needs no new machinery: a user
  or plugin calls
  `DataFrameSchema.register_backend(MyFrame, MyBackend, force=True)`
  before/after the default registration, same as today.
- `pandera/backends/register_checks.py::register_default_check_backends`
  gains a terminal fallback branch: if no module-name route matches, try
  `is_narwhals_compatible` before raising.

### 3.4 What does *not* change

- `pandera/backends/narwhals/*` is reused unmodified as the default
  backend. It already accepts arbitrary narwhals-wrappable inputs and
  round-trips eager/lazy/native kinds.
- The `use_narwhals_backend` config toggle for the *per-library* APIs is
  unaffected; it remains the bridge until those APIs are replaced (§8).
- `pandera/api/dataframe/` stays the shared base. The generic API is a
  sibling of polars/ibis, not a rewrite of the base.

---

## 4. Checks

### 4.1 Built-in checks

Built-in checks (`Field(ge=0)`, `Check.isin([...])`, ...) already exist as
narwhals-expression implementations in
`pandera/backends/narwhals/builtin_checks.py`. Nothing new is needed.

### 4.2 Custom narwhals checks (the default, portable form)

The portable custom-check contract mirrors polars', with `NarwhalsData` in
place of `PolarsData`:

```python
import narwhals as nw
from pandera.api.narwhals.types import NarwhalsData

# expression-style: receives NarwhalsData(frame: nw.LazyFrame, key: str)
pa.Check(lambda data: data.frame.select(nw.col(data.key) > 0))

# element-wise
pa.Check(lambda x: x > 0, element_wise=True)

# in a DataFrameModel
class Model(pa.DataFrameModel):
    a: int

    @pa.check("a")
    def custom(cls, data: NarwhalsData) -> nw.LazyFrame:
        return data.frame.select(nw.col(data.key).is_between(0, 10))

    @pa.dataframe_check
    def df_check(cls, data: NarwhalsData) -> nw.LazyFrame:
        return data.frame.select(nw.col("a") <= nw.col("b"))
```

These run on **every** compatible dataframe type. This is the form the
documentation leads with.

### 4.3 Native-typed custom checks (goal 3)

Users porting existing schemas — or needing an operation narwhals doesn't
expose — can write checks against a **native** frame type. The check
declares its native signature; the narwhals check backend converts at the
boundary and resumes orchestration:

```python
import polars as pl

class Model(pa.DataFrameModel):
    a: int

    @pa.dataframe_check
    def native_check(cls, df: pl.LazyFrame) -> pl.LazyFrame:
        return df.select(pl.col("a") > 0)
```

Mechanism — extend `NarwhalsCheckBackend.apply()` (which already inspects
signatures to choose between 2-arg narwhals-style and 1-arg
`PolarsData`-style calling conventions):

1. Inspect the check function's first-parameter annotation.
2. If it is `NarwhalsData` / `nw.Expr` / `nw.LazyFrame` / unannotated →
   current behavior.
3. If it is a native frame/series type: assert it matches the validated
   object's implementation (`nw.get_native_namespace`); if it doesn't,
   raise a `SchemaDefinitionError` at check-execution time with a message
   naming both types. Then `_to_native` the lazyframe, call the function,
   and re-wrap the output with `nw.from_native` for postprocessing.

**Multi-library checks** use per-implementation registration on a single
check, so one schema can carry native fast paths for several libraries with
the narwhals form as fallback:

```python
@pa.dataframe_check
def custom(cls, data: NarwhalsData) -> nw.LazyFrame:      # portable default
    ...

@custom.register(pl.LazyFrame)                            # singledispatch-style
def _(cls, df: pl.LazyFrame) -> pl.LazyFrame: ...

@custom.register(pd.DataFrame)
def _(cls, df: pd.DataFrame) -> pd.Series: ...
```

This builds on `pandera/api/function_dispatch.py` (already present for
multi-backend built-in check registration). Dispatch key: the native type
of the check object at validation time; fall back to the narwhals default
when no native override matches.

### 4.4 `register_check_method`

`pandera.api.extensions.register_check_method` gains a
`supported_types` entry for narwhals inputs so extension checks written as
`nw.Expr → nw.Expr` become available on the generic API's `Check`
namespace, same as today's polars extension checks.

---

## 5. Typing

The typing design follows the two-track approach established on
`nielsb/fix-series-typing` (commits `d629ae9`, `4d4d88c`, `8bdfd59`):
mypy gets plugin hooks, pyright gets sound-by-construction generics;
**runtime types are never changed by `validate()`** (`cast`, not wrap —
the `4d4d88c` rule).

### 5.1 `pandera/typing/narwhals.py`

```python
from narwhals.typing import IntoDataFrameT, IntoFrameT
import narwhals as nw

class DataFrame(DataFrameBase[T], nw.DataFrame[Any]):
    """Generic eager frame annotation: DataFrame[Model]."""

class LazyFrame(DataFrameBase[T], nw.LazyFrame[Any]):
    """Generic lazy frame annotation: LazyFrame[Model]."""

class Series(SeriesBase[GenericDtype], nw.Series[Any]):
    """Series[dtype] for model fields — optional, fields may also use
    bare dtypes (polars-style)."""
```

- Follows the corrected parameterization from `d629ae9`:
  `DataFrameBase[T]`, not `DataFrameBase, Generic[T]`.
- `DataFrame[Model](native_or_nw_obj)` validates at construction via the
  existing `__patched_generic_alias_call` machinery in `typing/common.py`.
- `AnnotationInfo` and `check_types` learn these types the same way they
  know the polars ones; `typing/__init__.py` adds narwhals entries to
  `dataframe_types`.
- Model fields support **both** annotation styles, like polars: bare
  dtypes (`a: int`) and `Series[int]`. Bare dtypes are the documented
  default since narwhals has no strong standalone-series culture for
  SQL-lazy backends.

### 5.2 Preserving native types through validation

The central static-typing problem for a *generic* API: `validate()` should
return **the same static type it was given** — `pl.LazyFrame` in →
`pl.LazyFrame` out — while `check_types` should accept
`DataFrame[Model]`-annotated functions over any compatible frame.

Approach:

```python
TFrame = TypeVar("TFrame")  # deliberately unbound; runtime enforces compatibility

class DataFrameSchema(...):
    @overload
    def validate(self, check_obj: TFrame, ...) -> TFrame: ...

class DataFrameModel(...):
    @classmethod
    def validate(cls, check_obj: TFrame, ...) -> TFrame: ...
    # documented pattern for schema-tagged returns:
    #   return cast(DataFrame[Model], Model.validate(df))
```

- `TFrame` unbound rather than `bound=IntoFrame`: narwhals'
  `IntoFrame`/`IntoDataFrame` unions only cover libraries narwhals ships
  protocols for and would produce false negatives for compliant
  third-party frames under pyright's strict variance rules. Runtime
  compatibility is enforced by `is_narwhals_compatible` anyway; static
  over-permissiveness here is the lesser evil. (Revisit if narwhals
  stabilizes a `Protocol` for into-frame types.)
- For **`check_types`-decorated functions**, `DataFrame[Model]` /
  `LazyFrame[Model]` annotations are nominal tags exactly as in the pandas
  and polars APIs; at runtime the native object passes through unchanged.

### 5.3 mypy plugin

Extend `pandera/mypy.py` (as reshaped by `d629ae9`):

- Add the narwhals model/typing fullnames to `DATAFRAMEMODEL_FULLNAMES`,
  `FIELD_GENERICS_FULLNAMES`, and the `__getitem__`-hook class list, so
  `df["col"]` on `pandera.typing.narwhals.DataFrame[Model]` resolves to the
  declared field type.
- No new hook kinds are required; the pandas hooks generalize because they
  key on pandera fullnames, not pandas ones.

### 5.4 pyright / Pylance

- The generics in §5.1–5.2 must typecheck cleanly under
  `pyright --strict` **without** any plugin: this is a CI gate
  (`tests/pyright/`, extending the suite added on the typing branch).
- Column-level inference (`df["col"]` → field type) is a **mypy-plugin
  feature only**. Pyright/Pylance users get schema-level types
  (`DataFrame[Model]` retention, native-type preservation) but annotate
  or `cast` for typed column access; no code generation is offered — a
  generate-and-paste workflow was prototyped on the typing branch and
  rejected as poor devex.
- One known compromise: `nw.DataFrame` is itself generic over the native
  frame type; `pandera.typing.narwhals.DataFrame[T]` fixes that parameter
  to `Any`. Users who need the *native* parameter statically should
  annotate with the native type and rely on §5.2 type preservation
  instead.

### 5.5 Test matrix

- `tests/mypy/narwhals_modules/` and `tests/pyright/modules/` gain
  generic-API cases: schema retention through `validate`, `check_types`
  I/O, column access, native-type preservation (pl/pd/duckdb samples).
- Both suites run in CI for every supported Python version, as on the
  typing branch.

---

## 6. Extensibility: Registering New DataFrame Types

Three tiers, from zero-effort to full control:

### Tier 0 — narwhals already supports it (automatic)

Nothing to do. `is_narwhals_compatible` returns True, the narwhals backend
registers on first `validate()`. This covers the long tail by
construction and is the answer for most libraries.

### Tier 1 — declarative registration (`register_frame_type`)

For libraries that need pandera-specific behavior but not a new backend:

```python
from pandera.narwhals import register_frame_type

register_frame_type(
    MyFrame,
    # all optional:
    check_backend=MyCheckBackend,        # override just check execution
    engine_dtypes=my_dtype_registrations, # extra Engine.register_dtype calls
    is_lazy=True,                         # SQL-lazy semantics (see utils._is_sql_lazy)
    materialize=my_collect_fn,            # how to collect for failure cases
)
```

Implementation: a thin façade over the existing primitives —
`BaseSchema.register_backend`, `narwhals_engine.Engine.register_dtype`,
and a new small extension table consulted by
`pandera/api/narwhals/utils.py` (which currently hard-codes the
`Implementation` enum for lazy/materialize dispatch; that table becomes
the single place third parties plug into).

### Tier 2 — full backend override

Unchanged from today: subclass `BaseSchemaBackend`/`BaseCheckBackend` and
`DataFrameSchema.register_backend(MyFrame, MyBackend, force=True)`. This
is also how a *non*-narwhals-compatible library attaches to the generic
API, and how the per-library backends keep working during migration.

### Entry-point discovery

New setuptools entry-point group **`pandera.frame_types`**: a third-party
package exposes `my_lib = my_lib.pandera_plugin:register`, and pandera
invokes it lazily — on the first `BackendNotFoundError` for an unknown
type, before re-raising. This keeps import time flat while letting
`pip install pandera-mylib` be the entire user story. (Pandera has no
entry-point mechanism today; this is the first, so keep the contract
minimal: a zero-arg callable that performs Tier 1/2 registrations.)

---

## 7. Configuration

- `CONFIG.validation_backend` (new, default `"narwhals"`): names the
  default backend family for the generic API, making goal 1's
  "customizable backend" reachable via config as well as via
  `register_backend`. Recognized values: `"narwhals"` plus any name
  registered through Tier 1/2.
- `CONFIG.use_narwhals_backend` (existing) is orthogonal: it toggles the
  narwhals backend under the *per-library* APIs. During the deprecation
  window (§8) it defaults to `True`, then the per-library flag disappears
  along with the per-library backends.
- Validation depth (`PANDERA_VALIDATION_DEPTH`), `PANDERA_VALIDATION_ENABLED`,
  and cache settings apply unchanged.

---

## 8. Replacement Roadmap

The end state: `pandera/api/dataframe/` + `pandera/api/narwhals/` +
`pandera/backends/narwhals/` is **the** implementation, and
`pandera.pandas`, `pandera.polars`, `pandera.pyspark`, `pandera.ibis` are
thin compatibility shims (subclasses that pin typing surface and defaults,
no backend code of their own).

### Phase 0 — prerequisite (done)

Narwhals backend + engine at parity for polars/ibis/pyspark behind
`use_narwhals_backend` (shipped; gaps enumerated in §2.3).

### Phase 1 — generic API (alpha)

- `pandera.narwhals` accessor with `DataFrameSchema`, `Column`,
  `DataFrameModel`, `Field`, `Check`, `BaseConfig`.
- Dynamic backend resolution (§3.3), typing module + mypy/pyright suites
  (§5), native-typed custom checks (§4.3).
- Docs: a "Generic DataFrame validation" guide; the narwhals_backend.md
  page links to it.
- Explicitly labeled experimental; no per-library API changes.

### Phase 2 — parity hardening (beta)

- Close §2.3 gaps that block replacement: column-level coercion,
  `add_missing_columns`/`set_default`, schema IO (YAML/JSON round-trip for
  generic schemas), `drop_invalid_rows` for uniqueness, groupby checks
  (eager backends first).
- Strategies: hypothesis synthesis via the narwhals engine's dtype →
  strategy mapping (may land after GA; not a replacement blocker for
  SQL-lazy backends where strategies never existed).
- Run the shared conformance suite (`tests/common/`) over the generic API
  for every implementation in CI's narwhals matrix.

### Phase 3 — per-library APIs become shims

Order: **ibis → polars → pyspark → pandas** (ascending API-surface risk).
For each library:

1. Concrete `DataFrameSchema`/`Column`/`DataFrameModel` re-parented onto
   the generic classes; `register_default_backends` routes to the narwhals
   backend unconditionally; the per-library backend package is deprecated.
2. Library-specific kwargs/dtypes that the generic API can't express are
   kept on the shim (e.g. pyspark's error-report dict shape) or promoted
   into the generic API when broadly useful.
3. One minor release with the shim opt-in
   (`use_narwhals_backend` default flip), one release deprecated-by-default,
   then backend package removal in the next major version.

### 8.3 The pandas problem (flagged, not solved here)

Narwhals deliberately has no index concept, and modeling
`Index`/`MultiIndex` validation is the single biggest blocker to making
`pandera.pandas` a shim. Options (decide in a follow-up spec):

- (a) shim keeps a thin pandas-native pre-pass that validates
  index/multiindex, then hands the reset-index frame to the narwhals
  backend;
- (b) generic API grows an optional `index=` seam that only
  index-bearing implementations honor;
- (c) `pandera.pandas` retains its native backend indefinitely for
  index-bearing schemas only.

Option (a) is the working assumption: it keeps the generic API clean and
confines index semantics to the pandas shim. geopandas/modin/dask ride on
whatever pandas does.

---

## 9. Generic NDArray Type for the Xarray API

Goal 5 applies the same "generic over compatible implementations" pattern
to the N-dimensional world. There is no "narwhals for labeled arrays", so
the compatibility layer is a **Protocol**, not a translation library.

### 9.1 `NDArrayLike` protocol and `NDArray` annotation

```python
# pandera/typing/ndarray.py
@runtime_checkable
class NDArrayLike(Protocol):
    """Structural type for xarray.DataArray-like objects."""
    @property
    def dims(self) -> tuple[Hashable, ...]: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> Any: ...
    @property
    def coords(self) -> Mapping[Hashable, Any]: ...
    @property
    def attrs(self) -> Mapping[Hashable, Any]: ...

class NDArray(XarrayAnnotationBase, Generic[T]):
    """Annotation generic: NDArray[MyDataArrayModel] — validates any
    NDArrayLike, not just xr.DataArray."""
    __slots__ = ()
```

- `NDArray[Model]` sits beside the existing
  `pandera.typing.xarray.DataArray[Model]` (which stays xr-specific);
  `AnnotationInfo.is_generic_xarray` generalizes to recognize both.
- `DataArraySchema.validate` accepts any `NDArrayLike`; the xarray backend
  already operates through `dims`/`coords`/`dtype`/`attrs` accessors, so
  the change is (i) loosening isinstance gates to the protocol, and
  (ii) routing backend registration through a small
  `register_ndarray_type()` analogous to §6 Tier 1 — this is the
  registration hook for duck-xarray libraries.
- Value-level checks execute through the underlying duck array
  (`.data`), consistent with the existing duck-array support documented in
  `docs/source/xarray_guide/duck_arrays.md`.
- Out of scope here: a full generic `Dataset`/`DataTree` protocol.
  `NDArrayLike` (DataArray-shaped) is the deliberate MVP; dataset-level
  protocols follow only if a concrete second implementation shows up.

---

## 10. Testing & CI

- New CI job group `generic-dataframe`: runs the generic API's own test
  package plus the shared `tests/common/` conformance suite across the
  narwhals matrix (polars, pandas, pyarrow, duckdb, ibis, pyspark,
  modin/dask smoke).
- Static typing gates: `mypy` (plugin on) and `pyright --strict` over
  `tests/{mypy,pyright}` narwhals modules; both must pass for every PR
  touching `pandera/typing/` or `pandera/api/narwhals/`.
- Tier 1 registration is exercised by an in-repo toy frame type
  (a minimal narwhals-compatible wrapper) so extensibility doesn't regress
  silently.
- Phase 3 gate per library: the library's *existing* test suite passes
  with its schema classes re-parented onto the generic implementation.

---

## 11. Open Questions

1. **Series-level API**: should the generic API expose a standalone
   `SeriesSchema` (pandas has one; polars/ibis don't)? Current position:
   no — columns-in-frames only, matching narwhals' center of gravity.
2. **`pandera.generic` alias**: worth shipping `pandera.generic` as an
   alias of `pandera.narwhals` from day one so user code doesn't reference
   the backend name? Leaning yes, as a pure re-export.
3. **Error-report shape**: pyspark's dict-style error report differs from
   the `SchemaErrors` exception model. Unify on `SchemaErrors` +
   `use_pyspark_error_format` shim flag, or keep divergent?
4. **Strategies scope**: is hypothesis synthesis a GA blocker for the
   generic API, or an eager-backends-only feature indefinitely?
5. **Entry-point security posture**: lazy plugin invocation on
   `BackendNotFoundError` executes third-party code implicitly; do we
   require an explicit `pandera.load_plugins()` opt-in instead?
6. **narwhals version pinning**: the generic API's public behavior now
   tracks narwhals' `Implementation` coverage; define a minimum-version
   policy and a compatibility test against narwhals `main` (nightly job?).
