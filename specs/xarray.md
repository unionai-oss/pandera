# Xarray Integration Spec

> **Status:** Draft
> **Issue:** [#705](https://github.com/unionai-oss/pandera/issues/705)
> **Author:** pandera maintainers
> **Related work:**
> [xarray-schema](https://github.com/xarray-contrib/xarray-schema),
> [xarray-validate](https://github.com/leroyvn/xarray-validate)

---

## 1. Motivation

[xarray](https://docs.xarray.dev) is the dominant Python library for
labelled, N-dimensional array data. It is widely used in climate science,
remote sensing, genomics, and any domain that works with multi-dimensional
gridded data.

Pandera already supports tabular backends (pandas, polars, pyspark, ibis).
Adding xarray extends pandera's reach to the N-dimensional array world,
providing a single, consistent validation API across both tabular and array
data.

Two community projects — `xarray-schema` and its fork `xarray-validate` —
demonstrate demand for this feature but are either unmaintained or limited in
scope. Per the [discussion on #705](https://github.com/unionai-oss/pandera/issues/705),
both the pandera maintainers and the community prefer housing this
functionality inside pandera itself, with a design that keeps the public API
surface small (i.e. schema objects for `DataArray`, `Dataset`, and `DataTree`
only, with all other concerns expressed as keyword arguments or `Check`
objects — **not** as standalone schema component classes like `DTypeSchema`,
`DimsSchema`, `ShapeSchema`, etc.).

---

## 2. xarray Data Model Primer

Reference: [xarray data structures](https://docs.xarray.dev/en/stable/user-guide/data-structures.html),
[terminology](https://docs.xarray.dev/en/stable/user-guide/terminology.html).

### 2.1 Core Container Types

| xarray type | Description |
|---|---|
| `xr.DataArray` | A single N-dimensional array with labelled dimensions, coordinates, attributes, and a name. Analogous to a pandas `Series`. |
| `xr.Dataset` | A dict-like container of `DataArray` objects that share dimensions and coordinates. Analogous to a pandas `DataFrame`. |
| `xr.DataTree` | A [tree of `Dataset` nodes](https://docs.xarray.dev/en/stable/user-guide/hierarchical-data.html) (added in xarray 2024.10). Nodes are linked parent-to-child, each holding a `Dataset`. Accessed via filesystem-like paths (e.g. `tree["/group/subgroup"]`). |

### 2.2 Key Concepts

- **dims** — ordered tuple of dimension names (e.g. `("time", "lat", "lon")`).
  Each dimension has a name and a length.
- **sizes** — a `Mapping[str, int]` from dimension names to their lengths
  (e.g. `{"time": 12, "lat": 180, "lon": 360}`). This is the idiomatic
  xarray way to refer to dimension sizes and is often preferred over
  positional `shape`.
- **shape** — tuple of integers describing array dimensions, positional
  (same as NumPy `ndarray.shape`).
- **coords** — dict-like mapping of coordinate names to `DataArray` values.
  Coordinates are a superset of dims. xarray distinguishes several kinds:
  - **Dimension coordinate** — a 1-D coordinate whose name matches one of
    the dimension names (e.g. a `time` coordinate on the `time` dim).
    Usually indexed and usable for label-based selection/alignment.
  - **Non-dimension coordinate** ("auxiliary coordinate") — a coordinate
    whose name is *not* in `dims`. Can be multi-dimensional (e.g. 2-D
    `lat`/`lon` arrays on a projected grid).
  - **Indexed coordinate** — a coordinate with an associated `Index`
    object (by default `PandasIndex`), enabling `sel()`-style
    label-based indexing.
  - **Non-indexed coordinate** — a coordinate without an associated
    `Index`. Represents fixed labels but cannot be used for
    label-based selection.
- **data_vars** — the named data variables in a `Dataset`, each a
  `DataArray`.
- **attrs** — arbitrary metadata dict on any xarray object. xarray does
  not interpret attrs; propagation is limited to unambiguous cases.
- **dtype** — NumPy (or compatible) dtype of the underlying array data.
- **name** — optional string name on a `DataArray`.
- **encoding** — a dict on `Variable`/`DataArray`/`Dataset` that controls
  serialization to formats like netCDF and Zarr. Common keys include
  `_FillValue`, `scale_factor`, `add_offset`, `dtype`, `compression`.
  May be worth validating in data pipeline contexts.
- **chunks** — Dask chunk structure (when the underlying array is a
  `dask.array.Array`). `None` when the array is eager (e.g. NumPy).
- **Variable** — a low-level NetCDF-like object consisting of `dims`,
  `data`, `attrs`, and `encoding`. Each `DataArray` wraps a `Variable`
  (accessible via `da.variable`). **`Variable` is intentionally not
  exposed as a pandera schema component** — it is an internal xarray
  building block that users rarely interact with directly.

### 2.3 Duck Array Backends

xarray wraps [duck arrays](https://docs.xarray.dev/en/stable/user-guide/duckarrays.html)
— any array type implementing the NumPy-like interface. Supported backends
include:

| Backend | Description |
|---|---|
| `numpy.ndarray` | Default in-memory array (eager) |
| `dask.array.Array` | Lazy, parallel, larger-than-memory computation |
| `sparse.COO` / `sparse.DOK` | Sparse arrays (memory-efficient for mostly-zero data) |
| `cupy.ndarray` | GPU-accelerated arrays |
| `pint.Quantity` (via `pint-xarray`) | Arrays with physical units |

Validation should be possible regardless of the underlying array backend.
The `array_type` schema parameter (see Section 4.2) can optionally
constrain which backend is expected.

### 2.4 DataTree Specifics

Key `DataTree` characteristics relevant to validation:

- **Coordinate inheritance** — coordinates defined on a parent node are
  inherited by all child nodes. A child's `coords` includes both its own
  and its ancestors' coordinates.
- **Path-based access** — nodes are addressed by filesystem-like paths
  (e.g. `"/surface/temperature"`). `DataTree.from_dict()` constructs
  trees from `{path: Dataset}` dicts.
- **`groups`** — tuple of all group paths in the tree.
- **`isomorphic()`** — checks whether two trees have the same node
  structure (paths match), regardless of data.
- Each node has `dataset`, `data_vars`, `coords`, `attrs`, `dims`, and
  `children` properties.

---

## 3. Design Principles

### 3.1 Small Public API Surface

The community feedback on #705 is clear:
[exposing many low-level components](https://github.com/unionai-oss/pandera/issues/705#issuecomment-3888722278)
like `DTypeSchema`, `DimsSchema`, `ShapeSchema`, etc. is too heavy.
Instead, the public API should consist of **three schema classes** and
**three model classes** — one pair for each xarray container type:

| Schema class | Model class | Validates |
|---|---|---|
| `DataArraySchema` | `DataArrayModel` | `xr.DataArray` |
| `DatasetSchema` | `DatasetModel` | `xr.Dataset` |
| `DataTreeSchema` | `DataTreeModel` | `xr.DataTree` |

All other validation concerns (dims, shape, coords, attrs, chunks) are
expressed as **keyword arguments** on these classes, or as pandera `Check`
objects. This mirrors how pandera's pandas API uses `DataFrameSchema` and
`Column` rather than exposing separate schema objects for dtype, nullability,
uniqueness, etc.

### 3.2 Consistent with Existing Pandera Patterns

The implementation must follow pandera's layered architecture:

1. **API layer** (`pandera/api/xarray/`) — schema and model definitions
2. **Backend layer** (`pandera/backends/xarray/`) — validation logic
3. **Engine layer** (`pandera/engines/xarray_engine.py`) — dtype resolution

Backend registration, lazy validation, `Check` integration, and
`@check_types` decorator support must all follow the patterns established by
the polars and pandas backends.

### 3.3 The `DataArray` is the Fundamental Schema Component

In the pandas world, `Column` is the schema component that describes a single
series inside a `DataFrameSchema`. In the xarray world, `DataArray` plays the
analogous role: a `DatasetSchema` contains a mapping of variable names to
`DataArraySchema` objects.

- `DataArraySchema` → analogous to `Column` (component of a dataset) AND a
  standalone schema (like `SeriesSchema`)
- `DatasetSchema` → analogous to `DataFrameSchema` (container of components)
- `DataTreeSchema` → a new concept: a tree of `DatasetSchema` nodes

### 3.4 Leverage `Check` for Data-Level Validation

Structural validation (dims, coords, dtype, shape, attrs) is handled by
schema keyword arguments. Data-level validation (value ranges, statistical
properties, custom predicates) is handled by pandera's existing `Check`
system, extended to support xarray objects.

---

## 4. Public API Design

### 4.1 Entry Point

```python
import pandera.xarray as pa
```

This module exposes:

```python
__all__ = [
    "Check",
    "DataArraySchema",
    "DatasetSchema",
    "DataTreeSchema",
    "DataArrayModel",
    "DatasetModel",
    "DataTreeModel",
    "Field",
    "check_input",
    "check_output",
    "check_io",
    "check_types",
    "errors",
]
```

### 4.2 `DataArraySchema`

`DataArraySchema` is the atomic unit of xarray validation. It validates a
single `xr.DataArray`.

```python
class DataArraySchema(BaseSchema):
    def __init__(
        self,
        dtype: DTypeLike | None = None,
        dims: tuple[str | None, ...] | None = None,
        sizes: dict[str, int | None] | None = None,
        shape: tuple[int | None, ...] | None = None,
        coords: dict[str, DataArraySchema] | list[str] | None = None,
        attrs: dict[str, Any] | None = None,
        name: str | None = None,
        checks: Check | list[Check] | None = None,
        parsers: Parser | list[Parser] | None = None,
        coerce: bool = False,
        nullable: bool = False,
        chunked: bool | None = None,
        array_type: type | None = None,
        strict_coords: bool = False,
        strict_attrs: bool = False,
        title: str | None = None,
        description: str | None = None,
        metadata: dict | None = None,
    ):
        ...

    def validate(
        self,
        check_obj: xr.DataArray,
        lazy: bool = False,
        inplace: bool = False,
    ) -> xr.DataArray:
        ...
```

**Parameter semantics:**

| Parameter | Type | Description |
|---|---|---|
| `dtype` | numpy dtype, type, or string | Expected dtype of the underlying data (resolved via `np.issubdtype`). Coerced if `coerce=True`. |
| `dims` | `tuple[str \| None, ...]` | Expected dimension names, in order. `None` entries act as wildcards (match any dim name). Length of the tuple also constrains `ndim`. |
| `sizes` | `dict[str, int \| None]` | Expected dimension sizes as a name→length mapping (e.g. `{"lat": 180, "lon": 360}`). More idiomatic than positional `shape` for xarray. `None` values act as wildcards. Mutually exclusive with `shape`. |
| `shape` | `tuple[int \| None, ...]` | Expected positional shape. `None` entries act as wildcards for that axis. Mutually exclusive with `sizes`. |
| `coords` | `dict[str, DataArraySchema] \| list[str]` | If a dict, mapping of coordinate name to a `DataArraySchema` that validates the coordinate's values. If a list of strings, shorthand for "these coordinate names must exist" (no value validation). |
| `attrs` | `dict[str, Any]` | Expected attributes. Values are matched for equality; use `Check` for complex attr validation. |
| `name` | `str` | Expected `.name` attribute. |
| `checks` | `Check` or list | Pandera `Check` objects for data-level validation. |
| `parsers` | `Parser` or list | Pandera `Parser` objects for data transformation. |
| `coerce` | `bool` | If `True`, coerce dtype before validation via `da.astype()`. |
| `nullable` | `bool` | If `False` (default), NaN/null values raise a validation error. |
| `chunked` | `bool \| None` | If `True`, require the array to be Dask-backed (i.e. `da.chunks is not None`). If `False`, require it to be eager. If `None` (default), don't check. |
| `array_type` | `type \| None` | Expected type of the underlying array (e.g. `numpy.ndarray`, `dask.array.Array`, `sparse.COO`). Checked via `isinstance(da.data, array_type)`. If `None`, any duck array backend is accepted. |
| `strict_coords` | `bool` | If `True`, fail when the DataArray has coordinates not listed in `coords`. Default `False`. |
| `strict_attrs` | `bool` | If `True`, fail when the DataArray has attributes not listed in `attrs`. Default `False`. |

**Example — imperative API:**

```python
import numpy as np
import xarray as xr
import pandera.xarray as pa

schema = pa.DataArraySchema(
    dtype=np.float64,
    dims=("time", "lat", "lon"),
    shape=(None, 180, 360),  # time dimension is unconstrained
    coords={
        "time": pa.DataArraySchema(dtype="datetime64[ns]"),
        "lat": pa.DataArraySchema(
            dtype=np.float64,
            checks=pa.Check.in_range(-90, 90),
        ),
        "lon": pa.DataArraySchema(
            dtype=np.float64,
            checks=pa.Check.in_range(-180, 180),
        ),
    },
    name="temperature",
    checks=pa.Check.in_range(150, 350),  # Kelvin
)

da = xr.DataArray(
    np.random.uniform(200, 300, (12, 180, 360)),
    dims=("time", "lat", "lon"),
    coords={
        "time": pd.date_range("2020", periods=12, freq="MS"),
        "lat": np.linspace(-89.5, 89.5, 180),
        "lon": np.linspace(-179.5, 179.5, 360),
    },
    name="temperature",
)

validated = schema.validate(da)
```

### 4.3 `DatasetSchema`

`DatasetSchema` validates an `xr.Dataset`. It contains a dict of
`DataArraySchema` objects — one per data variable — plus optional
dataset-level coordinate, attribute, and check specifications.

A `Dataset`'s dimensions are the union of the dimensions across all its
data variables. The `sizes` parameter on `DatasetSchema` validates
dataset-level dimension sizes, while per-variable dimensions are validated
via the `DataArraySchema` in `data_vars`.

```python
class DatasetSchema(BaseSchema):
    def __init__(
        self,
        data_vars: dict[str, DataArraySchema | None] | None = None,
        coords: dict[str, DataArraySchema] | list[str] | None = None,
        dims: tuple[str, ...] | None = None,
        sizes: dict[str, int | None] | None = None,
        attrs: dict[str, Any] | None = None,
        checks: Check | list[Check] | None = None,
        parsers: Parser | list[Parser] | None = None,
        strict: bool | str = False,
        strict_coords: bool = False,
        strict_attrs: bool = False,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        metadata: dict | None = None,
    ):
        ...

    def validate(
        self,
        check_obj: xr.Dataset,
        lazy: bool = False,
        inplace: bool = False,
    ) -> xr.Dataset:
        ...
```

**Parameter semantics:**

| Parameter | Type | Description |
|---|---|---|
| `data_vars` | `dict[str, DataArraySchema \| None]` | Mapping of variable names to their `DataArraySchema`. A `None` value means "must exist, but no further validation". |
| `coords` | `dict[str, DataArraySchema] \| list[str]` | Dataset-level coordinate schemas. If a dict, validates coordinate values. If a list, only checks coordinate existence. |
| `dims` | `tuple[str, ...]` | Expected dataset-level dimension names (the union of all variable dims). |
| `sizes` | `dict[str, int \| None]` | Expected dataset-level dimension sizes as a name→length mapping. |
| `attrs` | `dict[str, Any]` | Expected dataset-level attributes. |
| `checks` | `Check` or list | Dataset-wide checks (receive the full `xr.Dataset`). |
| `strict` | `bool \| "filter"` | If `True`, fail on unexpected data variables. If `"filter"`, drop them. |
| `strict_coords` | `bool` | If `True`, fail on unexpected coordinates. |
| `strict_attrs` | `bool` | If `True`, fail on unexpected attributes. |

**Example — imperative API:**

```python
schema = pa.DatasetSchema(
    data_vars={
        "temperature": pa.DataArraySchema(
            dtype=np.float64,
            dims=("time", "lat", "lon"),
            checks=pa.Check.in_range(150, 350),
        ),
        "precipitation": pa.DataArraySchema(
            dtype=np.float64,
            dims=("time", "lat", "lon"),
            checks=pa.Check.ge(0),
        ),
    },
    coords={
        "time": pa.DataArraySchema(dtype="datetime64[ns]"),
        "lat": pa.DataArraySchema(
            dtype=np.float64,
            checks=pa.Check.in_range(-90, 90),
        ),
        "lon": pa.DataArraySchema(
            dtype=np.float64,
            checks=pa.Check.in_range(-180, 180),
        ),
    },
)

ds = xr.Dataset({...})
validated = schema.validate(ds)
```

### 4.4 `DataTreeSchema`

`DataTreeSchema` validates an `xr.DataTree` — a hierarchical tree of
datasets. Since `DataTree` is relatively new (xarray >= 2024.10), this is
lower priority and may ship in a follow-up release.

`DataTree` has an important **coordinate inheritance** model: coordinates
defined on a parent node are automatically inherited by all descendant
nodes. The schema must account for this — a child node's `coords` is the
merge of its own coordinates with its ancestors'.

`DataTreeSchema` supports two ways of specifying child schemas: by
**direct child name** (single level) and by **path** (multi-level, using
`"/"` separators), mirroring `DataTree.from_dict()`:

```python
class DataTreeSchema(BaseSchema):
    def __init__(
        self,
        children: dict[str, DatasetSchema | DataTreeSchema] | None = None,
        dataset: DatasetSchema | None = None,
        attrs: dict[str, Any] | None = None,
        checks: Check | list[Check] | None = None,
        strict: bool = False,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        metadata: dict | None = None,
    ):
        ...

    def validate(
        self,
        check_obj: xr.DataTree,
        lazy: bool = False,
    ) -> xr.DataTree:
        ...
```

**Parameter semantics:**

| Parameter | Type | Description |
|---|---|---|
| `children` | `dict[str, DatasetSchema \| DataTreeSchema]` | Mapping of child node names (or `/`-separated paths) to their schemas. Nesting allows recursive tree validation. |
| `dataset` | `DatasetSchema` | Schema for the dataset attached to this node (its own `data_vars`, `coords`, `attrs`). |
| `attrs` | `dict[str, Any]` | Expected node-level attributes. |
| `strict` | `bool` | If `True`, fail on unexpected child nodes. |

**Example — path-based tree schema:**

```python
schema = pa.DataTreeSchema(
    dataset=pa.DatasetSchema(attrs={"conventions": "CF-1.8"}),
    children={
        "surface": pa.DatasetSchema(
            data_vars={
                "temperature": pa.DataArraySchema(
                    dtype=np.float64,
                    dims=("time", "lat", "lon"),
                ),
            },
        ),
        "surface/diagnostics": pa.DatasetSchema(
            data_vars={"rmse": pa.DataArraySchema(dtype=np.float64)},
        ),
    },
    strict=True,
)
```

### 4.5 Class-Based Models (Declarative API)

Following pandera's `DataFrameModel` pattern, class-based models use type
annotations and `Field()` descriptors to declare schemas.

#### `DataArrayModel`

```python
import pandera.xarray as pa

class Temperature(pa.DataArrayModel):
    dtype = np.float64
    dims = ("time", "lat", "lon")
    name = "temperature"

    class Config:
        checks = [pa.Check.in_range(150, 350)]
        coords = {
            "lat": pa.DataArraySchema(
                dtype=np.float64,
                checks=pa.Check.in_range(-90, 90),
            ),
        }

da = Temperature.validate(da)
```

For `DataArrayModel`, fields on the class body represent **coordinates**,
allowing access to coordinate names as class attributes (addressing the
use case raised by [@avcopan](https://github.com/unionai-oss/pandera/issues/705#issuecomment-2562316655)):

```python
class RateConstant(pa.DataArrayModel):
    pressure: np.float64
    temperature: np.float64

    class Config:
        dtype = np.float64

RateConstant.validate(da)

# Access coordinate names as class attributes
print(RateConstant.pressure)   # "pressure"
print(RateConstant.temperature)  # "temperature"
```

#### `DatasetModel`

For `DatasetModel`, fields represent **data variables**. Type annotations
map to `DataArraySchema` instances with the annotated dtype. Coordinate
and attribute schemas are defined in the `Config` class.

```python
class ClimateData(pa.DatasetModel):
    temperature: np.float64 = pa.Field(
        dims=("time", "lat", "lon"),
        checks=pa.Check.in_range(150, 350),
    )
    precipitation: np.float64 = pa.Field(
        dims=("time", "lat", "lon"),
        checks=pa.Check.ge(0),
    )

    class Config:
        coords = {
            "time": pa.DataArraySchema(dtype="datetime64[ns]"),
            "lat": pa.DataArraySchema(dtype=np.float64),
            "lon": pa.DataArraySchema(dtype=np.float64),
        }

ClimateData.validate(ds)

# Access data variable names as class attributes
print(ClimateData.temperature)   # "temperature"
print(ClimateData.precipitation) # "precipitation"
```

#### `DataTreeModel`

```python
class ClimateTree(pa.DataTreeModel):
    surface: ClimateData  # reference to a DatasetModel
    upper_atmosphere: UpperAtmData

    class Config:
        strict = True

ClimateTree.validate(dt)
```

### 4.6 `Field` Descriptor

`Field()` provides per-variable configuration in model classes, similar to
its role in `DataFrameModel`:

```python
def Field(
    dtype: DTypeLike | None = None,
    dims: tuple[str, ...] | None = None,
    sizes: dict[str, int | None] | None = None,
    shape: tuple[int | None, ...] | None = None,
    checks: Check | list[Check] | None = None,
    nullable: bool = False,
    coerce: bool = False,
    title: str | None = None,
    description: str | None = None,
    metadata: dict | None = None,
    alias: str | None = None,
) -> FieldInfo:
    ...
```

### 4.7 Decorator Support

The existing `@check_types`, `@check_input`, `@check_output`, and
`@check_io` decorators should work with xarray type annotations:

```python
from pandera.typing.xarray import DataArray, Dataset

@pa.check_types
def process(
    ds: Dataset[ClimateData],
) -> Dataset[ClimateData]:
    ...
```

---

## 5. Architecture

### 5.1 Module Layout

```
pandera/
├── api/
│   └── xarray/
│       ├── __init__.py
│       ├── container.py        # DataArraySchema, DatasetSchema, DataTreeSchema
│       ├── model.py            # DataArrayModel, DatasetModel, DataTreeModel
│       ├── model_config.py     # BaseConfig for xarray models
│       ├── model_components.py # FieldInfo for xarray
│       ├── types.py            # type aliases (XarrayCheckObjects, etc.)
│       └── utils.py            # validation depth helpers
├── backends/
│   └── xarray/
│       ├── __init__.py
│       ├── base.py             # XarraySchemaBackend (shared logic)
│       ├── data_array.py       # DataArraySchemaBackend
│       ├── dataset.py          # DatasetSchemaBackend
│       ├── data_tree.py        # DataTreeSchemaBackend
│       ├── checks.py           # XarrayCheckBackend
│       ├── builtin_checks.py   # xarray-specific built-in checks
│       └── register.py         # register_xarray_backends()
├── engines/
│   └── xarray_engine.py        # Engine + dtype registry for xarray
├── typing/
│   └── xarray.py               # DataArray[T], Dataset[T] generic types
└── xarray.py                   # Entry point: import pandera.xarray as pa
```

### 5.2 API Layer

#### `pandera/api/xarray/types.py`

```python
from typing import NamedTuple, Union
import xarray as xr

class XarrayData(NamedTuple):
    """Container passed to Check backends."""
    obj: Union[xr.DataArray, xr.Dataset]
    key: str = "*"

XarrayCheckObjects = Union[xr.DataArray, xr.Dataset, xr.DataTree]
```

#### `pandera/api/xarray/container.py`

The `DataArraySchema` does **not** inherit from
`pandera.api.dataframe.container.DataFrameSchema` because xarray's data
model is fundamentally different from the tabular model (dimensions and
coordinates vs. rows and columns). Instead, it directly inherits from
`BaseSchema`:

```python
from pandera.api.base.schema import BaseSchema

class DataArraySchema(BaseSchema):
    ...

class DatasetSchema(BaseSchema):
    ...

class DataTreeSchema(BaseSchema):
    ...
```

Each schema class overrides `register_default_backends` to point at the
xarray registration function, and `validate()` to delegate to the
appropriate backend.

#### `pandera/api/xarray/model.py`

The model classes inherit from `BaseModel` and implement `to_schema()` to
produce the corresponding schema object:

```python
from pandera.api.base.model import BaseModel

class DataArrayModel(BaseModel):
    @classmethod
    def to_schema(cls) -> DataArraySchema:
        ...

class DatasetModel(BaseModel):
    @classmethod
    def to_schema(cls) -> DatasetSchema:
        ...

class DataTreeModel(BaseModel):
    @classmethod
    def to_schema(cls) -> DataTreeSchema:
        ...
```

### 5.3 Backend Layer

#### Registration (`pandera/backends/xarray/register.py`)

```python
from functools import lru_cache
import xarray as xr

@lru_cache
def register_xarray_backends():
    from pandera.api.checks import Check
    from pandera.api.xarray.container import (
        DataArraySchema,
        DatasetSchema,
        DataTreeSchema,
    )
    from pandera.backends.xarray.data_array import DataArraySchemaBackend
    from pandera.backends.xarray.dataset import DatasetSchemaBackend
    from pandera.backends.xarray.data_tree import DataTreeSchemaBackend
    from pandera.backends.xarray.checks import XarrayCheckBackend

    DataArraySchema.register_backend(xr.DataArray, DataArraySchemaBackend)
    DatasetSchema.register_backend(xr.Dataset, DatasetSchemaBackend)
    DataTreeSchema.register_backend(xr.DataTree, DataTreeSchemaBackend)
    Check.register_backend(xr.DataArray, XarrayCheckBackend)
    Check.register_backend(xr.Dataset, XarrayCheckBackend)
```

#### Validation Flow

**`DataArraySchemaBackend.validate()`:**

1. Check dtype (coerce if needed via `da.astype()`)
2. Check name
3. Check dims (names and order)
4. Check sizes (named dimension lengths) OR shape (positional lengths),
   with wildcard support for `None` values
5. Check coords — for each entry in the `coords` schema:
   - Verify the coordinate exists on the DataArray
   - If the coord schema is a `DataArraySchema`, delegate to
     `DataArraySchemaBackend.validate()` on `da.coords[name]`
   - Distinguish dimension coordinates (name in `da.dims`) from
     non-dimension/auxiliary coordinates (name not in `da.dims`) for
     error reporting
6. Check strict_coords (reject unexpected coordinates)
7. Check attrs (equality matching) and strict_attrs
8. Check chunked status (`da.chunks is not None`)
9. Check array_type (`isinstance(da.data, array_type)`)
10. Check nullable (scan for NaN/null via `da.isnull().any()`)
11. Run `Check` objects via `run_checks()`
12. Collect errors in `ErrorHandler` (lazy mode) or raise immediately

**`DatasetSchemaBackend.validate()`:**

1. Check dataset-level dims (the union of all variable dims)
2. Check dataset-level sizes
3. Check data variable presence; enforce strict mode
4. For each data variable schema, delegate to
   `DataArraySchemaBackend.validate()` on `ds[var_name]`
5. Check dataset-level coords (shared across variables)
6. Check dataset-level attrs and strict_attrs
7. Run dataset-level checks
8. Collect errors or raise

**`DataTreeSchemaBackend.validate()`:**

1. Validate node-level dataset via `DatasetSchemaBackend` (if `dataset`
   schema is specified)
2. Validate node-level attrs
3. For each child in `children` schema:
   - Resolve the child node (supports both direct names and
     `/`-separated paths)
   - Recursively validate via `DatasetSchemaBackend` or
     `DataTreeSchemaBackend` depending on the child schema type
4. Enforce strict mode on child nodes (fail on unexpected children)
5. Account for coordinate inheritance: inherited coords from parent
   nodes are visible on child nodes and should be validated accordingly

### 5.4 Engine Layer

`pandera/engines/xarray_engine.py` registers NumPy dtypes for xarray.
Since xarray arrays are typically backed by NumPy (or Dask wrapping NumPy),
the engine mostly delegates to `numpy_engine` for dtype resolution:

```python
from pandera.engines import engine, numpy_engine

class DataType(numpy_engine.DataType):
    """xarray-compatible data type."""

    def coerce(self, data_container: xr.DataArray) -> xr.DataArray:
        return data_container.astype(self.type)

    def check(self, pandera_dtype, data_container=None):
        return np.issubdtype(data_container.dtype, self.type)

class Engine(metaclass=engine.Engine, base_pandera_dtypes=DataType):
    pass
```

The engine should handle:

- **Standard NumPy dtypes** — `float64`, `int32`, `bool`, etc.
- **Abstract NumPy type hierarchy** — `np.floating`, `np.integer`,
  `np.signedinteger`, etc. (validated via `np.issubdtype`)
- **Datetime/timedelta dtypes** — `datetime64[ns]`, `timedelta64[s]`,
  with unit-aware comparison
- **String dtypes** — NumPy `U` (fixed-width) and `object` (Python str)
- **Duck array dtypes** — when xarray wraps duck arrays (dask, sparse,
  cupy, pint), the dtype is still accessible via `da.dtype` and follows
  NumPy conventions. The engine should not require materializing lazy
  arrays to check dtype.
- **Encoding dtypes** — when data is read from netCDF/Zarr, the on-disk
  dtype (in `encoding["dtype"]`) may differ from the in-memory dtype
  (due to `scale_factor`/`add_offset` unpacking). The engine validates
  the in-memory dtype by default.

### 5.5 Type Annotations (`pandera/typing/xarray.py`)

Generic type wrappers for use with `@check_types`:

```python
from typing import Generic, TypeVar
import xarray as xr

T = TypeVar("T")

class DataArray(xr.DataArray, Generic[T]):
    """Annotation type for validated DataArrays."""
    ...

class Dataset(xr.Dataset, Generic[T]):
    """Annotation type for validated Datasets."""
    ...

class DataTree(xr.DataTree, Generic[T]):
    """Annotation type for validated DataTrees."""
    ...
```

---

## 6. Check System Integration

### 6.1 Built-in Checks for xarray

Most existing pandera built-in checks (e.g. `ge`, `le`, `in_range`, `isin`,
`notnull`, `str_matches`) operate on element-wise values and should work on
`xr.DataArray` with minimal adaptation, since DataArrays support the same
comparison operators as pandas Series.

Additional xarray-specific built-in checks:

| Check | Description |
|---|---|
| `Check.has_dims(*dims)` | Assert that specific dims exist (order-independent) |
| `Check.has_coords(*coords)` | Assert that specific coordinates exist |
| `Check.has_attrs(**attrs)` | Assert specific attribute key-value pairs |
| `Check.ndim(n)` | Assert number of dimensions |
| `Check.dim_size(dim, size)` | Assert a specific dimension has a given size |
| `Check.is_monotonic(dim, increasing=True)` | Assert that a dimension coordinate is monotonically increasing or decreasing (common for time, lat, lon) |
| `Check.no_duplicates_in_coord(coord)` | Assert that a coordinate has no duplicate values |

### 6.2 Custom Checks

Users can register custom checks via `@pandera.extensions.register_check_method`
for xarray objects, following the same pattern as other backends:

```python
import pandera.extensions as extensions

@extensions.register_check_method(statistics=["threshold"])
def has_positive_mean(xr_obj, *, threshold=0):
    return float(xr_obj.mean()) > threshold
```

### 6.3 `XarrayCheckBackend`

The check backend receives an `XarrayData` named tuple and applies the
check function. For element-wise checks, the result is a boolean
`xr.DataArray`. For aggregate checks, the result is a scalar `bool`.

```python
class XarrayCheckBackend(BaseCheckBackend):
    def __call__(self, check_obj, key=None):
        ...

    def apply(self, check_obj):
        result = self.check._check_fn(check_obj)
        if isinstance(result, xr.DataArray):
            return result  # element-wise boolean mask
        return result  # scalar bool

    def postprocess(self, check_obj, check_output):
        # Convert xr.DataArray[bool] → failure cases
        ...
```

---

## 7. Error Reporting

Errors follow pandera's existing hierarchy:

- `SchemaError` — raised on the first validation failure (eager mode)
- `SchemaErrors` — collects all failures (lazy mode, `lazy=True`)
- `SchemaInitError` — raised on invalid schema definitions

### 7.1 Structural Errors

Structural validation failures (wrong dims, wrong dtype, missing coords)
produce clear, actionable messages:

```python
SchemaError(
    "DataArraySchema 'temperature' failed validation: "
    "expected dims ('time', 'lat', 'lon'), got ('time', 'x', 'y')"
)
```

### 7.2 Data-Level Errors

For data-level `Check` failures on N-dimensional arrays, failure cases
include coordinate context for each failing element:

```python
{
    "schema": "DataArraySchema",
    "column": "temperature",      # data_var or coord name
    "check": "in_range(150, 350)",
    "failure_case": 149.2,
    "index": {"time": "2020-01-01", "lat": 45.0, "lon": -120.0},
}
```

The multi-dimensional index is flattened into a dict of dimension → value
pairs for each failing element, making errors interpretable for
N-dimensional data. For DataArrays without dimension coordinates on some
axes, the positional index is used instead.

### 7.3 DataTree Error Paths

For `DataTree` validation, errors include the tree path to the failing
node, enabling users to locate errors in deeply nested structures:

```python
SchemaError(
    "DataTreeSchema failed at path '/surface/diagnostics': "
    "data variable 'rmse' not found in Dataset"
)
```

---

## 8. Configuration Integration

The xarray backend respects pandera's global configuration:

- `PANDERA_VALIDATION_ENABLED` — disable all validation
- `PANDERA_VALIDATION_DEPTH` — `SCHEMA_ONLY`, `DATA_ONLY`,
  `SCHEMA_AND_DATA`
- `config_context(...)` — context manager for local overrides

`SCHEMA_ONLY` skips data-level `Check` execution and only validates
structural properties (dims, shape, dtype, coords, attrs).

---

## 9. Optional Dependencies

In `pyproject.toml`:

```toml
[project.optional-dependencies]
xarray = ["xarray >= 2024.1.0", "numpy >= 1.24.4"]
```

The `xarray` extra is added to the `all` meta-extra. The minimum xarray
version is 2024.1.0 to ensure modern API compatibility. `DataTree` support
requires xarray >= 2024.10 (when `DataTree` was promoted to the main
namespace); this is handled as a runtime feature check, not a hard minimum.

---

## 10. Implementation Plan

### Phase 1: Core DataArray + Dataset (MVP)

**Goal:** Ship `DataArraySchema`, `DatasetSchema`, and their backends.

| Task | Details |
|---|---|
| API layer | `pandera/api/xarray/` — `DataArraySchema`, `DatasetSchema`, types, utils |
| Backend layer | `pandera/backends/xarray/` — `DataArraySchemaBackend`, `DatasetSchemaBackend`, `XarrayCheckBackend`, built-in checks, `register.py` |
| Engine layer | `pandera/engines/xarray_engine.py` — NumPy-based dtype registry |
| Entry point | `pandera/xarray.py` |
| Type stubs | `pandera/typing/xarray.py` |
| Tests | `tests/xarray/` — structural validation, data checks, lazy mode, coercion, error reporting |
| Dependencies | `pyproject.toml` — add `xarray` extra |

### Phase 2: Class-Based Models

**Goal:** Ship `DataArrayModel` and `DatasetModel`.

| Task | Details |
|---|---|
| Model classes | `pandera/api/xarray/model.py` — `DataArrayModel`, `DatasetModel` |
| `Field` descriptor | `pandera/api/xarray/model_components.py` — xarray-specific `FieldInfo` |
| Decorator support | `@check_types` with `DataArray[T]` and `Dataset[T]` |
| Tests | Model-based validation, `@check_types` integration |

### Phase 3: DataTree Support

**Goal:** Ship `DataTreeSchema` and `DataTreeModel`.

| Task | Details |
|---|---|
| API | `DataTreeSchema`, `DataTreeModel` |
| Backend | `DataTreeSchemaBackend` |
| Tests | Tree-structured validation, nested schemas |

### Phase 4: Advanced Features

| Feature | Details |
|---|---|
| Schema inference | `pa.infer_schema(da)` / `pa.infer_schema(ds)` — infer schema from data (dims, dtype, coords, attrs) |
| IO/serialization | YAML/JSON round-trip for xarray schemas |
| Hypothesis strategies | Data synthesis for property-based testing (generate conforming DataArrays/Datasets) |
| Dask integration | Validate Dask-backed xarray objects: structural checks without `.compute()`, configurable data-level check behavior |
| Encoding validation | Validate `encoding` dicts on Variables/DataArrays/Datasets (useful for netCDF/Zarr pipeline correctness) |
| Regex/glob coord and variable matching | Pattern-based keys in `data_vars` and `coords` (e.g. `"x_*"` matches `x_0`, `x_1`) |
| `cf_xarray` integration | Optional CF convention awareness for standard name-based validation |

---

## 11. Testing Strategy

Tests live in `tests/xarray/` and are gated behind the `xarray` extra in
the nox test matrix.

| Test category | Coverage |
|---|---|
| `test_data_array_schema.py` | `DataArraySchema` — dtype, dims, sizes, shape, coords (dimension and non-dimension), attrs, name, checks, nullable, coerce, strict_coords, strict_attrs, lazy |
| `test_dataset_schema.py` | `DatasetSchema` — data_vars, dims, sizes, coords, attrs, strict, checks, lazy |
| `test_data_tree_schema.py` | `DataTreeSchema` — nested structure, path-based children, recursive validation, coordinate inheritance, strict child nodes |
| `test_data_array_model.py` | `DataArrayModel` — class-based definition, `to_schema()`, `validate()`, coordinate name access |
| `test_dataset_model.py` | `DatasetModel` — class-based definition, field access, `@check_types` |
| `test_data_tree_model.py` | `DataTreeModel` — class-based definition, nested models |
| `test_checks.py` | Built-in checks (`has_dims`, `is_monotonic`, etc.) and custom checks on xarray objects |
| `test_engine.py` | Dtype resolution, coercion, abstract dtype hierarchy (`np.floating`), datetime dtypes |
| `test_duck_arrays.py` | Validation with Dask-backed, sparse, and other duck arrays; `chunked` and `array_type` parameters |
| `test_decorators.py` | `@check_types`, `@check_input`, `@check_output` |
| `test_error_reporting.py` | Error messages, failure case formatting with N-D coordinate context, lazy vs eager, DataTree path errors |

---

## 12. Open Questions

1. **Dimension coordinates vs. auxiliary coordinates.** xarray formally
   distinguishes dimension coordinates (1-D, name matches a dim name,
   usually indexed) from non-dimension/auxiliary coordinates (can be
   multi-dimensional, name not in `dims`). Should the schema surface this
   distinction — e.g. a parameter to require certain coords be dimension
   coordinates — or is this an implementation detail left to `Check`
   objects? The current design validates coords by name and value, but
   does not enforce whether a coordinate is a dimension coordinate.

2. **MultiIndex coordinates.** xarray supports `pandas.MultiIndex` as
   coordinates, which expose "virtual" level coordinates. Should
   `DataArraySchema.coords` be able to express MultiIndex structures, or
   is that too niche for Phase 1?

3. **Indexed vs. non-indexed coordinates.** Should the schema be able to
   require that a coordinate has an associated `Index` (i.e. is usable
   for label-based selection via `.sel()`)? This matters for pipelines
   that rely on alignment and selection.

4. **Regex/glob patterns for variable and coordinate names.**
   `xarray-validate` supports glob and regex patterns in `data_vars` and
   `coords` keys (e.g. `"x_*"` to match `x_0`, `x_1`, ...). Is this a
   Phase 1 requirement or a later addition?

5. **Encoding validation.** xarray objects carry an `encoding` dict that
   controls serialization (netCDF `_FillValue`, `scale_factor`,
   `add_offset`, `dtype`, `compression`, etc.). Should pandera support
   validating encoding attributes? This is useful in data pipeline
   contexts where encoding correctness matters for downstream consumers.
   Could be a `Check.has_encoding(...)` or a schema-level `encoding`
   parameter.

6. **Integration with `cf_xarray`.** The CF conventions community uses
   `cf_xarray` for accessing data by standard names (e.g.
   `ds.cf["temperature"]`). Should pandera's xarray support be aware of
   CF conventions, or is that a separate extension?

7. **Duck array validation depth.** When the underlying array is lazy
   (e.g. Dask), should data-level checks trigger computation? The
   current design validates dtype and structural properties without
   materializing, but element-wise checks (e.g. `Check.in_range`) would
   require `.compute()`. Should this be configurable, or should lazy
   arrays skip data-level checks by default?

8. **Narwhals integration.** The narwhals effort
   ([#2081](https://github.com/unionai-oss/pandera/pull/2081)) is
   standardizing backends for dataframe libraries. Since xarray is not a
   dataframe library, the xarray backend will be independent of narwhals.
   However, the engine layer should be designed so that any future
   convergence is not blocked.

9. **`sizes` vs. `shape` precedence.** The spec offers both `sizes`
   (name→length dict) and `shape` (positional tuple) for dimension size
   validation, marked as mutually exclusive. Should we deprecate `shape`
   entirely in favor of `sizes` (which is more idiomatic in xarray), or
   keep both for users coming from NumPy?
