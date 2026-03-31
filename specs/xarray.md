# Xarray Integration Spec

> **Status:** Draft
> **Issue:** [#705](https://github.com/unionai-oss/pandera/issues/705)
> **Author:** pandera maintainers
> **Related work:**
> [xarray-schema](https://github.com/xarray-contrib/xarray-schema),
> [xarray-validate](https://github.com/leroyvn/xarray-validate)

---

## User-facing documentation

End-user guides (installation, `DataArraySchema` / `DatasetSchema`, checks, and
validation depth for Dask-backed data) live in the Sphinx source tree:

- [`docs/source/xarray_guide/index.md`](../docs/source/xarray_guide/index.md)
  — landing page and toctree
- Subpages: `data_array_schema.md`, `dataset_schema.md`,
  `xarray_models.md` (`DataArrayModel` / `DatasetModel`),
  `checks_configuration.md`

The built site lists this under **Integrations → Xarray**. This spec remains the
design and roadmap document; the guide targets library users, not implementers.

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
Instead, the public API consists of **three schema classes**, **three
model classes**, and two **component classes**:

| Schema class | Model class | Validates |
|---|---|---|
| `DataArraySchema` | `DataArrayModel` | `xr.DataArray` |
| `DatasetSchema` | `DatasetModel` | `xr.Dataset` |
| `DataTreeSchema` | `DataTreeModel` | `xr.DataTree` |

| Component class | Analogous to (pandas) | Purpose |
|---|---|---|
| `DataVar` | `Column` | One `Dataset` data variable: array-level checks + container-only flags (`alias`, `regex`, `default`, alignment with peers, optional presence) |
| `Coordinate` | `Index` | Validates a single coordinate array |

All other validation concerns (dims, shape, attrs, chunks) are expressed
as **keyword arguments** on these classes, or as pandera `Check` objects.
This mirrors how pandera's pandas API uses `DataFrameSchema`, `Column`,
and `Index` rather than exposing separate schema objects for dtype,
nullability, uniqueness, etc.

### 3.2 Consistent with Existing Pandera Patterns

The implementation must follow pandera's layered architecture:

1. **API layer** (`pandera/api/xarray/`) — schema and model definitions
2. **Backend layer** (`pandera/backends/xarray/`) — validation logic
3. **Engine layer** (`pandera/engines/xarray_engine.py`) — dtype resolution

Backend registration, lazy validation, `Check` integration, and
`@check_types` decorator support must all follow the patterns established by
the polars and pandas backends.

### 3.3 Schema Components Mirror the pandas Pattern

In the pandas world, `Column` and `Index` are the schema components that
describe parts of a `DataFrameSchema`. In the xarray world, the analogous
components are `DataVar` (for data variables) and `Coordinate` (for
coordinate arrays):

- `DataArraySchema` → standalone schema for a single `xr.DataArray`
  (primary use: `schema.validate(da)`), analogous to `SeriesSchema` for
  one column’s worth of data without the surrounding `DataFrame`.
- `DataVar` → analogous to `Column`: declares one entry in
  `DatasetSchema.data_vars`, including the same per-array constraints as
  `DataArraySchema` plus **dataset-only** options (`required`, `alias`,
  `regex`, `default`, `aligned_with`, `broadcastable_with`, etc.—see §4.4).
- `Coordinate` → analogous to `Index` (labels that accompany the data).
  Validates dimension coordinates (1-D, name matches a dim) and
  non-dimension coordinates (auxiliary, possibly multi-dimensional).
- `DatasetSchema` → analogous to `DataFrameSchema` (container of
  components)
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
    "Coordinate",
    "DataVar",
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
        ordered_dims: bool = True,
        sizes: dict[str, int | None] | None = None,
        shape: tuple[int | None, ...] | None = None,
        coords: dict[str, Coordinate] | list[str] | None = None,
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
| `dims` | `tuple[str \| None, ...]` | Expected dimension names. When `ordered_dims=True` (default), the order must match exactly. When `ordered_dims=False`, only the set of names is checked. `None` entries act as wildcards (match any dim name at that position); wildcards are only meaningful with `ordered_dims=True`. Length of the tuple also constrains `ndim` (when `ordered_dims=True`). |
| `ordered_dims` | `bool` | If `True` (default), `dims` validation is positional — the order of dimension names must match the schema. If `False`, only the set of dim names is checked, regardless of order. This is useful for datasets where dimension order is not semantically meaningful. |
| `sizes` | `dict[str, int \| None]` | Expected dimension sizes as a name→length mapping (e.g. `{"lat": 180, "lon": 360}`). More idiomatic than positional `shape` for xarray. `None` values act as wildcards. Mutually exclusive with `shape`. |
| `shape` | `tuple[int \| None, ...]` | Expected positional shape. `None` entries act as wildcards for that axis. Mutually exclusive with `sizes`. |
| `coords` | `dict[str, Coordinate] \| list[str]` | If a dict, mapping of coordinate name to a `Coordinate` that validates the coordinate's values. If a list of strings, shorthand for "these coordinate names must exist" (no value validation). |
| `attrs` | `dict[str, Any]` | Expected attributes. Each value determines how validation is performed: (1) **literal values** are matched by equality, (2) **strings starting with `^`** are treated as regex patterns and matched against `str(actual_value)` via `re.fullmatch`, and (3) **callables** `(value) -> bool` are invoked with the actual attribute value and must return `True` for the check to pass. |
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
        "time": pa.Coordinate(dtype="datetime64[ns]"),
        "lat": pa.Coordinate(
            dtype=np.float64,
            checks=pa.Check.in_range(-90, 90),
        ),
        "lon": pa.Coordinate(
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

**Attribute validation modes** — the `attrs` dict supports three matching
strategies per value:

```python
schema = pa.DataArraySchema(
    attrs={
        # 1. Equality: literal value must match exactly
        "source": "ERA5",
        # 2. Regex: string starting with "^" is a regex pattern
        "units": "^(K|°C|°F)$",
        # 3. Callable: function(value) -> bool
        "version": lambda v: isinstance(v, int) and v >= 2,
    },
)
```

### 4.3 `Coordinate`

`Coordinate` is a schema component for validating individual xarray
coordinates. It is the xarray equivalent of pandas pandera's `Index` —
just as `Index` validates the labels of a `DataFrame`, `Coordinate`
validates the label arrays of a `DataArray` or `Dataset`.

In xarray, [coordinates](https://docs.xarray.dev/en/stable/user-guide/data-structures.html#coordinates)
are ancillary variables that label the points in a data array or dataset.
They come in two flavors:

- **Dimension coordinates**: 1-D coordinates whose name matches a
  dimension name (e.g. a `time` coordinate on the `time` dim). These
  are used for label-based indexing and alignment, analogous to a pandas
  `Index`.
- **Non-dimension coordinates** ("auxiliary coordinates"): coordinates
  whose name does not match any dimension. They can be multi-dimensional
  (e.g. 2-D `lat`/`lon` on a projected grid) and are not used for
  alignment.

`Coordinate` validates both kinds. It shares the core validation
semantics of `DataArraySchema` (since coordinates *are* `DataArray`
objects in xarray) but is a distinct component type to make the schema
structure explicit and to parallel the `Column` / `Index` distinction in
the pandas API.

```python
class Coordinate:
    def __init__(
        self,
        dtype: DTypeLike | None = None,
        dims: tuple[str, ...] | None = None,
        dimension: bool | None = None,
        required: bool = True,
        checks: Check | list[Check] | None = None,
        parsers: Parser | list[Parser] | None = None,
        nullable: bool = False,
        coerce: bool = False,
        indexed: bool | None = None,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        metadata: dict | None = None,
    ):
        ...
```

**Parameter semantics:**

| Parameter | Type | Description |
|---|---|---|
| `dtype` | numpy dtype, type, or string | Expected dtype of the coordinate values. |
| `dims` | `tuple[str, ...]` | Expected dimensions of the coordinate array. For dimension coordinates this is always `(name,)` and can be omitted. Required for non-dimension (auxiliary) coordinates that span multiple dims (e.g. 2-D `lat`/`lon` on a projected grid). |
| `dimension` | `bool \| None` | If `True`, require the coordinate to be a dimension coordinate (1-D, name in parent object's `dims`, and `dims=(name,)`). If `False`, require an auxiliary/non-dimension coordinate (name not in parent object's `dims`). If `None` (default), don't enforce coordinate kind. |
| `required` | `bool` | If `True` (default), the coordinate must exist on the data object. If `False`, the coordinate is optional; when present all other constraints apply, when absent no error is raised. Note: when `strict_coords=True` on the parent schema, all coordinates on the data object must appear in the schema, but coordinates with `required=False` need not appear on the data object. |
| `checks` | `Check` or list | Data-level checks on coordinate values (e.g. `Check.in_range(-90, 90)` for latitude). |
| `parsers` | `Parser` or list | Parsers/transformations applied before validation. |
| `nullable` | `bool` | If `False` (default), NaN/null coordinate values raise a validation error. |
| `coerce` | `bool` | If `True`, coerce dtype before validation. |
| `indexed` | `bool \| None` | If `True`, require the coordinate to have an associated xarray `Index` (i.e. usable for `.sel()`). If `False`, require it to be non-indexed. If `None` (default), don't check. |
| `name` | `str` | Expected coordinate name (inferred from the key in the `coords` dict when used inside a schema). |

**Example:**

```python
lat_coord = pa.Coordinate(
    dtype=np.float64,
    checks=pa.Check.in_range(-90, 90),
)
time_coord = pa.Coordinate(dtype="datetime64[ns]")
```

### 4.4 `DataVar`

`DataVar` is the schema component for one **data variable** inside a
`Dataset`, analogous to pandas pandera’s `Column` inside a
`DataFrameSchema`. It takes the same structural and data-level parameters
as `DataArraySchema` (dtype, dims, sizes/shape, per-variable coords,
attrs, checks, etc.) and adds **dataset-context** options that only make
sense when the variable is declared inside `DatasetSchema.data_vars`.

`DataArraySchema` stays focused on **standalone** `xr.DataArray`
validation (`schema.validate(da)`). `DatasetSchema.data_vars` accepts
`DataVar` (or `None` for “must exist, no value-level validation”), not a
bare `DataArraySchema`, so container semantics (`required`, alias,
defaults, cross-variable alignment, key patterns) are not overloaded onto
the standalone schema.

```python
class DataVar:
    def __init__(
        self,
        *,
        required: bool = True,
        alias: str | None = None,
        regex: bool = False,
        default: Any | xr.DataArray | None = None,
        aligned_with: tuple[str, ...] | None = None,
        broadcastable_with: tuple[str, ...] | None = None,
        dtype: DTypeLike | None = None,
        dims: tuple[str | None, ...] | None = None,
        ordered_dims: bool = True,
        sizes: dict[str, int | None] | None = None,
        shape: tuple[int | None, ...] | None = None,
        coords: dict[str, Coordinate] | list[str] | None = None,
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

    def to_data_array_schema(self) -> DataArraySchema:
        """Materialize array-level validation (no dataset-only fields)."""
        ...
```

**Shared with `DataArraySchema` (§4.2):** All parameters from `dtype`
through `metadata` in the signature above behave like `DataArraySchema`.
See §4.2 for `strict_coords` / `strict_attrs` on the **slice**
`ds[...]`; that is independent of `DatasetSchema.strict_coords`, which
concerns **dataset-level** coordinate keys (see below and §4.5).

**`DataVar`-only / dataset-context parameters:**

| Parameter | Type | Description |
|---|---|---|
| `required` | `bool` | If `True` (default), the variable must exist (under the resolved name—see `alias`). If `False`, it may be omitted; when present, array-level constraints apply. |
| `alias` | `str \| None` | Actual `Dataset` data variable **name** on disk (e.g. `"temp"`) when it differs from the **`data_vars` dict key** (the logical/schema name, e.g. `"temperature"`). Default `None` means the dict key is the dataset name. Mirrors `Column.alias` / `Field(alias=...)` in pandas pandera. |
| `regex` | `bool` | If `True`, the dict key is treated as a **regex** (or glob—implementation choice documented at ship time) matching one or more variables in the dataset, and this `DataVar` spec is **reused** for each match—pandas `Column(regex=True)` pattern. Meaningless without a `Dataset` key space. |
| `default` | `Any \| xr.DataArray \| None` | When `required=False` and the variable is **missing**, validation may **insert** a default: broadcast scalar, fill value, or template `DataArray` aligned to dataset coords/dims. `None` means “do not synthesize.” Applies only in `DatasetSchema.validate()`; standalone `DataArraySchema` never runs on a missing object. |
| `aligned_with` | `tuple[str, ...] \| None` | Other **logical** `data_vars` keys (after resolving `alias` to know which arrays exist) whose dimension names and sizes must **match** this variable’s grid. Purely multi-variable / `Dataset` semantics. |
| `broadcastable_with` | `tuple[str, ...] \| None` | Weaker than `aligned_with`: referenced variables must be **broadcast-compatible** with this one under xarray’s rules (e.g. `(time,)` vs `(time, 1)`). If both `aligned_with` and `broadcastable_with` are set, implementations should validate both; overlapping names should be an init error. |

**Implementation note:** Resolve each slot’s **dataset variable name**
(`alias or dict_key`), expand `regex` entries to concrete names (if
any), then for each present variable build `DataArraySchema` via
`to_data_array_schema()` and delegate. Apply `default` only after
**structural** checks if inserting missing optional variables is enabled
(policy TBD: strict vs “repair” mode). Enforce `aligned_with` /
`broadcastable_with` **after** per-variable validation so referenced
arrays exist or are explicitly optional.

### 4.5 `DatasetSchema`

`DatasetSchema` validates an `xr.Dataset`. It contains a dict of
`DataVar` specs — one per declared data variable — plus optional
dataset-level coordinate, attribute, and check specifications.

A `Dataset`'s dimensions are the union of the dimensions across all its
data variables. The `sizes` parameter on `DatasetSchema` validates
dataset-level dimension sizes, while per-variable structure is validated
via each `DataVar` (which delegates to `DataArraySchema` semantics for the
corresponding `ds[var_name]`).

```python
class DatasetSchema(BaseSchema):
    def __init__(
        self,
        data_vars: dict[str, DataVar | None] | None = None,
        coords: dict[str, Coordinate] | list[str] | None = None,
        dims: tuple[str, ...] | None = None,
        ordered_dims: bool = True,
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
| `data_vars` | `dict[str, DataVar \| None]` | Logical name → spec. Lookup uses `DataVar.alias` when set (see §4.4). Keys paired with `DataVar(..., regex=True)` are patterns matching one or more dataset variables. `DataVar(...)` applies constraints when the variable is present; `required=True` (default) requires existence under the resolved name; `required=False` allows omission and optional `default` fill. `None` means required presence, no value-level validation. |
| `coords` | `dict[str, Coordinate] \| list[str]` | Dataset-level coordinate schemas. If a dict, validates coordinate values. If a list, only checks coordinate existence. |
| `dims` | `tuple[str, ...]` | Expected dataset-level dimension names (the union of all variable dims). When `ordered_dims=False`, only the set of names is checked. |
| `ordered_dims` | `bool` | If `True` (default), dataset-level `dims` validation checks names in order. If `False`, only the set of names is compared. For `DatasetSchema` this is less commonly needed than for `DataArraySchema`, since dataset-level dims are the union of all variable dims and order is often uncontrolled. |
| `sizes` | `dict[str, int \| None]` | Expected dataset-level dimension sizes as a name→length mapping. |
| `attrs` | `dict[str, Any]` | Expected dataset-level attributes. Each value determines how validation is performed: (1) **literal values** are matched by equality, (2) **strings starting with `^`** are treated as regex patterns and matched against `str(actual_value)` via `re.fullmatch`, and (3) **callables** `(value) -> bool` are invoked with the actual attribute value and must return `True` for the check to pass. |
| `checks` | `Check` or list | Dataset-wide checks (receive the full `xr.Dataset`). |
| `strict` | `bool \| "filter"` | If `True`, fail on unexpected data variables. If `"filter"`, drop them. |
| `strict_coords` | `bool` | If `True`, fail on unexpected **dataset-level** coordinate keys. Distinct from per-variable `DataVar.strict_coords` on each `ds[var]` slice (§4.4 / §4.2). |
| `strict_attrs` | `bool` | If `True`, fail on unexpected attributes. |

**Cross-variable and disjunctive rules:** Constraints that reference
**several** data variables at once (“exactly one of”, “at least one of”,
mutually exclusive sets) do **not** belong on a single `DataVar`. Express
them as **`DatasetSchema` checks** (`checks=` receiving the full
`xr.Dataset`) or as small **helper checks** / a future
`data_var_groups`-style API (Phase 2+ design). Keep `DataVar` focused on
one slot (plus `aligned_with` / `broadcastable_with` ties to named peers).

**Example — imperative API:**

```python
schema = pa.DatasetSchema(
    data_vars={
        "temperature": pa.DataVar(
            dtype=np.float64,
            dims=("time", "lat", "lon"),
            checks=pa.Check.in_range(150, 350),
        ),
        "precipitation": pa.DataVar(
            dtype=np.float64,
            dims=("time", "lat", "lon"),
            checks=pa.Check.ge(0),
        ),
        # optional variable: validate if present, allow if missing
        "uncertainty": pa.DataVar(
            dtype=np.float64,
            dims=("time", "lat", "lon"),
            required=False,
        ),
    },
    coords={
        "time": pa.Coordinate(dtype="datetime64[ns]"),
        "lat": pa.Coordinate(
            dtype=np.float64,
            checks=pa.Check.in_range(-90, 90),
        ),
        "lon": pa.Coordinate(
            dtype=np.float64,
            checks=pa.Check.in_range(-180, 180),
        ),
    },
)

ds = xr.Dataset({...})
validated = schema.validate(ds)
```

### 4.6 `DataTreeSchema`

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
| `attrs` | `dict[str, Any]` | Expected node-level attributes. Values support equality, regex patterns (`^...`), and callables `(value) -> bool` (same semantics as `DataArraySchema.attrs`). |
| `strict` | `bool` | If `True`, fail on unexpected child nodes. |

**Example — path-based tree schema:**

```python
schema = pa.DataTreeSchema(
    dataset=pa.DatasetSchema(attrs={"conventions": "CF-1.8"}),
    children={
        "surface": pa.DatasetSchema(
            data_vars={
                "temperature": pa.DataVar(
                    dtype=np.float64,
                    dims=("time", "lat", "lon"),
                ),
            },
        ),
        "surface/diagnostics": pa.DatasetSchema(
            data_vars={"rmse": pa.DataVar(dtype=np.float64)},
        ),
    },
    strict=True,
)
```

### 4.7 Class-Based Models (Declarative API)

Following pandera's `DataFrameModel` pattern, class-based models use type
annotations and `Field()` descriptors to declare schemas.

#### `DataArrayModel`

A `DataArray` is a single N-dimensional array — unlike a `Dataset`
(which contains multiple data variables), it has no sub-components to
enumerate except its **coordinates**. Following the pandera convention
(where class attributes represent sub-components and `Config` holds
schema-level properties), `DataArrayModel` uses:

- **Class attributes** with `Coordinate[dtype]` → coordinate schemas
  (analogous to `Index[dtype]` in pandas `DataFrameModel`), and a special `data`
  attribute for the data array itself.
- **`Config`** → properties of the data array itself (`dtype`, `dims`,
  `name`, `sizes`/`shape`, `coerce`, `nullable`, etc.)

```python
import pandera.xarray as pa
from pandera.typing.xarray import Coordinate

class Temperature(pa.DataArrayModel):
    data: np.float64 = pa.Field(
        in_range={"min_value": 150, "max_value": 350},
        dims=("time", "lat", "lon"),
    )
    time: Coordinate["datetime64[ns]"]
    lat: Coordinate[np.float64] = pa.Field(
        in_range={"min_value": -90, "max_value": 90},
    )
    lon: Coordinate[np.float64] = pa.Field(
        in_range={"min_value": -180, "max_value": 180},
    )

    class Config:
        dtype = np.float64
        dims = ("time", "lat", "lon")
        name = "temperature"

da = Temperature.validate(da)
```

This maps 1:1 to the imperative API:

| Imperative (`DataArraySchema`) | Declarative (`DataArrayModel`) |
|---|---|
| `dtype=np.float64` | `Config.dtype = np.float64` |
| `dims=("time", "lat", "lon")` | `Config.dims = (...)` |
| `ordered_dims=False` | `Config.ordered_dims = False` |
| `name="temperature"` | `Config.name = "temperature"` |
| `data: np.float64 = pa.Field(in_range={"min_value": 150, "max_value": 350}, dims=("time", "lat", "lon"))` | `data: np.float64 = pa.Field(...)` |
| `coords={"lat": pa.Coordinate(...)}` | `lat: Coordinate[np.float64] = pa.Field(...)` |

`Coordinate`-annotated fields also allow access to coordinate names as
class attributes (addressing the use case raised by
[@avcopan](https://github.com/unionai-oss/pandera/issues/705#issuecomment-2562316655)):

```python
class RateConstant(pa.DataArrayModel):
    data: np.float64 = pa.Field(
        in_range={"min_value": 150, "max_value": 350},
        dims=("x", "y"),
    )
    x: Coordinate[np.float64]
    y: Coordinate[np.float64]

    class Config:
        dtype = np.float64

RateConstant.validate(da)

# Access coordinate names as class attributes
print(RateConstant.x)   # "x"
print(RateConstant.y)  # "y"
```

#### `DatasetModel`

A `Dataset` contains multiple data variables and coordinates — both are
sub-components, so both become class attributes. Plain type annotations
map to data variables (`DataVar` under the hood), while
`Coordinate[dtype]` annotations map to coordinate schemas. This directly
parallels `Series` vs `Index` in a pandas `DataFrameModel`.

```python
class Temperature(pa.DataArrayModel):
    data: np.float64 = pa.Field(
        in_range={"min_value": 150, "max_value": 350},
        dims=("time", "lat", "lon"),
    )
    time: Coordinate["datetime64[ns]"]
    lat: Coordinate[np.float64] = pa.Field(
        in_range={"min_value": -90, "max_value": 90},
    )
    lon: Coordinate[np.float64] = pa.Field(
        in_range={"min_value": -180, "max_value": 180},
    )

class Precipitation(pa.DataArrayModel):
    data: np.float64 = pa.Field(
        dims=("time", "lat", "lon"),
        ge=0,
    )
    time: Coordinate["datetime64[ns]"]
    lat: Coordinate[np.float64] = pa.Field(
        in_range={"min_value": -90, "max_value": 90},
    )
    lon: Coordinate[np.float64] = pa.Field(
        in_range={"min_value": -180, "max_value": 180},
    )

class ClimateDataWithDataArrayModels(pa.DatasetModel):
    temperature: Temperature
    precipitation: Precipitation


class ClimateDataWithFields(pa.DatasetModel):
    # alternatively, a DatasetModel can be defined using Fields and coordinate
    # types
    temperature: np.float64 = pa.Field(
        dims=("time", "lat", "lon"),
        in_range={"min_value": 150, "max_value": 350},
    )
    precipitation: np.float64 = pa.Field(
        dims=("time", "lat", "lon"),
        ge=0,
    )
    humidity: np.float64 | None = pa.Field(ge=0)  # Optional variable

    # Coordinates (like Index in DataFrameModel)
    time: Coordinate["datetime64[ns]"]
    lat: Coordinate[np.float64] = pa.Field(
        in_range={"min_value": -90, "max_value": 90},
    )
    lon: Coordinate[np.float64] = pa.Field(
        in_range={"min_value": -180, "max_value": 180},
    )

ClimateData.validate(ds)

# Access field names as class attributes
print(ClimateData.temperature)   # "temperature"
print(ClimateData.precipitation) # "precipitation"
```

This maps 1:1 to the imperative API:

| Imperative (`DatasetSchema`) | Declarative (`DatasetModel`) |
|---|---|
| `data_vars={"temperature": pa.DataVar(dtype=np.float64, ...)}` | `temperature: np.float64 = pa.Field(...)` |
| `data_vars={"humidity": pa.DataVar(..., required=False)}` | `humidity: np.float64 = pa.Field(..., required=False)` or `humidity: np.float64 \| None = pa.Field(...)` |
| `DataVar(alias="temp", ...)` (key `temperature`) | `temperature: ... = pa.Field(..., alias="temp")` |
| Pattern entry + `DataVar(..., regex=True)` | `pa.Field(..., regex=True)` on model field (Phase 2 model support) |
| `DataVar(..., default=0.0, required=False)` | `pa.Field(..., default=0.0, required=False)` |
| `DataVar(..., aligned_with=("precipitation",))` | `pa.Field(..., aligned_with=("precipitation",))` |
| `DatasetSchema(checks=...)` for XOR / exactly-one | `@pa.check` / `Config` dataset checks (see §4.5) |
| `coords={"time": pa.Coordinate(dtype="datetime64[ns]")}` | `time: Coordinate["datetime64[ns]"]` |
| `strict=True` | `Config.strict = True` |
| `checks=pa.Check(...)` | `@pa.dataframe_check` method or Config attribute |

#### `DataTreeModel`

```python
class ClimateTree(pa.DataTreeModel):
    surface: ClimateData  # reference to a DatasetModel
    upper_atmosphere: UpperAtmData

    class Config:
        strict = True

ClimateTree.validate(dt)
```

### 4.8 `Field` Descriptor

`Field()` provides per-variable configuration in model classes, similar to
its role in `DataFrameModel`. On `DatasetModel`, non-coordinate fields
compile to `DataVar` (including `required`, `alias`, `regex`, `default`,
`aligned_with`, and `broadcastable_with` where applicable).

```python
def Field(
    *,
    # built-in check kwargs (dispatched to Check methods)
    eq: Any | None = None,
    ne: Any | None = None,
    gt: Any | None = None,
    ge: Any | None = None,
    lt: Any | None = None,
    le: Any | None = None,
    in_range: dict[str, Any] | tuple | None = None,
    isin: Iterable[Any] | None = None,
    notin: Iterable[Any] | None = None,
    # xarray-specific per-field config
    dims: tuple[str, ...] | None = None,
    ordered_dims: bool = True,
    sizes: dict[str, int | None] | None = None,
    shape: tuple[int | None, ...] | None = None,
    aligned_with: tuple[str, ...] | None = None,
    broadcastable_with: tuple[str, ...] | None = None,
    # common field config
    required: bool = True,
    nullable: bool = False,
    coerce: bool = False,
    regex: bool = False,
    default: Any | None = None,
    alias: str | None = None,
    title: str | None = None,
    description: str | None = None,
    metadata: dict | None = None,
    **kwargs: Any,  # registered custom checks
) -> Any:
    ...
```

On `DatasetModel` **data-variable** fields, **`default`** may be a scalar,
array-like, or `xr.DataArray` template, consistent with `DataVar.default`
(§4.4).

### 4.9 Decorator Support

The existing `@check_types`, `@check_input`, `@check_output`, and
`@check_io` decorators should work with xarray type annotations:

```python
from pandera.typing.xarray import DataArray, Dataset, Coordinate

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
│       ├── components.py       # Coordinate, DataVar
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
│   └── xarray.py               # DataArray[T], Dataset[T], Coordinate[T] generic types
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

#### `pandera/api/xarray/components.py`

`Coordinate` and `DataVar` are the schema components for validating
xarray coordinates and dataset data variables, analogous to `Index` and
`Column` for pandas:

```python
class Coordinate:
    """Schema component for validating xarray coordinates."""
    ...

class DataVar:
    """Schema component for one `Dataset` data variable.

    Carries array-level options (mirroring `DataArraySchema`) plus
    dataset-only flags: `required`, `alias`, `regex`, `default`,
    `aligned_with`, `broadcastable_with`, etc.
    """
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
3. Check dims (names and, when `ordered_dims=True`, order)
4. Check sizes (named dimension lengths) OR shape (positional lengths),
   with wildcard support for `None` values
5. Check coords — for each entry in the `coords` schema:
   - If `Coordinate.required` is `True` (default), verify the coordinate
     exists on the DataArray; if `required=False` and the coordinate is
     absent, skip it
   - If the coord spec is a `Coordinate`, validate `da.coords[name]`
     (value/shape/dtype checks mirror the `DataArray`-shaped validation path)
   - Distinguish dimension coordinates (name in `da.dims`) from
     non-dimension/auxiliary coordinates (name not in `da.dims`) for
     error reporting
   - If `Coordinate.dimension` is set, enforce dimension-vs-auxiliary kind
6. Check strict_coords (reject unexpected coordinates; coordinates with
   `required=False` are still allowed entries in the schema for
   `strict_coords` purposes)
7. Check attrs (equality, regex, or callable matching) and strict_attrs
8. Check chunked status (`da.chunks is not None`)
9. Check array_type (`isinstance(da.data, array_type)`)
10. Check nullable (scan for NaN/null via `da.isnull().any()`)
11. Run `Check` objects via `run_checks()`
12. Collect errors in `ErrorHandler` (lazy mode) or raise immediately

**`DatasetSchemaBackend.validate()`:**

1. Check dataset-level dims (the union of all variable dims)
2. Check dataset-level sizes
3. Expand `data_vars` keys with `DataVar.regex=True` to concrete variable
   names; build a logical-name → spec map (handle collisions per init
   rules)
4. Check data variable presence using resolved names (`alias` or dict
   key); enforce `DatasetSchema.strict` on unexpected data vars
5. For missing `required=False` slots with a non-`None` `default`, insert
   variables per documented policy (broadcast / align to coords)
6. For each present variable, materialize `DataArraySchema` from `DataVar`
   and delegate to `DataArraySchemaBackend.validate()` on `ds[resolved]`
7. Validate `aligned_with` / `broadcastable_with` between the resolved
   arrays (after optional variables are handled)
8. Check dataset-level coords (shared across variables) and
   `DatasetSchema.strict_coords`
9. Check dataset-level attrs and `strict_attrs`
10. Run dataset-level checks (including disjunctive / multi-var logic)
11. Collect errors or raise

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

class Coordinate(Generic[T]):
    """Annotation type for xarray coordinates in model classes.

    Used like Index in pandas DataFrameModel: distinguishes coordinate
    fields from data variable fields in DataArrayModel and DatasetModel.

    Example::

        class MyModel(pa.DataArrayModel):
            lat: Coordinate[np.float64] = pa.Field(ge=-90, le=90)
    """
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

When `PANDERA_VALIDATION_DEPTH` is unset, chunked (Dask-backed) DataArrays and
Datasets default to `SCHEMA_ONLY` for data-level checks (Polars LazyFrame
behavior). Set `SCHEMA_AND_DATA` or `DATA_ONLY` to run checks that would
compute lazy arrays. Eager arrays default to `SCHEMA_AND_DATA`.

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

**Goal:** Ship `DataArraySchema`, `DataVar`, `DatasetSchema`, and their
backends.

| Task | Details |
|---|---|
| API layer | `pandera/api/xarray/` — `DataArraySchema`, `DataVar`, `DatasetSchema`, types, utils |
| Backend layer | `pandera/backends/xarray/` — `DataArraySchemaBackend`, `DatasetSchemaBackend`, `XarrayCheckBackend`, built-in checks, `register.py` |
| Engine layer | `pandera/engines/xarray_engine.py` — NumPy-based dtype registry |
| Entry point | `pandera/xarray.py` |
| Type stubs | `pandera/typing/xarray.py` |
| Coordinate kind checks | `Coordinate(dimension=True/False)` to distinguish dimension vs auxiliary coordinates |
| `DataVar` container semantics | `required=False`, `alias`, `default`, `aligned_with`, `broadcastable_with` (Core); `regex` paired with Phase 2 pattern keys (§10 Phase 2) |
| Lazy data check policy | One config switch for running data-level checks on lazy arrays, defaulting to disabled |
| Tests | `tests/xarray/` — structural validation, data checks, lazy mode, coercion, error reporting |
| Dependencies | `pyproject.toml` — add `xarray` extra |

### Phase 2: Class-Based Models + Pattern Matching

**Goal:** Ship `DataArrayModel`, `DatasetModel`, and pattern-based key
matching for coordinates/data variables.

| Task | Details |
|---|---|
| Model classes | `pandera/api/xarray/model.py` — `DataArrayModel`, `DatasetModel` |
| `Field` descriptor | `pandera/api/xarray/model_components.py` — xarray-specific `FieldInfo` |
| Decorator support | `@check_types` with `DataArray[T]` and `Dataset[T]` |
| Pattern matching | `DataVar(..., regex=True)` and pattern keys in `data_vars` / `coords` for variable families (e.g. `"^band_[0-9]+$"`); document regex vs glob choice |
| Tests | Model-based validation, `@check_types` integration |

### Phase 3: DataTree Support

**Goal:** Ship `DataTreeSchema` and `DataTreeModel`.

| Task | Details |
|---|---|
| API | `DataTreeSchema`, `DataTreeModel` |
| Backend | `DataTreeSchemaBackend` |
| MultiIndex coordinates | Add MultiIndex/virtual-level coordinate support for advanced xarray indexing use cases |
| Tests | Tree-structured validation, nested schemas |

### Phase 4: Advanced Features

| Feature | Details |
|---|---|
| Schema inference | `pa.infer_schema(da)` / `pa.infer_schema(ds)` — infer schema from data (dims, dtype, coords, attrs) |
| IO/serialization | YAML/JSON round-trip for xarray schemas |
| Hypothesis strategies | Data synthesis for property-based testing (generate conforming DataArrays/Datasets) |
| Dask integration | Validate Dask-backed xarray objects: structural checks without `.compute()`, configurable data-level check behavior |
| Encoding validation | Validate `encoding` dicts on Variables/DataArrays/Datasets (useful for netCDF/Zarr pipeline correctness) |
| `cf_xarray` integration | Optional CF convention awareness for standard name-based validation |

---

## 11. Testing Strategy

Tests live in `tests/xarray/` and are gated behind the `xarray` extra in
the nox test matrix.

| Test category | Coverage |
|---|---|
| `test_data_array_schema.py` | `DataArraySchema` — dtype, dims, `ordered_dims`, sizes, shape, coords (dimension and non-dimension), attrs (equality, regex, callable), name, checks, nullable, coerce, strict_coords, strict_attrs, lazy |
| `test_dataset_schema.py` | `DatasetSchema` — `DataVar` (`required`, `alias`, `regex`, `default`, `aligned_with`, `broadcastable_with`, per-var `strict_coords` vs dataset `strict_coords`), data_vars, dims, `ordered_dims`, sizes, coords, attrs (equality, regex, callable), strict, checks, lazy, disjunctive rules via dataset checks |
| `test_coordinate.py` | `Coordinate` — `required=True/False`, interaction with `strict_coords`, dtype, dimension, indexed, checks |
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

## 12. Open Questions and Decisions

1. **Dimension coordinates vs. auxiliary coordinates (resolved).**
   Community feedback indicates this distinction is important in practice.
   The API will expose `Coordinate(dimension=True/False)` in Phase 1,
   with `None` as the default for backward-compatible permissiveness.

2. **MultiIndex coordinates (deferred).** This is important for a smaller
   but significant subset of users. It is explicitly deferred to Phase 3
   to keep Phase 1 focused.

3. **Indexed vs. non-indexed coordinates.** The `Coordinate` component
   now includes an `indexed` parameter for this. Remaining question: is
   the boolean sufficient, or should `Coordinate` accept a specific
   `Index` class (e.g. `PandasIndex` vs a custom xarray `Index`
   subclass)?

4. **Regex/glob patterns for variable and coordinate names (prioritized).**
   Feedback identifies this as high-value and often required for real
   datasets with variable families. Phase 2 will pair pattern keys with
   `DataVar(..., regex=True)` (and coordinate-side analogue); exact
   regex-vs-glob behavior is finalized at implementation time.

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

7. **Duck array validation depth (resolved).** Provide a single config
   option controlling whether data-level checks on lazy arrays trigger
   computation. Default is **disabled** for lazy arrays to avoid surprise
   compute costs; schema-only checks still run.

8. **Narwhals integration.** The narwhals effort
   ([#2081](https://github.com/unionai-oss/pandera/pull/2081)) is
   standardizing backends for dataframe libraries. Since xarray is not a
   dataframe library, the xarray backend will be independent of narwhals.
   However, the engine layer should be designed so that any future
   convergence is not blocked.

9. **`sizes` vs. `shape` precedence (resolved).** Mirror xarray semantics:
   `DataArraySchema` keeps both `sizes` and `shape` (mutually exclusive),
   while `DatasetSchema` exposes `sizes` only.

10. **Disjunctive data-variable groups.** “Exactly one of / at least one of /
    XOR” involves multiple keys and is modeled via `DatasetSchema.checks`
    or a future `data_var_groups` helper—not duplicated on each `DataVar`.
    Final helper API shape TBD in Phase 2+.

11. **Dimension ordering (resolved).** Community feedback requests the ability
    to make dimension ordering checks optional. The `ordered_dims` parameter
    (default `True` for backward compatibility) is added to
    `DataArraySchema`, `DataVar`, `DatasetSchema`, and `Field`. When
    `ordered_dims=False`, dims are validated as a set (names only, no
    positional check), and wildcards (`None`) in the `dims` tuple are
    ignored.

12. **Attribute validation beyond equality (resolved).** The `attrs` dict
    now supports three matching modes per value: (1) literal equality, (2)
    regex pattern matching for string values starting with `^`, and (3)
    callable predicates `(value) -> bool`. This covers common use cases
    like validating that a `units` attribute matches a set of accepted
    strings or checking semantic version constraints.

13. **Optional coordinates (resolved).** `Coordinate` now accepts
    `required=True/False` (default `True`), paralleling `DataVar.required`.
    When `required=False` the coordinate may be absent without error; when
    present, all constraints apply. This is independent of `strict_coords`,
    which governs *extra* coordinates on the data object.
