---
file_format: mystnb
---

(xarray-dataset-schema)=

# Dataset Schemas

{class}`~pandera.api.xarray.container.DatasetSchema` validates an
{class}`~xarray.Dataset` — a dict-like container of aligned
{class}`~xarray.DataArray` objects. It is the xarray counterpart of
{class}`~pandera.api.pandas.container.DataFrameSchema`: each data variable
corresponds to a `Column`, and shared coordinates correspond to an `Index`.

You can also express the same constraints with the declarative
{ref}`DatasetModel <xarray-data-models>`.

## Basic usage

```{code-cell} python
import numpy as np
import xarray as xr
import pandera.xarray as pa

schema = pa.DatasetSchema(
    data_vars={
        "temperature": pa.DataVar(dtype=np.float64, dims=("x", "y")),
        "pressure": pa.DataVar(dtype=np.float64, dims=("x", "y")),
    },
    coords={"x": pa.Coordinate(dtype=np.float64)},
)

ds = xr.Dataset(
    {
        "temperature": (("x", "y"), np.random.rand(3, 4)),
        "pressure": (("x", "y"), np.random.rand(3, 4)),
    },
    coords={"x": np.arange(3, dtype=np.float64)},
)
schema.validate(ds)
```

## `DataVar`

{class}`~pandera.api.xarray.components.DataVar` describes one variable inside a
Dataset. It carries the same structural constraints as `DataArraySchema`
(`dtype`, `dims`, `sizes`, `shape`, `coords`, `attrs`, `checks`, `parsers`,
`coerce`, `nullable`, `chunked`, `array_type`, `strict_coords`, `strict_attrs`)
plus dataset-only options.

### Required variables

By default every `DataVar` is required. Set `required=False` to make it
optional:

```{code-cell} python
schema = pa.DatasetSchema(
    data_vars={
        "temperature": pa.DataVar(dtype=np.float64, dims=("x",)),
        "humidity": pa.DataVar(dtype=np.float64, dims=("x",), required=False),
    },
)

ds_no_humidity = xr.Dataset({"temperature": (("x",), np.ones(3))})
schema.validate(ds_no_humidity)
```

### Default values

When `required=False`, you can specify a `default` to fill in missing
variables during validation:

```{code-cell} python
schema = pa.DatasetSchema(
    data_vars={
        "temperature": pa.DataVar(dtype=np.float64, dims=("x",)),
        "humidity": pa.DataVar(
            dtype=np.float64, dims=("x",), required=False, default=0.0
        ),
    },
)

validated = schema.validate(ds_no_humidity)
validated
```

### Aliases

If the logical schema name differs from the actual name in the dataset:

```{code-cell} python
schema = pa.DatasetSchema(
    data_vars={
        "temp": pa.DataVar(dtype=np.float64, alias="temp_kelvin"),
    },
)

ds_alias = xr.Dataset({"temp_kelvin": (("x",), np.ones(3))})
schema.validate(ds_alias)
```

### Alignment constraints

`aligned_with` and `broadcastable_with` express grid relationships to other
data variables:

```{code-cell} python
schema = pa.DatasetSchema(
    data_vars={
        "temperature": pa.DataVar(dtype=np.float64, dims=("x", "y")),
        "pressure": pa.DataVar(
            dtype=np.float64,
            dims=("x", "y"),
            aligned_with=("temperature",),
        ),
        "elevation": pa.DataVar(
            dtype=np.float64,
            dims=("x",),
            broadcastable_with=("temperature",),
        ),
    },
)

ds_aligned = xr.Dataset(
    {
        "temperature": (("x", "y"), np.random.rand(3, 4)),
        "pressure": (("x", "y"), np.random.rand(3, 4)),
        "elevation": (("x",), np.ones(3)),
    },
)
schema.validate(ds_aligned)
```

### Using `DataArraySchema` directly

Instead of `DataVar`, you can pass a `DataArraySchema` as the spec for a
variable. This reuses a schema you've already defined:

```{code-cell} python
temp_schema = pa.DataArraySchema(dtype=np.float64, dims=("x", "y"))

schema = pa.DatasetSchema(
    data_vars={
        "temperature": temp_schema,
        "pressure": pa.DataVar(dtype=np.float64, dims=("x", "y")),
    },
)
schema.validate(ds)
```

### `None` as a placeholder

`None` means "variable must exist, but no value-level checks":

```{code-cell} python
schema = pa.DatasetSchema(data_vars={"temperature": None})
schema.validate(ds)
```

(xarray-coordinate)=

## `Coordinate`

{class}`~pandera.api.xarray.components.Coordinate` validates an individual
coordinate array. Use it inside the `coords` dict on `DataArraySchema` or
`DatasetSchema`.

### Dimension vs auxiliary coordinates

In xarray, a **dimension coordinate** is a 1-D coordinate whose name matches
a dimension name (used for label-based indexing). An **auxiliary coordinate**
does not match any dimension name and can be multi-dimensional.

```{code-cell} python
ds_with_aux = xr.Dataset(
    {"a": (("x", "y"), np.random.rand(3, 4))},
    coords={
        "x": np.arange(3, dtype=np.float64),
        "label": ("x", ["site_a", "site_b", "site_c"]),
    },
)

schema = pa.DatasetSchema(
    data_vars={"a": pa.DataVar(dtype=float, dims=("x", "y"))},
    coords={
        "x": pa.Coordinate(dtype=np.float64, dimension=True),
        "label": pa.Coordinate(dtype=str, dimension=False),
    },
)
schema.validate(ds_with_aux)
```

### Indexed coordinates

An **indexed** coordinate has an associated xarray `Index` and can be used
with `.sel()`. `indexed=True` requires this; `indexed=False` forbids it.

```{code-cell} python
schema = pa.DatasetSchema(
    data_vars={"a": pa.DataVar(dtype=float, dims=("x",))},
    coords={
        "x": pa.Coordinate(dtype=np.float64, dimension=True, indexed=True),
    },
)

ds_indexed = xr.Dataset(
    {"a": (("x",), np.ones(3))},
    coords={"x": np.arange(3, dtype=np.float64)},
)
schema.validate(ds_indexed)
```

### Checks on coordinates

Coordinates are `DataArray` objects, so you can attach checks:

```{code-cell} python
schema = pa.DatasetSchema(
    data_vars={"a": pa.DataVar(dtype=float, dims=("lat",))},
    coords={
        "lat": pa.Coordinate(
            dtype=np.float64,
            checks=pa.Check(lambda c: float(c.min()) >= -90),
        ),
    },
)

ds_lat = xr.Dataset(
    {"a": (("lat",), np.ones(5))},
    coords={"lat": np.linspace(-45, 45, 5)},
)
schema.validate(ds_lat)
```

## Dimensions and sizes

Dataset-level `dims` and `sizes` constrain the overall dimension structure,
independent of individual `DataVar` specs:

```{code-cell} python
schema = pa.DatasetSchema(
    data_vars={
        "temperature": pa.DataVar(dtype=float, dims=("x", "y")),
    },
    dims=("x", "y"),
    sizes={"x": 3, "y": 4},
)

ds_sized = xr.Dataset(
    {"temperature": (("x", "y"), np.random.rand(3, 4))},
)
schema.validate(ds_sized)
```

## Attributes

`attrs` validates the Dataset's `.attrs` dict. Each value in the schema's
`attrs` dict determines how the corresponding attribute is checked:

- **Literal values** — matched by equality (`==`).
- **Regex patterns** — strings that start with `^` are treated as regular
  expressions and matched against `str(actual_value)` via `re.fullmatch`.
- **Callable predicates** — any callable `(value) -> bool` is invoked with the
  actual attribute value; validation passes when the function returns `True`.

### Equality matching

```{code-cell} python
schema = pa.DatasetSchema(
    data_vars={"temperature": pa.DataVar(dtype=float)},
    attrs={"source": "reanalysis"},
)

ds_attrs = xr.Dataset(
    {"temperature": (("x",), np.ones(3))},
    attrs={"source": "reanalysis"},
)
schema.validate(ds_attrs)
```

### Regex matching

Use a regex pattern (starting with `^`) to validate an attribute against a
set of acceptable values:

```{code-cell} python
schema = pa.DatasetSchema(
    data_vars={"temperature": pa.DataVar(dtype=float)},
    attrs={"units": "^(K|degC|degF)$"},
)

ds_units = xr.Dataset(
    {"temperature": (("x",), np.ones(3))},
    attrs={"units": "K"},
)
schema.validate(ds_units)
```

```{code-cell} python
ds_bad_units = xr.Dataset(
    {"temperature": (("x",), np.ones(3))},
    attrs={"units": "meters"},
)

try:
    schema.validate(ds_bad_units)
except pa.errors.SchemaError as exc:
    print(exc)
```

### Callable predicates

Pass a function that receives the attribute value and returns a boolean:

```{code-cell} python
schema = pa.DatasetSchema(
    data_vars={"temperature": pa.DataVar(dtype=float)},
    attrs={
        "version": lambda v: isinstance(v, int) and v >= 2,
    },
)

ds_v3 = xr.Dataset(
    {"temperature": (("x",), np.ones(3))},
    attrs={"version": 3},
)
schema.validate(ds_v3)
```

```{code-cell} python
ds_v1 = xr.Dataset(
    {"temperature": (("x",), np.ones(3))},
    attrs={"version": 1},
)

try:
    schema.validate(ds_v1)
except pa.errors.SchemaError as exc:
    print(exc)
```

All three modes also work on
{class}`~pandera.api.xarray.container.DataArraySchema` — see
{ref}`xarray-data-array-schema`.

## Strict mode

- `strict=True` — fail if the dataset has data variables not listed in
  `data_vars`.
- `strict="filter"` — drop unlisted variables and return the filtered
  dataset.
- `strict=False` (default) — extra variables are allowed.

```{code-cell} python
schema = pa.DatasetSchema(
    data_vars={"temperature": pa.DataVar(dtype=float)},
    strict=True,
)

ds_extra = xr.Dataset({
    "temperature": (("x",), np.ones(3)),
    "extra": (("x",), np.zeros(3)),
})

try:
    schema.validate(ds_extra)
except pa.errors.SchemaError as exc:
    print(exc)
```

```{code-cell} python
filter_schema = pa.DatasetSchema(
    data_vars={"temperature": pa.DataVar(dtype=float)},
    strict="filter",
)

filtered = filter_schema.validate(ds_extra)
print(list(filtered.data_vars))
```

### Strict coordinates and attributes

`strict_coords` and `strict_attrs` work the same way at the coordinate and
attribute level:

```{code-cell} python
schema = pa.DatasetSchema(
    data_vars={"a": pa.DataVar(dtype=float, dims=("x",))},
    coords={"x": pa.Coordinate()},
    strict_coords=True,
)

ds_one_coord = xr.Dataset(
    {"a": (("x",), np.ones(3))},
    coords={"x": np.arange(3, dtype=np.float64)},
)
schema.validate(ds_one_coord)
```

## Dataset-level checks

Checks on the `DatasetSchema` receive the entire {class}`~xarray.Dataset`:

```{code-cell} python
schema = pa.DatasetSchema(
    data_vars={
        "a": pa.DataVar(dtype=float, dims=("x",)),
        "b": pa.DataVar(dtype=float, dims=("x",)),
    },
    checks=pa.Check(lambda ds: bool((ds["a"] < ds["b"]).all())),
)

ds_ordered = xr.Dataset({
    "a": (("x",), [1.0, 2.0, 3.0]),
    "b": (("x",), [4.0, 5.0, 6.0]),
})
schema.validate(ds_ordered)
```

## Lazy validation

Pass `lazy=True` to collect all errors into a single
{class}`~pandera.errors.SchemaErrors`:

```{code-cell} python
schema = pa.DatasetSchema(
    data_vars={
        "temperature": pa.DataVar(dtype=np.float64, dims=("x",)),
    },
    strict=True,
)

ds_bad = xr.Dataset({
    "temperature": (("y",), np.ones(3)),
    "extra_var": (("x",), np.zeros(3)),
})

try:
    schema.validate(ds_bad, lazy=True)
except pa.errors.SchemaErrors as exc:
    print(exc)
```

## See also

- {ref}`xarray-data-array-schema` — single-array {class}`~pandera.api.xarray.container.DataArraySchema` validation
- {ref}`xarray-data-models` — class-based {class}`~pandera.api.xarray.model.DatasetModel`
- {ref}`xarray-checks-parsers` — checks, parsers, lazy validation
- {ref}`xarray-decorators` — `check_input`, `check_output`, `check_io`, and `check_types`
- {ref}`xarray-configuration` — {class}`~pandera.config.ValidationDepth`,
  {class}`~pandera.config.ValidationScope`, Dask, environment variables
- {ref}`api-xarray` — full API reference for all xarray classes
- {ref}`api-core` — {class}`~pandera.config.PanderaConfig`,
  {class}`~pandera.config.ValidationDepth`,
  {class}`~pandera.config.ValidationScope`
