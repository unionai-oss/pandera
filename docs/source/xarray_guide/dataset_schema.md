---
file_format: myst
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

```python
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

```python
schema = pa.DatasetSchema(
    data_vars={
        "temperature": pa.DataVar(dtype=np.float64, dims=("x",)),
        "humidity": pa.DataVar(dtype=np.float64, dims=("x",), required=False),
    },
)

ds = xr.Dataset({"temperature": (("x",), np.ones(3))})
schema.validate(ds)  # passes — humidity is absent but not required
```

### Default values

When `required=False`, you can specify a `default` to fill in missing
variables during validation:

```python
pa.DataVar(dtype=np.float64, dims=("x",), required=False, default=0.0)
```

### Aliases

If the logical schema name differs from the actual name in the dataset:

```python
pa.DataVar(dtype=np.float64, alias="temp_kelvin")
```

### Alignment constraints

`aligned_with` and `broadcastable_with` express grid relationships to other
data variables:

```python
pa.DataVar(
    dtype=np.float64,
    dims=("x", "y"),
    aligned_with=("pressure",),      # must share exact dims/shape
)

pa.DataVar(
    dtype=np.float64,
    dims=("x",),
    broadcastable_with=("temperature",),
)
```

### Using `DataArraySchema` directly

Instead of `DataVar`, you can pass a `DataArraySchema` as the spec for a
variable. This reuses a schema you've already defined:

```python
temp_schema = pa.DataArraySchema(dtype=np.float64, dims=("x", "y"))

schema = pa.DatasetSchema(
    data_vars={
        "temperature": temp_schema,
        "pressure": pa.DataVar(dtype=np.float64, dims=("x", "y")),
    },
)
```

### `None` as a placeholder

`None` means "variable must exist, but no value-level checks":

```python
pa.DatasetSchema(data_vars={"temperature": None})
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

```python
schema = pa.DatasetSchema(
    data_vars={"a": pa.DataVar(dtype=float, dims=("x", "y"))},
    coords={
        "x": pa.Coordinate(dtype=np.float64, dimension=True),
        "label": pa.Coordinate(dtype=str, dimension=False),
    },
)
```

### Indexed coordinates

An **indexed** coordinate has an associated xarray `Index` and can be used
with `.sel()`. `indexed=True` requires this; `indexed=False` forbids it.

```python
pa.Coordinate(dtype=np.float64, dimension=True, indexed=True)
```

### Checks on coordinates

Coordinates are `DataArray` objects, so you can attach checks:

```python
pa.Coordinate(
    dtype=np.float64,
    checks=pa.Check(lambda c: float(c.min()) >= -90),
)
```

## Dimensions and sizes

Dataset-level `dims` and `sizes` constrain the overall dimension structure,
independent of individual `DataVar` specs:

```python
pa.DatasetSchema(
    data_vars={...},
    dims=("x", "y"),
    sizes={"x": 100, "y": 200},
)
```

## Attributes

`attrs` validates the Dataset's `.attrs` dict by key and value equality.

## Strict mode

- `strict=True` — fail if the dataset has data variables not listed in
  `data_vars`.
- `strict="filter"` — drop unlisted variables and return the filtered
  dataset.
- `strict=False` (default) — extra variables are allowed.

```python
schema = pa.DatasetSchema(
    data_vars={"temperature": pa.DataVar(dtype=float)},
    strict=True,
)

ds = xr.Dataset({
    "temperature": (("x",), np.ones(3)),
    "extra": (("x",), np.zeros(3)),
})

try:
    schema.validate(ds)
except pa.errors.SchemaError as exc:
    print(exc)  # "extra" is not in schema
```

### Strict coordinates and attributes

`strict_coords` and `strict_attrs` work the same way at the coordinate and
attribute level:

```python
pa.DatasetSchema(
    data_vars={...},
    coords={"x": pa.Coordinate()},
    strict_coords=True,  # fail if extra coords exist
    strict_attrs=True,   # fail if extra attrs exist
)
```

## Dataset-level checks

Checks on the `DatasetSchema` receive the entire {class}`~xarray.Dataset`:

```python
schema = pa.DatasetSchema(
    data_vars={
        "a": pa.DataVar(dtype=float, dims=("x",)),
        "b": pa.DataVar(dtype=float, dims=("x",)),
    },
    checks=pa.Check(lambda ds: (ds["a"] < ds["b"]).all()),
)
```

## Lazy validation

Pass `lazy=True` to collect all errors into a single
{class}`~pandera.errors.SchemaErrors`:

```python
try:
    schema.validate(ds, lazy=True)
except pa.errors.SchemaErrors as exc:
    print(exc.failure_cases)
```

## See also

- {ref}`xarray-data-array-schema` — single-array validation
- {ref}`xarray-data-models` — class-based `DatasetModel`
- {ref}`xarray-checks-parsers` — checks, parsers, lazy validation
- {ref}`xarray-configuration` — validation depth, Dask, environment variables
