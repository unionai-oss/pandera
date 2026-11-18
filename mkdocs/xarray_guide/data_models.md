# Data Models

[`DataArrayModel`](../reference/generated/pandera.api.xarray.model.DataArrayModel.html) and
[`DatasetModel`](../reference/generated/pandera.api.xarray.model.DatasetModel.html) provide a class-based,
pydantic-style API for defining xarray schemas — the same pattern as
[DataFrameModel](../dataframe_models.html) for pandas.

Type annotations and [`Field`](../reference/generated/pandera.api.xarray.model_components.Field.html)
descriptors define the schema; call `validate()` or `to_schema()` to use it.

The imperative counterparts are [DataArray Schemas](./data_array_schema.html) and
[DatasetSchema](./dataset_schema.html).

## Annotation `Coordinate` vs component `Coordinate`

Two objects share the name "Coordinate". To avoid confusion, the convention in
this guide is to prefix with the module path when context is ambiguous:

- **[`Coordinate`](../reference/generated/pandera.typing.xarray.Coordinate.html)**
  (`pandera.typing.xarray.Coordinate`) — a *typing marker* used in model
  annotations, analogous to `Index` in a `DataFrameModel`.
- **[`Coordinate`](../reference/generated/pandera.api.xarray.components.Coordinate.html)**
  (`pandera.api.xarray.components.Coordinate`, also available as
  `pa.Coordinate`) — the schema *component* you pass to
  `DataArraySchema(coords=...)` or `DatasetSchema(coords=...)`.

```python
from pandera.typing.xarray import Coordinate  # annotation marker
import pandera.xarray as pa

pa.Coordinate(dtype=float)  # imperative component
```

## `DataArrayModel`

### Basic usage

Every `DataArrayModel` must define a **`data`** field whose type annotation is
the array dtype. Other fields use `Coordinate[dtype]` to declare coordinate
schemas. Schema-level options live on a nested `Config` class.

```python
import numpy as np
import xarray as xr

class Temperature(pa.DataArrayModel):
    data: np.float64 = pa.Field()
    time: Coordinate[np.float64]
    lat: Coordinate[np.float64]
    lon: Coordinate[np.float64]

    class Config:
        dims = ("time", "lat", "lon")
        name = "temperature"

da = xr.DataArray(
    np.random.rand(12, 180, 360),
    dims=("time", "lat", "lon"),
    coords={
        "time": np.arange(12, dtype=np.float64),
        "lat": np.linspace(-89.5, 89.5, 180),
        "lon": np.linspace(-179.5, 179.5, 360),
    },
    name="temperature",
)
Temperature.validate(da)
```

### Field names as attributes

Accessing a class attribute on the model returns the coordinate or variable
name as a string, useful for programmatic indexing:

```python
print(Temperature.time)
print(Temperature.lat)
```

### `Config` options

`DataArrayModel.Config`
([`DataArrayConfig`](../reference/generated/pandera.api.xarray.model_config.DataArrayConfig.html)) accepts:
`dtype`, `dims`, `sizes`, `shape`, `name`, `coerce`, `nullable`,
`strict_coords`, `strict_attrs`, `attrs`, `chunked`, `array_type`.

These mirror the keyword arguments on
[`DataArraySchema`](../reference/generated/pandera.api.xarray.container.DataArraySchema.html).

### Using `Field` on `data`

The `data` field can carry the same per-field structural constraints that
you would pass as `DataArraySchema` constructor arguments:

```python
class Grid(pa.DataArrayModel):
    data: np.float64 = pa.Field(
        dims=("x", "y"),
        sizes={"x": 3, "y": 4},
    )
    x: Coordinate[np.float64]
    y: Coordinate[np.float64]

    class Config:
        name = "grid"

da_grid = xr.DataArray(
    np.random.rand(3, 4),
    dims=("x", "y"),
    coords={
        "x": np.arange(3, dtype=np.float64),
        "y": np.arange(4, dtype=np.float64),
    },
    name="grid",
)
Grid.validate(da_grid)
```

When both `Field(dims=...)` and `Config.dims` are set, the `Field` value
takes precedence.

### Using `Field` on coordinates

Coordinate fields accept the same built-in check keywords as
[`Field`](../reference/generated/pandera.api.dataframe.model_components.Field.html): `eq`, `ge`, `le`,
`in_range`, `isin`, etc. Plus `nullable` and `coerce`.

```python
class Geo(pa.DataArrayModel):
    data: np.float64 = pa.Field()
    lat: Coordinate[np.float64] = pa.Field(ge=-90, le=90)
    lon: Coordinate[np.float64] = pa.Field(ge=-180, le=180)

    class Config:
        dims = ("lat", "lon")

da_geo = xr.DataArray(
    np.ones((5, 10)),
    dims=("lat", "lon"),
    coords={
        "lat": np.linspace(-45, 45, 5),
        "lon": np.linspace(-90, 90, 10),
    },
)
Geo.validate(da_geo)
```

### `to_schema()` and `validate()`

```python
schema = Temperature.to_schema()
print(type(schema))

Temperature.validate(da)
```

### Error on missing `data` field

If the `data` field is omitted, calling `to_schema()` raises
[`SchemaInitError`](../reference/generated/pandera.errors.SchemaInitError.html):

```python
class Bad(pa.DataArrayModel):
    x: Coordinate[np.float64]

    class Config:
        dims = ("x",)

try:
    Bad.to_schema()
except pa.errors.SchemaInitError as exc:
    print(exc)
```

## `DatasetModel`

### Basic usage

Data variable fields are annotated with a dtype, and coordinate fields use
`Coordinate[dtype]`:

```python
class Surface(pa.DatasetModel):
    temperature: np.float64 = pa.Field(dims=("x", "y"))
    pressure: np.float64 = pa.Field(dims=("x", "y"))
    x: Coordinate[np.float64]
    y: Coordinate[np.float64]

    class Config:
        strict = True

ds = xr.Dataset(
    {
        "temperature": (("x", "y"), np.random.rand(3, 4)),
        "pressure": (("x", "y"), np.random.rand(3, 4)),
    },
    coords={
        "x": np.arange(3, dtype=np.float64),
        "y": np.arange(4, dtype=np.float64),
    },
)
Surface.validate(ds)
```

### `Config` options

`DatasetModel.Config`
([`DatasetConfig`](../reference/generated/pandera.api.xarray.model_config.DatasetConfig.html)) accepts:
`strict`, `strict_coords`, `strict_attrs`, `dims`, `sizes`, plus the
common `name`, `title`, `description`, `coerce`.

### `Field` on data variables

`Field` on a data-variable annotation supports `dims`, `sizes`, `shape`,
`aligned_with`, `broadcastable_with`, `required`, and all the built-in check
keywords:

```python
class BoundedGrid(pa.DatasetModel):
    temperature: np.float64 = pa.Field(dims=("x", "y"), ge=150, le=350)
    x: Coordinate[np.float64]
    y: Coordinate[np.float64]

ds_bounded = xr.Dataset(
    {"temperature": (("x", "y"), np.full((3, 4), 273.15))},
    coords={
        "x": np.arange(3, dtype=np.float64),
        "y": np.arange(4, dtype=np.float64),
    },
)
BoundedGrid.validate(ds_bounded)
```

### Nested `DataArrayModel`

Instead of a bare dtype, annotate a data variable with a `DataArrayModel`
subclass to reuse a full array schema:

```python
class TemperatureArray(pa.DataArrayModel):
    data: np.float64 = pa.Field()
    time: Coordinate[np.float64]

    class Config:
        dims = ("time",)
        name = "temperature"

class Climate(pa.DatasetModel):
    temperature: TemperatureArray
    time: Coordinate[np.float64]

ds_climate = xr.Dataset(
    {"temperature": (("time",), np.ones(12))},
    coords={"time": np.arange(12, dtype=np.float64)},
)
Climate.validate(ds_climate)
```

The nested model compiles to a
[`DataArraySchema`](../reference/generated/pandera.api.xarray.container.DataArraySchema.html) inside the dataset's
`data_vars`.

### Optional variables

Use `T | None` with `Field(required=False)`:

```python
class Flexible(pa.DatasetModel):
    required_var: np.float64 = pa.Field(dims=("x",))
    optional_var: np.float64 | None = pa.Field(dims=("x",), required=False)
    x: Coordinate[np.float64]

ds_minimal = xr.Dataset(
    {"required_var": (("x",), np.ones(3))},
    coords={"x": np.arange(3, dtype=np.float64)},
)
Flexible.validate(ds_minimal)
```

### `to_schema()` and `validate()`

```python
schema = Surface.to_schema()
print(type(schema))

Surface.validate(ds)
```

## Schema inheritance

Models support regular Python inheritance. Child classes inherit fields and
`Config` options, and can override them:

```python
class BaseGrid(pa.DataArrayModel):
    data: np.float64 = pa.Field()
    x: Coordinate[np.float64]

    class Config:
        dims = ("x",)

class DetailedGrid(BaseGrid):
    y: Coordinate[np.float64]

    class Config:
        dims = ("x", "y")
        name = "detailed"

da_detailed = xr.DataArray(
    np.ones((3, 4)),
    dims=("x", "y"),
    coords={
        "x": np.arange(3, dtype=np.float64),
        "y": np.arange(4, dtype=np.float64),
    },
    name="detailed",
)
DetailedGrid.validate(da_detailed)
```

## Excluded attributes

Class variables starting with an underscore (`_`) are excluded from the
model. `Config` is a reserved name.

## See also

- [DataArray Schemas](./data_array_schema.html) / [DatasetSchema](./dataset_schema.html) — imperative API
- [DataTree Validation](./data_tree.html) — `DataTreeSchema` and `DataTreeModel`
- [Checks and Parsers](./checks_and_parsers.html) — checks, parsers, lazy validation
- [Decorators](./decorators.html) — `check_input`, `check_output`, `check_io`, and `check_types`
- [Configuration](./configuration.html) — [`ValidationDepth`](../reference/generated/pandera.config.ValidationDepth.html),
  [`ValidationScope`](../reference/generated/pandera.config.ValidationScope.html), Dask, environment variables
- [DataFrame Models](../dataframe_models.html) — dataframe class-based API (same patterns)
- [Xarray API reference](../reference/xarray.html) — full API reference for all xarray classes
- [Core API reference](../reference/core.html) — [`PanderaConfig`](../reference/generated/pandera.config.PanderaConfig.html),
  [`ValidationDepth`](../reference/generated/pandera.config.ValidationDepth.html),
  [`ValidationScope`](../reference/generated/pandera.config.ValidationScope.html)
