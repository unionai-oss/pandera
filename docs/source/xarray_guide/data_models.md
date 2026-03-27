---
file_format: myst
---

(xarray-data-models)=

# Data Models

{class}`~pandera.api.xarray.model.DataArrayModel` and
{class}`~pandera.api.xarray.model.DatasetModel` provide a class-based,
pydantic-style API for defining xarray schemas — the same pattern as
{ref}`DataFrameModel <dataframe-models>` for pandas.

Type annotations and {func}`~pandera.api.xarray.model_components.Field`
descriptors define the schema; call `validate()` or `to_schema()` to use it.

The imperative counterparts are {ref}`xarray-data-array-schema` and
{ref}`xarray-dataset-schema`.

## Annotation vs component `Coordinate`

Two objects share the name "Coordinate":

- **{class}`~pandera.typing.xarray.Coordinate`** (from
  `pandera.typing.xarray`) — a *typing marker* used in model annotations, like
  `Index` in a `DataFrameModel`.
- **{class}`~pandera.api.xarray.components.Coordinate`** (from
  `pandera.xarray`) — the schema *component* you pass to
  `DataArraySchema(coords=...)`.

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
import pandera.xarray as pa
from pandera.typing.xarray import Coordinate

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
print(Temperature.time)  # "time"
print(Temperature.lat)   # "lat"
```

### `Config` options

`DataArrayModel.Config`
({class}`~pandera.api.xarray.model_config.DataArrayConfig`) accepts:
`dtype`, `dims`, `sizes`, `shape`, `name`, `coerce`, `nullable`,
`strict_coords`, `strict_attrs`, `attrs`, `chunked`, `array_type`.

These mirror the keyword arguments on
{class}`~pandera.api.xarray.container.DataArraySchema`.

### Using `Field` on `data`

The `data` field can carry the same per-field structural constraints that
you would pass as `DataArraySchema` constructor arguments:

```python
class Grid(pa.DataArrayModel):
    data: np.float64 = pa.Field(
        dims=("x", "y"),
        sizes={"x": 100, "y": 200},
    )
    x: Coordinate[np.float64]
    y: Coordinate[np.float64]

    class Config:
        name = "grid"
```

When both `Field(dims=...)` and `Config.dims` are set, the `Field` value
takes precedence.

### Using `Field` on coordinates

Coordinate fields accept the same built-in check keywords as
{func}`~pandera.api.dataframe.model_components.Field`: `eq`, `ge`, `le`,
`in_range`, `isin`, etc. Plus `nullable` and `coerce`.

```python
class Geo(pa.DataArrayModel):
    data: np.float64 = pa.Field()
    lat: Coordinate[np.float64] = pa.Field(ge=-90, le=90)
    lon: Coordinate[np.float64] = pa.Field(ge=-180, le=180)

    class Config:
        dims = ("lat", "lon")
```

### `to_schema()` and `validate()`

```python
schema = Temperature.to_schema()   # returns DataArraySchema
Temperature.validate(da)           # equivalent to schema.validate(da)
Temperature(da)                    # syntactic sugar
```

### Error on missing `data` field

If the `data` field is omitted, calling `to_schema()` raises
{class}`~pandera.errors.SchemaInitError`:

```python
class Bad(pa.DataArrayModel):
    x: Coordinate[np.float64]

    class Config:
        dims = ("x",)

# Bad.to_schema()  ->  SchemaInitError: DataArrayModel requires a 'data' field.
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
({class}`~pandera.api.xarray.model_config.DatasetConfig`) accepts:
`strict`, `strict_coords`, `strict_attrs`, `dims`, `sizes`, plus the
common `name`, `title`, `description`, `coerce`.

### `Field` on data variables

`Field` on a data-variable annotation supports `dims`, `sizes`, `shape`,
`aligned_with`, `broadcastable_with`, `required`, and all the built-in check
keywords:

```python
class Grid(pa.DatasetModel):
    temperature: np.float64 = pa.Field(
        dims=("x", "y"),
        ge=150,
        le=350,
    )
    x: Coordinate[np.float64]
    y: Coordinate[np.float64]
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
```

The nested model compiles to a
{class}`~pandera.api.xarray.container.DataArraySchema` inside the dataset's
`data_vars`.

### Optional variables

Use `T | None` with `Field(required=False)`:

```python
class Flexible(pa.DatasetModel):
    required_var: np.float64 = pa.Field(dims=("x",))
    optional_var: np.float64 | None = pa.Field(dims=("x",), required=False)
    x: Coordinate[np.float64]
```

### `to_schema()` and `validate()`

```python
schema = Surface.to_schema()   # returns DatasetSchema
Surface.validate(ds)
Surface(ds)
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
```

## Excluded attributes

Class variables starting with an underscore (`_`) are excluded from the
model. `Config` is a reserved name.

## See also

- {ref}`xarray-data-array-schema` / {ref}`xarray-dataset-schema` — imperative API
- {ref}`xarray-checks-parsers` — checks, parsers, lazy validation
- {ref}`xarray-configuration` — validation depth, Dask, environment variables
- {ref}`dataframe-models` — dataframe class-based API (same patterns)
