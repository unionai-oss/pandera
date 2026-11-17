# Decorators

## `check_input` / `check_output`

These accept any schema object, including
[`DataArraySchema`](../reference/generated/pandera.api.xarray.container.DataArraySchema.html) and
[`DatasetSchema`](../reference/generated/pandera.api.xarray.container.DatasetSchema.html), and validate function
arguments or return values:

```python
import numpy as np
import xarray as xr
import pandera.xarray as pa

schema = pa.DataArraySchema(dtype=np.float64, dims=("x",))

@pa.check_input(schema, "da")
def process(da: xr.DataArray) -> xr.DataArray:
    return da * 2

@pa.check_output(schema)
def generate() -> xr.DataArray:
    return xr.DataArray(np.ones(3), dims="x")

da = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims="x")
process(da)
```

```python
generate()
```

Validation errors are raised if the input or output doesn't match:

```python
bad_da = xr.DataArray(np.zeros(3), dims=("z",))

try:
    process(bad_da)
except pa.errors.SchemaError as exc:
    print(exc)
```

## `check_io`

`check_io` combines input and output validation in a single decorator.
Pass keyword arguments matching parameter names for inputs and `out` for
the return value:

```python
in_schema = pa.DataArraySchema(dtype=np.float64, dims=("x",))
out_schema = pa.DataArraySchema(dtype=np.float64, dims=("x",))

@pa.check_io(da=in_schema, out=out_schema)
def scale(da: xr.DataArray) -> xr.DataArray:
    return da * 10

da = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims="x")
scale(da)
```

## `check_types`

`check_types` inspects type annotations and validates against the
referenced model. Use the generic types from `pandera.typing.xarray`:

```python
from pandera.typing.xarray import Coordinate, DataArray, Dataset

class Temperature(pa.DataArrayModel):
    data: np.float64 = pa.Field()
    x: Coordinate[np.float64]

    class Config:
        dims = ("x",)
        name = "temperature"

@pa.check_types
def transform(da: DataArray[Temperature]) -> DataArray[Temperature]:
    return da * 2

da = xr.DataArray(
    np.ones(5),
    dims="x",
    coords={"x": np.arange(5, dtype=np.float64)},
    name="temperature",
)
transform(da)
```

For datasets:

```python
class Surface(pa.DatasetModel):
    temperature: np.float64 = pa.Field(dims=("x",))
    x: Coordinate[np.float64]

@pa.check_types
def process_dataset(ds: Dataset[Surface]) -> Dataset[Surface]:
    return ds

ds = xr.Dataset(
    {"temperature": (("x",), np.ones(3))},
    coords={"x": np.arange(3, dtype=np.float64)},
)
process_dataset(ds)
```

Mixed annotations work too — for example a function that takes a
`DataArray` and returns a `Dataset`:

```python
@pa.check_types
def to_dataset(da: DataArray[Temperature]) -> Dataset[Surface]:
    return xr.Dataset(
        {"temperature": da},
        coords={"x": da.coords["x"]},
    )

to_dataset(da)
```

Pass `lazy=True` to collect all validation errors instead of failing on
the first one:

```python
@pa.check_types(lazy=True)
def strict_transform(da: DataArray[Temperature]) -> DataArray[Temperature]:
    return xr.DataArray(np.ones(3), dims=("z",), name="bad")

try:
    strict_transform(da)
except pa.errors.SchemaErrors as exc:
    print(exc)
```

For `DataTree` models:

```python
from pandera.typing.xarray import DataTree

class SurfaceDS(pa.DatasetModel):
    temperature: np.float64 = pa.Field(dims=("x",))
    x: Coordinate[np.float64]

class MyTree(pa.DataTreeModel):
    surface: SurfaceDS

@pa.check_types
def process_tree(dt: DataTree[MyTree]) -> DataTree[MyTree]:
    return dt

dt = xr.DataTree.from_dict({
    "/": xr.Dataset(),
    "/surface": xr.Dataset(
        {"temperature": (("x",), np.ones(3))},
        coords={"x": np.arange(3, dtype=np.float64)},
    ),
})
process_tree(dt)
```

See [Decorators](../decorators.html) for the full decorator API.

## See also

- [Checks and Parsers](./checks_and_parsers.html) — checks, parsers, and lazy validation
- [Data Models](./data_models.html) — class-based [`DataArrayModel`](../reference/generated/pandera.api.xarray.model.DataArrayModel.html) / [`DatasetModel`](../reference/generated/pandera.api.xarray.model.DatasetModel.html)
- [DataTree Validation](./data_tree.html) — `DataTreeSchema` and `DataTreeModel`
- [Configuration](./configuration.html) — [`ValidationDepth`](../reference/generated/pandera.config.ValidationDepth.html),
  [`ValidationScope`](../reference/generated/pandera.config.ValidationScope.html), Dask, and environment variables
- [Xarray API reference](../reference/xarray.html) — full API reference for all xarray classes
- [Decorators](../decorators.html) — general decorator API (pandas-oriented)
