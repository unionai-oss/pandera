---
file_format: myst
---

(xarray-decorators)=

# Decorators

## `check_input` / `check_output`

These accept any schema object (including `DataArraySchema` and
`DatasetSchema`) and validate function arguments or return values:

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
```

## `check_io`

`check_io` combines input and output validation in a single decorator.
Pass keyword arguments matching parameter names for inputs and `out` for
the return value:

```python
in_schema = pa.DataArraySchema(dtype=np.float64, dims=("x",))
out_schema = pa.DataArraySchema(dtype=np.float64, dims=("x",))

@pa.check_io(da=in_schema, out=out_schema)
def process(da: xr.DataArray) -> xr.DataArray:
    return da * 2
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
```

For datasets:

```python
class Surface(pa.DatasetModel):
    temperature: np.float64 = pa.Field(dims=("x",))
    x: Coordinate[np.float64]

@pa.check_types
def process_dataset(ds: Dataset[Surface]) -> Dataset[Surface]:
    return ds
```

Mixed annotations work too — for example a function that takes a
`DataArray` and returns a `Dataset`:

```python
@pa.check_types
def to_dataset(
    da: DataArray[Temperature],
) -> Dataset[Surface]:
    return da.to_dataset()
```

Pass `lazy=True` to collect all validation errors instead of failing on
the first one:

```python
@pa.check_types(lazy=True)
def transform(da: DataArray[Temperature]) -> DataArray[Temperature]:
    return da * 2
```

See {ref}`decorators` for the full decorator API.

## See also

- {ref}`xarray-checks-parsers` — checks, parsers, and lazy validation
- {ref}`xarray-data-models` — class-based models
- {ref}`xarray-configuration` — validation depth, Dask, and environment variables
