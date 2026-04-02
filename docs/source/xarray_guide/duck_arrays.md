---
file_format: mystnb
---

(xarray-duck-arrays)=

# Dask and Duck Arrays

xarray can wrap any array type that implements NumPy's array protocol — Dask
arrays, sparse arrays, CuPy arrays, and more. Pandera's xarray backend
validates these **duck arrays** with two complementary mechanisms:

1. **Schema parameters** — `chunked` and `array_type` on
   {class}`~pandera.api.xarray.container.DataArraySchema` and
   {class}`~pandera.api.xarray.components.DataVar`.
2. **Validation depth** — automatic or manual control over whether data-level
   checks trigger computation on lazy backends.

## `chunked` — require or forbid lazy backing

```{code-cell} python
import numpy as np
import xarray as xr
import pandera.xarray as pa

da_eager = xr.DataArray(np.ones(10), dims="x")

pa.DataArraySchema(chunked=False).validate(da_eager)
```

When `chunked=True`, the underlying data must be chunked (i.e.
`da.chunks is not None`):

```{code-cell} python
da_dask = da_eager.chunk({"x": 5})

pa.DataArraySchema(chunked=True).validate(da_dask)
```

Passing an eager array to a `chunked=True` schema raises an error:

```{code-cell} python
try:
    pa.DataArraySchema(chunked=True).validate(da_eager)
except pa.errors.SchemaError as exc:
    print(exc)
```

Set `chunked=None` (the default) to accept either.

## `array_type` — assert the concrete storage type

```{code-cell} python
import dask.array

pa.DataArraySchema(array_type=dask.array.Array).validate(da_dask)
pa.DataArraySchema(array_type=np.ndarray).validate(da_eager)
```

When the actual type does not match:

```{code-cell} python
try:
    pa.DataArraySchema(array_type=np.ndarray).validate(da_dask)
except pa.errors.SchemaError as exc:
    print(exc)
```

## Structural checks run without `.compute()`

Pandera classifies every validation rule into a **scope**:

- **Schema scope** — dtype, dims, sizes, shape, coords, attrs, encoding, name,
  `chunked`, `array_type`. These inspect metadata only and never trigger
  computation.
- **Data scope** — `Check` objects and `nullable`. These need actual values.

For chunked (Dask-backed) data, **schema-scope checks always run**. Data-scope
checks are governed by {class}`~pandera.config.ValidationDepth`.

```{code-cell} python
schema = pa.DataArraySchema(
    dtype=np.float64,
    dims=("x",),
    sizes={"x": 10},
    name="values",
)

da_named = da_dask.rename("values")
schema.validate(da_named)
```

No `.compute()` was called — only metadata was inspected.

## Validation depth with Dask

By default, chunked arrays use `SCHEMA_ONLY` depth to avoid surprise
computation. You can override this:

```{code-cell} python
from pandera.config import ValidationDepth, config_context

schema_with_checks = pa.DataArraySchema(
    dtype=np.float64,
    dims=("x",),
    checks=pa.Check(lambda da: float(da.min()) >= 0),
)

with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
    schema_with_checks.validate(da_dask)
```

:::{note}
Setting `SCHEMA_AND_DATA` on chunked arrays will call `.compute()` during
validation. Be mindful of memory and compute costs for large datasets.
:::

Or set the environment variable:

```bash
export PANDERA_VALIDATION_DEPTH=SCHEMA_AND_DATA
```

See {ref}`xarray-configuration` for the full resolution order.

## Datasets with Dask-backed variables

`chunked` and `array_type` work on
{class}`~pandera.api.xarray.components.DataVar` inside a
{class}`~pandera.api.xarray.container.DatasetSchema`:

```{code-cell} python
ds = xr.Dataset({
    "temperature": (("x", "y"), np.random.rand(3, 4)),
    "pressure": (("x", "y"), np.random.rand(3, 4)),
}).chunk({"x": 2})

schema = pa.DatasetSchema(
    data_vars={
        "temperature": pa.DataVar(
            dtype=np.float64, dims=("x", "y"), chunked=True,
        ),
        "pressure": pa.DataVar(
            dtype=np.float64, dims=("x", "y"), chunked=True,
        ),
    },
)
schema.validate(ds)
```

## Lazy error collection

Lazy validation (`lazy=True`) works with Dask arrays. Structural errors are
collected without triggering computation:

```{code-cell} python
bad_ds = xr.Dataset({
    "temperature": (("z",), np.ones(3)),
}).chunk({"z": 2})

schema = pa.DatasetSchema(
    data_vars={
        "temperature": pa.DataVar(
            dtype=np.float64, dims=("x", "y"), chunked=True,
        ),
        "pressure": pa.DataVar(
            dtype=np.float64, dims=("x", "y"), chunked=True,
        ),
    },
)

try:
    schema.validate(bad_ds, lazy=True)
except pa.errors.SchemaErrors as exc:
    print(exc)
```

Validating both schema- and data-level checks triggers computation on the Dask array.

```{code-cell} python
with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
    try:
        schema.validate(bad_ds, lazy=True)
    except pa.errors.SchemaErrors as exc:
        print(exc)
```

## See also

- {ref}`xarray-configuration` — validation depth resolution, disabling
  validation
- {ref}`xarray-data-array-schema` — `DataArraySchema` parameters
- {ref}`xarray-dataset-schema` — `DatasetSchema` and `DataVar`
- {ref}`xarray-checks-parsers` — checks, parsers, lazy validation
