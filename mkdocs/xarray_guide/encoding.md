# Encoding Validation

When xarray reads data from netCDF or Zarr, each variable and the dataset
itself carry an `.encoding` dict that describes how values were serialized on
disk — fill values, scale factors, compression settings, and more.  Pandera
lets you validate these encoding dicts at three levels:

| Level | Schema parameter | Validated against |
|-------|-----------------|-------------------|
| Per-variable (DataArray) | `DataArraySchema(encoding=...)` | `da.encoding` |
| Per-variable (Dataset) | `DataVar(encoding=...)` | `ds[var].encoding` |
| Dataset-level | `DatasetSchema(encoding=...)` | `ds.encoding` |

## DataArray encoding

The `encoding` parameter on [`DataArraySchema`](../reference/generated/pandera.api.xarray.container.DataArraySchema.html)
validates the DataArray's `.encoding` dict.

### Dict-based validation

Values in the dict are matched using the same rules as `attrs`:

- **Literal values** — matched by equality.
- **Regex patterns** — strings starting with `^` are matched via
  `re.fullmatch`.
- **Callable predicates** — `(value) -> bool`.

```python
import numpy as np
import xarray as xr
import pandera.xarray as pa

da = xr.DataArray(np.arange(6, dtype="float64"), dims="x")
da.encoding = {
    "_FillValue": -999.0,
    "dtype": "float32",
    "scale_factor": 0.01,
}

schema = pa.DataArraySchema(
    encoding={
        "_FillValue": -999.0,
        "dtype": "^float.*",
        "scale_factor": lambda v: 0 < v < 1,
    },
)
schema.validate(da)
```

When a key is missing or a value doesn't match:

```python
da_bad = xr.DataArray(np.ones(3), dims="x")
da_bad.encoding = {"dtype": "int32"}

try:
    pa.DataArraySchema(
        encoding={"dtype": "^float.*"},
    ).validate(da_bad, lazy=True)
except pa.errors.SchemaErrors as exc:
    print(exc)
```

### Pydantic model

For structured validation, pass a `pydantic.BaseModel` **class**.
Pandera delegates to pydantic and converts each pydantic error into a
pandera `SchemaError`:

```python
from pydantic import BaseModel, Field as PydanticField

class NetCDFEncoding(BaseModel):
    dtype: str
    complevel: int = PydanticField(ge=1, le=9)
    scale_factor: float
```

```python
da_enc = xr.DataArray(np.ones(3), dims="x")
da_enc.encoding = {
    "dtype": "float32",
    "complevel": 4,
    "scale_factor": 0.01,
}

pa.DataArraySchema(encoding=NetCDFEncoding).validate(da_enc)
```

When pydantic validation fails:

```python
da_bad_enc = xr.DataArray(np.ones(3), dims="x")
da_bad_enc.encoding = {
    "dtype": "float32",
    "complevel": 99,
    "scale_factor": "not_a_number",
}

try:
    pa.DataArraySchema(encoding=NetCDFEncoding).validate(
        da_bad_enc, lazy=True,
    )
except pa.errors.SchemaErrors as exc:
    print(exc)
```

## Per-variable encoding in a Dataset

Use [`DataVar`](../reference/generated/pandera.api.xarray.components.DataVar.html) with `encoding=...` to
validate per-variable encoding within a
[`DatasetSchema`](../reference/generated/pandera.api.xarray.container.DatasetSchema.html). This validates against
`ds[var_name].encoding`:

```python
ds = xr.Dataset({
    "temperature": (("x",), np.arange(4.0)),
    "pressure": (("x",), np.ones(4)),
})
ds["temperature"].encoding = {
    "_FillValue": -999.0,
    "scale_factor": 0.01,
}
ds["pressure"].encoding = {
    "_FillValue": -999.0,
    "zlib": True,
}

schema = pa.DatasetSchema(
    data_vars={
        "temperature": pa.DataVar(
            dims=("x",),
            encoding={
                "_FillValue": -999.0,
                "scale_factor": lambda v: 0 < v < 1,
            },
        ),
        "pressure": pa.DataVar(
            dims=("x",),
            encoding={
                "_FillValue": -999.0,
                "zlib": True,
            },
        ),
    },
)
schema.validate(ds)
```

Pydantic models also work on `DataVar(encoding=...)`:

```python
class VarEncoding(BaseModel):
    scale_factor: float
    dtype: str

ds2 = xr.Dataset({"temp": (("x",), np.ones(3))})
ds2["temp"].encoding = {"scale_factor": 0.01, "dtype": "float32"}

schema = pa.DatasetSchema(
    data_vars={
        "temp": pa.DataVar(dims=("x",), encoding=VarEncoding),
    },
)
schema.validate(ds2)
```

## Dataset-level encoding

The `encoding` parameter on
[`DatasetSchema`](../reference/generated/pandera.api.xarray.container.DatasetSchema.html) validates
`ds.encoding` — the **dataset-level** encoding metadata. Common keys include
`unlimited_dims` and `source`:

```python
ds_enc = xr.Dataset({"temp": (("x",), np.ones(3))})
ds_enc.encoding = {"unlimited_dims": ["x"]}

schema = pa.DatasetSchema(
    data_vars={"temp": pa.DataVar(dims=("x",))},
    encoding={"unlimited_dims": ["x"]},
)
schema.validate(ds_enc)
```

Dataset-level encoding also supports pydantic models:

```python
class DatasetEncoding(BaseModel):
    unlimited_dims: list[str]

schema = pa.DatasetSchema(
    data_vars={"temp": pa.DataVar(dims=("x",))},
    encoding=DatasetEncoding,
)
schema.validate(ds_enc)
```

## Combining per-variable and dataset-level encoding

Per-variable and dataset-level encoding are validated independently and can
be used together:

```python
ds_full = xr.Dataset({"temp": (("x",), np.arange(4.0))})
ds_full.encoding = {"unlimited_dims": ["x"]}
ds_full["temp"].encoding = {"_FillValue": -999.0}

schema = pa.DatasetSchema(
    data_vars={
        "temp": pa.DataVar(
            dims=("x",),
            encoding={"_FillValue": -999.0},
        ),
    },
    encoding={"unlimited_dims": ["x"]},
)
schema.validate(ds_full)
```

## `Check.has_encoding()` — check-based alternative

For ad hoc validation or dataset-level checks, use
[`has_encoding`](../reference/generated/pandera.api.checks.Check.html):

```python
da = xr.DataArray(np.ones(3), dims="x")
da.encoding = {"_FillValue": -999.0, "dtype": "float32"}

pa.DataArraySchema(
    checks=pa.Check.has_encoding({"_FillValue": -999.0}),
).validate(da)
```

!!! note

    The schema-level `encoding=` parameter is preferred over `Check.has_encoding()`
    when you know the expected encoding upfront. It provides richer error messages
    and supports regex, callable, and pydantic matching modes.

## Encoding is schema-scope

Encoding validation is classified as **schema scope** — it runs even under
`ValidationDepth.SCHEMA_ONLY` and never triggers `.compute()` on Dask-backed
arrays. This makes it safe for lazy pipelines.

## See also

- [DataArray Schemas](./data_array_schema.html) — `DataArraySchema` details
- [DatasetSchema](./dataset_schema.html) — `DatasetSchema` and `DataVar`
- [Duck Arrays](./duck_arrays.html) — Dask integration and validation depth
- [Configuration](./configuration.html) — validation depth and scope
- [xarray I/O docs](https://docs.xarray.dev/en/stable/user-guide/io.html#reading-encoded-data) — how xarray populates `.encoding`
