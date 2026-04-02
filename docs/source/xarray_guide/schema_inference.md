---
file_format: mystnb
---

(xarray-schema-inference)=

# Schema Inference

Automatically infer a schema from an existing {class}`~xarray.DataArray` or
{class}`~xarray.Dataset`:

```{code-cell} python
import numpy as np
import xarray as xr
import pandera.xarray as pa

da = xr.DataArray(
    np.random.rand(3, 4),
    dims=("x", "y"),
    coords={"x": [1.0, 2.0, 3.0]},
    name="temperature",
)

schema = pa.infer_schema(da)
print(type(schema).__name__)
print(f"dims={schema.dims}, name={schema.name}")
```

For datasets:

```{code-cell} python
ds = xr.Dataset(
    {
        "temperature": (("x", "y"), np.random.rand(3, 4)),
        "pressure": (("x", "y"), np.random.rand(3, 4)),
    },
    coords={"x": [1.0, 2.0, 3.0]},
)

ds_schema = pa.infer_schema(ds)
print(type(ds_schema).__name__)
print(f"data_vars: {list(ds_schema.data_vars.keys())}")
```

The inferred schema captures dtype, dims, coords, nullable status, and
min/max bounds for numeric data. Re-validate the same data:

```{code-cell} python
schema.validate(da)
ds_schema.validate(ds)
```

```{admonition} See also
:class: tip

{ref}`xarray-io-serialization` for saving and loading schemas as YAML or JSON.
```
