---
file_format: mystnb
---

(xarray-configuration)=

# Configuration

## Validation depth and Dask / chunked data

Pandera uses {class}`~pandera.config.ValidationDepth` for xarray the same way
it does for Polars lazy frames:

- **`SCHEMA_ONLY`** — only structural validation (dims, dtype, coords, attrs,
  name, shape). Data-level `Check` objects are skipped.
- **`DATA_ONLY`** — only data-level checks.
- **`SCHEMA_AND_DATA`** — full validation (default for eager arrays).

### Chunked (Dask-backed) arrays

When an array is backed by Dask (i.e. `da.chunks is not None`), data-level
checks would trigger `.compute()`, which may be expensive. To avoid
surprises, **chunked arrays default to `SCHEMA_ONLY`** when no explicit depth
is set. Eager (NumPy-backed) arrays default to `SCHEMA_AND_DATA`.

### Opting in to data checks on Dask arrays

Set the validation depth explicitly:

```{code-cell} python
import numpy as np
import xarray as xr
import pandera.xarray as pa
from pandera.config import ValidationDepth, config_context

schema = pa.DataArraySchema(
    dtype=np.float64,
    dims=("x",),
    checks=pa.Check(lambda da: float(da.min()) >= 0),
)

da = xr.DataArray(np.ones(5), dims="x")

with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
    schema.validate(da)
```

Or set the environment variable before running your program:

```bash
export PANDERA_VALIDATION_DEPTH=SCHEMA_AND_DATA
```

### Resolution order

{func}`~pandera.api.xarray.utils.get_validation_depth` resolves the depth
in this order:

1. Active `config_context(validation_depth=...)` — highest priority.
2. Global config (`PANDERA_VALIDATION_DEPTH` env var or
   `PanderaConfig.validation_depth`).
3. Per-object default — `SCHEMA_ONLY` for chunked data, `SCHEMA_AND_DATA`
   for eager data.

## Disabling validation

Set `PANDERA_VALIDATION_ENABLED=false` (env var) or use
`config_context(validation_enabled=False)` to make `validate()` a no-op that
returns the input unchanged:

```{code-cell} python
with config_context(validation_enabled=False):
    bad_da = xr.DataArray([-999], dims="z", name="wrong")
    result = schema.validate(bad_da)
    print(f"Validation skipped, returned: {result.values}")
```

## See also

- {ref}`xarray-duck-arrays` — Dask integration, `chunked`, `array_type`, and
  lazy validation
- {ref}`xarray-checks-parsers` — checks, parsers, and lazy validation
- {ref}`xarray-decorators` — `check_input`, `check_output`, `check_io`, and `check_types`
- {ref}`xarray-data-array-schema` — {class}`~pandera.api.xarray.container.DataArraySchema` details
- {ref}`xarray-dataset-schema` — {class}`~pandera.api.xarray.container.DatasetSchema` details
- {ref}`xarray-data-models` — class-based {class}`~pandera.api.xarray.model.DataArrayModel` / {class}`~pandera.api.xarray.model.DatasetModel`
- {ref}`api-xarray` — full API reference for all xarray classes
- {ref}`api-core` — {class}`~pandera.config.PanderaConfig`,
  {class}`~pandera.config.ValidationDepth`,
  {class}`~pandera.config.ValidationScope`
- {ref}`configuration` — global {class}`~pandera.config.ValidationDepth`, {class}`~pandera.config.ValidationScope`, env vars
