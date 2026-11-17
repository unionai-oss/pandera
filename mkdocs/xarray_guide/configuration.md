# Configuration

## Validation depth and Dask / chunked data

Pandera uses [`ValidationDepth`](../reference/generated/pandera.config.ValidationDepth.html) for xarray the same way
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

```python
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

[`get_validation_depth`](../reference/generated/pandera.api.xarray.utils.get_validation_depth.html) resolves the depth
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

```python
with config_context(validation_enabled=False):
    bad_da = xr.DataArray([-999], dims="z", name="wrong")
    result = schema.validate(bad_da)
    print(f"Validation skipped, returned: {result.values}")
```

## See also

- [Duck Arrays](./duck_arrays.html) — Dask integration, `chunked`, `array_type`, and
  lazy validation
- [Checks and Parsers](./checks_and_parsers.html) — checks, parsers, and lazy validation
- [Decorators](./decorators.html) — `check_input`, `check_output`, `check_io`, and `check_types`
- [DataArraySchema](./data_array_schema.html) — [`DataArraySchema`](../reference/generated/pandera.api.xarray.container.DataArraySchema.html) details
- [DatasetSchema](./dataset_schema.html) — [`DatasetSchema`](../reference/generated/pandera.api.xarray.container.DatasetSchema.html) details
- [Data Models](./data_models.html) — class-based [`DataArrayModel`](../reference/generated/pandera.api.xarray.model.DataArrayModel.html) / [`DatasetModel`](../reference/generated/pandera.api.xarray.model.DatasetModel.html)
- [Xarray API reference](../reference/xarray.html) — full API reference for all xarray classes
- [Core API reference](../reference/core.html) — [`PanderaConfig`](../reference/generated/pandera.config.PanderaConfig.html),
  [`ValidationDepth`](../reference/generated/pandera.config.ValidationDepth.html),
  [`ValidationScope`](../reference/generated/pandera.config.ValidationScope.html)
- [Configuration](../configuration.html) — global [`ValidationDepth`](../reference/generated/pandera.config.ValidationDepth.html), [`ValidationScope`](../reference/generated/pandera.config.ValidationScope.html), env vars
