---
file_format: myst
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

```python
from pandera.config import ValidationDepth, config_context

with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
    schema.validate(dask_backed_da)
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
returns the input unchanged.

## See also

- {ref}`xarray-checks-parsers` — checks, parsers, and lazy validation
- {ref}`xarray-decorators` — `check_input`, `check_output`, `check_io`, and `check_types`
- {ref}`xarray-data-array-schema` — `DataArraySchema` details
- {ref}`xarray-dataset-schema` — `DatasetSchema` details
- {ref}`xarray-data-models` — class-based models
- {ref}`configuration` — global `ValidationDepth`, `ValidationScope`, env vars
