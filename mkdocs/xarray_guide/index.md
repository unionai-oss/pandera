# Xarray Data Validation

*New in 0.31.0*

[xarray](https://docs.xarray.dev/) provides labelled multi-dimensional arrays
`xarray.DataArray`, collections of aligned arrays
`xarray.Dataset`, and collections of datasets with `xarray.DataTree`.

Pandera validates them with the same patterns as the other dataframe backends:
schema objects, optional [`Check`](../reference/generated/pandera.api.checks.Check.html)
instances, and global [configuration](../configuration.html).

## Installation

```bash
pip install 'pandera[xarray]'
```

## Quick start

```python
import numpy as np
import xarray as xr
import pandera.xarray as pa

schema = pa.DatasetSchema(
    data_vars={
        "temperature": pa.DataVar(dtype=np.float64, dims=("x", "y")),
        "pressure": pa.DataVar(dtype=np.float64, dims=("x", "y")),
    },
    coords={"x": pa.Coordinate(dtype=np.float64)},
)

ds = xr.Dataset(
    {
        "temperature": (("x", "y"), np.random.rand(3, 4)),
        "pressure": (("x", "y"), np.random.rand(3, 4)),
    },
    coords={"x": np.arange(3, dtype=np.float64)},
)
schema.validate(ds)
```

## Dataset Model

```python
from pandera.typing.xarray import Coordinate

class Surface(pa.DatasetModel):
    temperature: np.float64 = pa.Field(dims=("x", "y"))
    pressure: np.float64 = pa.Field(dims=("x", "y"))
    x: Coordinate[np.float64]

Surface.validate(ds)
```

## Guide contents

- [DataArraySchema](./data_array_schema.html) — validating a single `xarray.DataArray`
- [DatasetSchema](./dataset_schema.html) — validating a `xarray.Dataset` with `DataVar` and `Coordinate`
- [DataTreeSchema](./data_tree.html) — validating a `xarray.DataTree` hierarchy
- [Data Models](./data_models.html) — class-based `DataArrayModel`, `DatasetModel`, and `DataTreeModel`
- [Checks and Parsers](./checks_and_parsers.html) — checks, parsers, and lazy validation
- [Decorators](./decorators.html) — `check_input`, `check_output`, `check_io`, and `check_types`
- [Configuration](./configuration.html) — validation depth, Dask, and environment variables
- [Duck Arrays](./duck_arrays.html) — `chunked`, `array_type`, validation depth, and lazy data checks
- [Encoding](./encoding.html) — validate `.encoding` dicts on DataArrays, DataVars, and Datasets
- [Error Reporting](./error_reporting.html) — `SchemaError` / `SchemaErrors`, lazy validation, and failure cases
- [CF Convention Checks](./cf_conventions.html) — CF standard name, units, and `cf_xarray` checks
- [Schema Inference](./schema_inference.html) — automatically infer schemas from data
- [Serialization](./io_serialization.html) — save and load schemas as YAML or JSON
- [Hypothesis Strategies](./hypothesis_strategies.html) — generate synthetic data with Hypothesis

## See also

- [Supported DataFrame Libraries](../supported_libraries.html) — other backends
- [Checks](../checks.html) — general `Check` behaviour
- [Lazy Validation](../lazy_validation.html) — `lazy=True` and `SchemaErrors`
- [Configuration](../configuration.html) — `ValidationDepth` and environment variables
