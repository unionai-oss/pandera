---
file_format: mystnb
---

(xarray-guide)=

# Xarray Data Validation

[xarray](https://docs.xarray.dev/) provides labelled multi-dimensional arrays
{class}`~xarray.DataArray`, collections of aligned arrays
{class}`~xarray.Dataset`, and collections of datasets with {class}`~xarray.DataTree`.

Pandera validates them with the same patterns as the other dataframe backends:
schema objects, optional {class}`~pandera.api.checks.Check`
instances, and global {ref}`configuration <configuration>`.

## Installation

```bash
pip install 'pandera[xarray]'
```

## Quick start

```{code-cell} python
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

```{code-cell} python
from pandera.typing.xarray import Coordinate

class Surface(pa.DatasetModel):
    temperature: np.float64 = pa.Field(dims=("x", "y"))
    pressure: np.float64 = pa.Field(dims=("x", "y"))
    x: Coordinate[np.float64]

Surface.validate(ds)
```

## Guide contents

```{toctree}
:maxdepth: 2
:hidden:

data_array_schema
dataset_schema
data_tree
data_models
checks_and_parsers
decorators
configuration
duck_arrays
encoding
error_reporting
cf_conventions
schema_inference
io_serialization
hypothesis_strategies
```

- {ref}`xarray-data-array-schema` — validating a single {class}`~xarray.DataArray`
- {ref}`xarray-dataset-schema` — validating a {class}`~xarray.Dataset` with `DataVar` and `Coordinate`
- {ref}`xarray-data-tree` — validating a {class}`~xarray.DataTree` hierarchy
- {ref}`xarray-data-models` — class-based `DataArrayModel`, `DatasetModel`, and `DataTreeModel`
- {ref}`xarray-checks-parsers` — checks, parsers, and lazy validation
- {ref}`xarray-decorators` — `check_input`, `check_output`, `check_io`, and `check_types`
- {ref}`xarray-configuration` — validation depth, Dask, and environment variables
- {ref}`xarray-duck-arrays` — `chunked`, `array_type`, validation depth, and lazy data checks
- {ref}`xarray-encoding` — validate `.encoding` dicts on DataArrays, DataVars, and Datasets
- {ref}`xarray-error-reporting` — `SchemaError` / `SchemaErrors`, lazy validation, and failure cases
- {ref}`xarray-cf-conventions` — CF standard name, units, and `cf_xarray` checks
- {ref}`xarray-schema-inference` — automatically infer schemas from data
- {ref}`xarray-io-serialization` — save and load schemas as YAML or JSON
- {ref}`xarray-hypothesis-strategies` — generate synthetic data with Hypothesis

## See also

- {ref}`supported-dataframe-libraries` — other backends
- {ref}`checks` — general `Check` behaviour
- {ref}`lazy-validation` — `lazy=True` and `SchemaErrors`
- {ref}`configuration` — `ValidationDepth` and environment variables
