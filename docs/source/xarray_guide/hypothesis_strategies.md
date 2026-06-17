---
file_format: mystnb
---

(xarray-hypothesis-strategies)=

# Hypothesis Data Strategies

Generate synthetic data conforming to a schema using the
[hypothesis](https://hypothesis.readthedocs.io/) library. Install:

```bash
pip install 'pandera[strategies]'
```

## DataArray strategies

```{code-cell} python
import numpy as np
import pandera.xarray as pa
from pandera.strategies.xarray_strategies import (
    data_array_strategy,
    data_array_schema_strategy,
    dataset_strategy,
    dataset_schema_strategy,
)

schema = pa.DataArraySchema(
    dtype="float64",
    dims=("x", "y"),
    sizes={"x": 3, "y": 4},
    name="temp",
)
```

Use `data_array_schema_strategy` inside a Hypothesis test:

```python
from hypothesis import given, settings

@given(data_array_schema_strategy(schema))
@settings(max_examples=10)
def test_generated_data(da):
    assert da.dims == ("x", "y")
    assert da.sizes["x"] == 3
    schema.validate(da)

test_generated_data()
```

## Dataset strategies

```{code-cell} python
ds_schema = pa.DatasetSchema(
    data_vars={
        "temperature": pa.DataVar(dtype="float64", dims=("x", "y")),
        "pressure": pa.DataVar(dtype="float64", dims=("x", "y")),
    },
    sizes={"x": 3, "y": 4},
)
```

```python
@given(dataset_schema_strategy(ds_schema))
@settings(max_examples=10)
def test_generated_dataset(ds):
    assert "temperature" in ds.data_vars
    assert "pressure" in ds.data_vars
    ds_schema.validate(ds)

test_generated_dataset()
```

## Low-level strategies

For more control, use the building blocks directly:

```python
from pandera.strategies.xarray_strategies import (
    xarray_dtype_strategy,
    data_array_strategy,
    dataset_strategy,
)

# Generate float64 scalars
xarray_dtype_strategy("float64")

# Generate DataArrays with explicit settings
data_array_strategy(
    dtype="int32",
    dims=("time", "lat"),
    sizes={"time": 10, "lat": 180},
    coords={"time": {"dtype": "float64"}},
    name="obs",
)

# Generate Datasets
dataset_strategy(
    data_vars={
        "a": {"dtype": "float64", "dims": ("x",)},
        "b": {"dtype": "int32", "dims": ("x", "y")},
    },
    sizes={"x": 5, "y": 10},
)
```

## Limitations

The xarray strategies currently generate data based on **structural**
properties only — dtype, dims, sizes, coords, name, and nullable. They do
**not** yet incorporate {class}`~pandera.api.checks.Check` constraints when
synthesizing values. For example, a schema with `Check.in_range(0, 1)` will
produce arrays with arbitrary floats rather than values restricted to
`[0, 1]`.

This means generated data may not pass `schema.validate()` when the schema
includes value-level checks. For now you can work around this by either:

- Adding a `hypothesis.assume(...)` filter in your test, or
- Using `hypothesis.strategies.floats(min_value=..., max_value=...)` with the
  low-level `data_array_strategy` to manually constrain the element domain.

Check-aware data generation (mirroring the pandas strategies integration) is
planned for a future release.

## Dask Integration

Chunked (Dask-backed) xarray objects are validated with structural checks
by default (dtype, dims, sizes, coords) without triggering
{meth}`~dask.array.Array.compute`. See {ref}`xarray-configuration` for
how to enable data-level checks on lazy arrays.
