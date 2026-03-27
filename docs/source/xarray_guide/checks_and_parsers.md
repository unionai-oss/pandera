---
file_format: myst
---

(xarray-checks-parsers)=

# Checks and Parsers

## Checks

The same {class}`~pandera.api.checks.Check` class used for pandas and polars
works with xarray. The xarray backends dispatch on
{class}`~xarray.DataArray` and {class}`~xarray.Dataset`.

```python
import numpy as np
import xarray as xr
import pandera.xarray as pa

schema = pa.DataArraySchema(
    dtype=np.float64,
    checks=[
        pa.Check(lambda da: float(da.min()) >= 0),
        pa.Check(lambda da: float(da.max()) <= 100),
    ],
)
```

### Built-in checks

All standard built-in checks work on xarray objects. On a `DataArray` they
operate on the array values; on a `Dataset` they operate across variables.

```python
pa.Check.greater_than(0)
pa.Check.less_than(100)
pa.Check.in_range(0, 1)
pa.Check.isin(["a", "b", "c"])
pa.Check.equal_to(42)
pa.Check.notin([0, -1])
```

String checks also work on string-typed arrays:

```python
pa.Check.str_matches(r"^[A-Z]+$")
pa.Check.str_contains("foo")
pa.Check.str_startswith("prefix")
pa.Check.str_endswith("suffix")
pa.Check.str_length(min_value=1, max_value=10)
```

### Xarray-specific checks

These checks are specific to xarray's structural model:

```python
pa.Check.has_dims(("time", "lat", "lon"))
pa.Check.has_coords(("time", "lat"))
pa.Check.has_attrs({"units": "K"})
pa.Check.ndim(3)
pa.Check.dim_size("time", 12)
pa.Check.is_monotonic("time")
pa.Check.is_monotonic("time", decreasing=True)
pa.Check.no_duplicates_in_coord("time")
```

:::{note}
Structural rules (`dims`, `coords`, `sizes`, `attrs`, …) are best expressed
as schema keyword arguments — they are validated first and produce clearer
error messages. The `Check.has_*` helpers are useful for:

- **Dataset-level** `checks=[...]` where you need structural assertions
  across the whole container.
- **Ad hoc** validation where you don't want a full schema.
- **Value-level** structural checks like `is_monotonic` and
  `no_duplicates_in_coord` that have no schema-kwarg equivalent.
:::

### Element-wise checks

`element_wise=True` is available but less common for N-D arrays. The
check function receives individual scalar values:

```python
pa.Check(lambda x: x > 0, element_wise=True)
```

### Custom checks

Write any callable that accepts a `DataArray` (or `Dataset`) and returns a
boolean or a boolean `DataArray`:

```python
def is_normalized(da):
    return float(da.min()) >= 0 and float(da.max()) <= 1

schema = pa.DataArraySchema(checks=pa.Check(is_normalized))
```

## Parsers

{class}`~pandera.api.parsers.Parser` objects transform the data **before**
checks run. This is useful for filling missing values, renaming, or other
pre-processing:

```python
schema = pa.DataArraySchema(
    parsers=[
        pa.Parser(lambda da: da.fillna(0)),
        pa.Parser(lambda da: da.rename("cleaned")),
    ],
    checks=pa.Check(lambda da: float(da.min()) >= 0),
)
```

## Lazy validation

By default, `validate()` raises {class}`~pandera.errors.SchemaError` at the
first failure. Pass `lazy=True` to collect all errors and raise a single
{class}`~pandera.errors.SchemaErrors`:

```python
schema = pa.DataArraySchema(
    dtype=np.float64,
    dims=("x",),
    name="values",
    checks=pa.Check(lambda da: (da > 0).all()),
)

da = xr.DataArray([-1, 2, 3], dims="x", name="wrong_name")

try:
    schema.validate(da, lazy=True)
except pa.errors.SchemaErrors as exc:
    print(exc.failure_cases)
```

This works the same way as {ref}`lazy-validation` for pandas.

## See also

- {ref}`xarray-data-array-schema` — `DataArraySchema` details
- {ref}`xarray-dataset-schema` — `DatasetSchema` details
- {ref}`xarray-data-models` — class-based models
- {ref}`xarray-decorators` — `check_input`, `check_output`, `check_io`, and `check_types`
- {ref}`xarray-configuration` — validation depth, Dask, and environment variables
- {ref}`checks` — general `Check` behaviour (pandas-oriented)
- {ref}`lazy-validation` — detailed lazy validation docs
