# Checks and Parsers

## Checks

The same [`Check`](../reference/generated/pandera.api.checks.Check.html) class used for pandas and polars
works with xarray. The xarray backends dispatch on
`xarray.DataArray` and `xarray.Dataset`.

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

da = xr.DataArray(np.linspace(0, 50, 12), dims="x")
schema.validate(da)
```

### Built-in checks

All standard built-in checks work on xarray objects. On a `DataArray` they
operate on the array values; on a `Dataset` they operate across variables.

```python
da_numeric = xr.DataArray(np.array([10, 20, 30]), dims="x")

pa.DataArraySchema(checks=pa.Check.greater_than(0)).validate(da_numeric)
pa.DataArraySchema(checks=pa.Check.less_than(100)).validate(da_numeric)
pa.DataArraySchema(checks=pa.Check.isin([10, 20, 30])).validate(da_numeric)
pa.DataArraySchema(checks=pa.Check.notin([0, -1])).validate(da_numeric)
pa.DataArraySchema(checks=pa.Check.in_range(0, 100)).validate(da_numeric)
```

String checks also work on string-typed arrays:

```python
da_str = xr.DataArray(np.array(["FOO_1", "BAR_2"]), dims="x")

pa.DataArraySchema(checks=pa.Check.str_matches(r"^[A-Z]+_\d+$")).validate(da_str)
pa.DataArraySchema(checks=pa.Check.str_contains("_")).validate(da_str)
pa.DataArraySchema(checks=pa.Check.str_length(min_value=3, max_value=10)).validate(da_str)
```

### Xarray-specific checks

These checks are specific to xarray's structural model:

```python
da_3d = xr.DataArray(
    np.ones((12, 5, 10)),
    dims=("time", "lat", "lon"),
    coords={
        "time": np.arange(12, dtype=np.float64),
        "lat": np.linspace(-90, 90, 5),
        "lon": np.linspace(-180, 180, 10),
    },
    attrs={"units": "K"},
)

pa.DataArraySchema(checks=pa.Check.has_dims(("time", "lat", "lon"))).validate(da_3d)
pa.DataArraySchema(checks=pa.Check.has_coords(("time", "lat"))).validate(da_3d)
pa.DataArraySchema(checks=pa.Check.has_attrs({"units": "K"})).validate(da_3d)
pa.DataArraySchema(checks=pa.Check.ndim(3)).validate(da_3d)
pa.DataArraySchema(checks=pa.Check.dim_size("time", 12)).validate(da_3d)
pa.DataArraySchema(checks=pa.Check.is_monotonic("time")).validate(da_3d)
pa.DataArraySchema(checks=pa.Check.no_duplicates_in_coord("time")).validate(da_3d)
```

#### Encoding check

```python
da_enc = xr.DataArray(np.ones(3), dims="x")
da_enc.encoding = {"_FillValue": -999.0, "dtype": "float32"}

pa.DataArraySchema(
    checks=pa.Check.has_encoding({"_FillValue": -999.0}),
).validate(da_enc)
```

See [Encoding](./encoding.html) for the richer schema-level `encoding=` parameter.

#### CF convention checks

```python
da_cf = xr.DataArray(
    np.ones(3), dims="x",
    attrs={"standard_name": "air_temperature", "units": "K"},
)

pa.DataArraySchema(
    checks=[
        pa.Check.cf_standard_name("air_temperature"),
        pa.Check.cf_units("K"),
    ],
).validate(da_cf)
```

See [CF Convention Checks](./cf_conventions.html) for all available CF checks.

!!! note

    Structural rules (`dims`, `coords`, `sizes`, `attrs`, …) are best expressed
    as schema keyword arguments — they are validated first and produce clearer
    error messages. The `Check.has_*` helpers are useful for:

    - **Dataset-level** `checks=[...]` where you need structural assertions
      across the whole container.
    - **Ad hoc** validation where you don't want a full schema.
    - **Value-level** structural checks like `is_monotonic` and
      `no_duplicates_in_coord` that have no schema-kwarg equivalent.

### Element-wise checks

`element_wise=True` is available but less common for N-D arrays. The
check function receives individual scalar values:

```python
schema = pa.DataArraySchema(
    checks=pa.Check(lambda x: x > 0, element_wise=True),
)

da_positive = xr.DataArray([1.0, 2.0, 3.0], dims="x")
schema.validate(da_positive)
```

### Custom checks

Write any callable that accepts a `DataArray` (or `Dataset`) and returns a
boolean or a boolean `DataArray`:

```python
def is_normalized(da):
    return float(da.min()) >= 0 and float(da.max()) <= 1

schema = pa.DataArraySchema(checks=pa.Check(is_normalized))

da_norm = xr.DataArray(np.linspace(0, 1, 10), dims="x")
schema.validate(da_norm)
```

```python
da_bad = xr.DataArray(np.linspace(-1, 2, 10), dims="x")

try:
    schema.validate(da_bad)
except pa.errors.SchemaError as exc:
    print(exc)
```

## Parsers

[`Parser`](../reference/generated/pandera.api.parsers.Parser.html) objects transform the data **before**
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

da_messy = xr.DataArray([1.0, np.nan, 3.0], dims="x", name="raw")
validated = schema.validate(da_messy)
print(f"name: {validated.name}, values: {validated.values}")
```

## Lazy validation

By default, `validate()` raises [`SchemaError`](../reference/generated/pandera.errors.SchemaError.html) at the
first failure. Pass `lazy=True` to collect all errors and raise a single
[`SchemaErrors`](../reference/generated/pandera.errors.SchemaErrors.html):

```python
schema = pa.DataArraySchema(
    dtype=np.float64,
    dims=("x",),
    name="values",
    checks=pa.Check(lambda da: bool((da > 0).all())),
)

da_multi_err = xr.DataArray([-1, 2, 3], dims="x", name="wrong_name")

try:
    schema.validate(da_multi_err, lazy=True)
except pa.errors.SchemaErrors as exc:
    print(exc)
```

### Understanding the lazy validation report

[`SchemaErrors`](../reference/generated/pandera.errors.SchemaErrors.html) attaches a nested JSON summary in
`exc.message` (top-level **`SCHEMA`** vs **`DATA`**, then reason codes, then
per-failure dicts with `schema`, `column`, `check`, and `error`). Use
`exc.schema_errors` for the full list of [`SchemaError`](../reference/generated/pandera.errors.SchemaError.html)
objects; element-wise check failures often expose a
`xarray.DataArray` mask on each error’s `failure_cases`.

See [Error Reporting](./error_reporting.html) for a full walkthrough, schema vs data
semantics, and how to interpret `failure_cases` next to [Lazy Validation](../lazy_validation.html)
and [Error Report](../error_report.html).

## See also

- [DataArraySchema](./data_array_schema.html) — [`DataArraySchema`](../reference/generated/pandera.api.xarray.container.DataArraySchema.html) details
- [DatasetSchema](./dataset_schema.html) — [`DatasetSchema`](../reference/generated/pandera.api.xarray.container.DatasetSchema.html) details
- [Data Models](./data_models.html) — class-based [`DataArrayModel`](../reference/generated/pandera.api.xarray.model.DataArrayModel.html) / [`DatasetModel`](../reference/generated/pandera.api.xarray.model.DatasetModel.html)
- [Encoding](./encoding.html) — encoding validation (netCDF/Zarr metadata)
- [Error Reporting](./error_reporting.html) — lazy validation reports and failure cases
- [CF Convention Checks](./cf_conventions.html) — CF convention checks
- [Duck Arrays](./duck_arrays.html) — Dask integration and validation depth
- [Decorators](./decorators.html) — `check_input`, `check_output`, `check_io`, and `check_types`
- [Configuration](./configuration.html) — [`ValidationDepth`](../reference/generated/pandera.config.ValidationDepth.html),
  [`ValidationScope`](../reference/generated/pandera.config.ValidationScope.html), Dask, and environment variables
- [Xarray API reference](../reference/xarray.html) — full API reference for all xarray classes
- [Core API reference](../reference/core.html) — [`PanderaConfig`](../reference/generated/pandera.config.PanderaConfig.html),
  [`ValidationDepth`](../reference/generated/pandera.config.ValidationDepth.html),
  [`ValidationScope`](../reference/generated/pandera.config.ValidationScope.html)
- [Checks](../checks.html) — general [`Check`](../reference/generated/pandera.api.checks.Check.html) behaviour (pandas-oriented)
- [Lazy Validation](../lazy_validation.html) — detailed lazy validation docs
