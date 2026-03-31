---
file_format: mystnb
---

(xarray-checks-parsers)=

# Checks and Parsers

## Checks

The same {class}`~pandera.api.checks.Check` class used for pandas and polars
works with xarray. The xarray backends dispatch on
{class}`~xarray.DataArray` and {class}`~xarray.Dataset`.

```{code-cell} python
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

```{code-cell} python
da_numeric = xr.DataArray(np.array([10, 20, 30]), dims="x")

pa.DataArraySchema(checks=pa.Check.greater_than(0)).validate(da_numeric)
pa.DataArraySchema(checks=pa.Check.less_than(100)).validate(da_numeric)
pa.DataArraySchema(checks=pa.Check.isin([10, 20, 30])).validate(da_numeric)
pa.DataArraySchema(checks=pa.Check.notin([0, -1])).validate(da_numeric)
pa.DataArraySchema(checks=pa.Check.in_range(0, 100)).validate(da_numeric)
```

String checks also work on string-typed arrays:

```{code-cell} python
da_str = xr.DataArray(np.array(["FOO_1", "BAR_2"]), dims="x")

pa.DataArraySchema(checks=pa.Check.str_matches(r"^[A-Z]+_\d+$")).validate(da_str)
pa.DataArraySchema(checks=pa.Check.str_contains("_")).validate(da_str)
pa.DataArraySchema(checks=pa.Check.str_length(min_value=3, max_value=10)).validate(da_str)
```

### Xarray-specific checks

These checks are specific to xarray's structural model:

```{code-cell} python
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

```{code-cell} python
schema = pa.DataArraySchema(
    checks=pa.Check(lambda x: x > 0, element_wise=True),
)

da_positive = xr.DataArray([1.0, 2.0, 3.0], dims="x")
schema.validate(da_positive)
```

### Custom checks

Write any callable that accepts a `DataArray` (or `Dataset`) and returns a
boolean or a boolean `DataArray`:

```{code-cell} python
def is_normalized(da):
    return float(da.min()) >= 0 and float(da.max()) <= 1

schema = pa.DataArraySchema(checks=pa.Check(is_normalized))

da_norm = xr.DataArray(np.linspace(0, 1, 10), dims="x")
schema.validate(da_norm)
```

```{code-cell} python
da_bad = xr.DataArray(np.linspace(-1, 2, 10), dims="x")

try:
    schema.validate(da_bad)
except pa.errors.SchemaError as exc:
    print(exc)
```

## Parsers

{class}`~pandera.api.parsers.Parser` objects transform the data **before**
checks run. This is useful for filling missing values, renaming, or other
pre-processing:

```{code-cell} python
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

By default, `validate()` raises {class}`~pandera.errors.SchemaError` at the
first failure. Pass `lazy=True` to collect all errors and raise a single
{class}`~pandera.errors.SchemaErrors`:

```{code-cell} python
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

The {class}`~pandera.errors.SchemaErrors` exception collects every failure into
a structured report. Each entry in the report corresponds to one failed
validation check and contains these fields:

- **`schema_context`** — the type of schema that raised the error (e.g.
  `DataArraySchema`, `DatasetSchema`). For per-variable errors inside a
  `DatasetSchema`, this shows the inner `DataArraySchema` that validated the
  individual variable.
- **`column`** — the name of the data variable, coordinate, or array that
  failed. For a standalone `DataArraySchema`, this is the array's `name`
  attribute. For a `DatasetSchema`, it is the `data_var` or `coord` key.
- **`check`** — the name or description of the check that failed. Structural
  checks use names like `"dims"`, `"dtype"`, `"name"`, `"attrs"`,
  `"strict_coords"`. Data-level checks show the check function description
  (e.g. `"in_range(0, 100)"` or `"<lambda>"`).
- **`check_number`** — the index of the check in the schema's check list
  (starting from 0). `None` for structural checks.
- **`failure_case`** — a summary of what went wrong. For structural checks this
  is the actual value that didn't match (e.g. the actual dims tuple or dtype
  string). For data-level checks this contains the failing values.
- **`index`** — for element-wise data checks, the N-dimensional coordinate
  context of each failing element (flattened into a dict of dimension → value
  pairs). `None` for aggregate and structural checks.

The report is formatted as a JSON list. You can access the underlying data
programmatically via `exc.schema_errors` (a list of
{class}`~pandera.errors.SchemaError` objects) or `exc.message` (the formatted
string).

This works the same way as {ref}`lazy-validation` for pandas, but adapted for
xarray's N-dimensional data model where indices are multi-dimensional coordinate
labels rather than flat row indices.

## See also

- {ref}`xarray-data-array-schema` — {class}`~pandera.api.xarray.container.DataArraySchema` details
- {ref}`xarray-dataset-schema` — {class}`~pandera.api.xarray.container.DatasetSchema` details
- {ref}`xarray-data-models` — class-based {class}`~pandera.api.xarray.model.DataArrayModel` / {class}`~pandera.api.xarray.model.DatasetModel`
- {ref}`xarray-decorators` — `check_input`, `check_output`, `check_io`, and `check_types`
- {ref}`xarray-configuration` — {class}`~pandera.config.ValidationDepth`,
  {class}`~pandera.config.ValidationScope`, Dask, and environment variables
- {ref}`api-xarray` — full API reference for all xarray classes
- {ref}`api-core` — {class}`~pandera.config.PanderaConfig`,
  {class}`~pandera.config.ValidationDepth`,
  {class}`~pandera.config.ValidationScope`
- {ref}`checks` — general {class}`~pandera.api.checks.Check` behaviour (pandas-oriented)
- {ref}`lazy-validation` — detailed lazy validation docs
