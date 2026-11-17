# CF Convention Checks

The [Climate and Forecast (CF) conventions](https://cfconventions.org/) are a
widely used metadata standard for geoscience data.  Pandera provides built-in
checks that validate CF metadata attributes on xarray objects.

There are two categories:

| Category | Requires `cf_xarray`? | What it checks |
|----------|----------------------|----------------|
| **Lightweight** (`cf_standard_name`, `cf_units`, `cf_has_cell_methods`) | No | Inspects `.attrs` directly |
| **Accessor-based** (`cf_has_standard_names`) | Yes | Uses the `cf_xarray` accessor to resolve standard names |

## Lightweight CF checks

These checks inspect `.attrs` directly and do not need any extra dependencies.

### `Check.cf_standard_name()`

Require that `.attrs["standard_name"]` equals a specific value:

```python
import numpy as np
import xarray as xr
import pandera.xarray as pa

da = xr.DataArray(
    np.ones((3, 4)),
    dims=("x", "y"),
    attrs={"standard_name": "air_temperature", "units": "K"},
)

pa.DataArraySchema(
    checks=pa.Check.cf_standard_name("air_temperature"),
).validate(da)
```

When the standard name doesn't match:

```python
try:
    pa.DataArraySchema(
        checks=pa.Check.cf_standard_name("sea_surface_temperature"),
    ).validate(da)
except pa.errors.SchemaError as exc:
    print(exc)
```

### `Check.cf_units()`

Require that `.attrs["units"]` equals a specific value:

```python
pa.DataArraySchema(
    checks=pa.Check.cf_units("K"),
).validate(da)
```

```python
try:
    pa.DataArraySchema(
        checks=pa.Check.cf_units("degC"),
    ).validate(da)
except pa.errors.SchemaError as exc:
    print(exc)
```

### `Check.cf_has_cell_methods()`

Require that `.attrs["cell_methods"]` equals a specific string:

```python
da_cell = xr.DataArray(
    np.ones(12),
    dims="time",
    attrs={"cell_methods": "time: mean"},
)

pa.DataArraySchema(
    checks=pa.Check.cf_has_cell_methods("time: mean"),
).validate(da_cell)
```

### Combining CF checks

Multiple CF checks can be combined in a single schema:

```python
schema = pa.DataArraySchema(
    dims=("time", "lat", "lon"),
    checks=[
        pa.Check.cf_standard_name("air_temperature"),
        pa.Check.cf_units("K"),
    ],
)

da_3d = xr.DataArray(
    np.ones((12, 5, 10)),
    dims=("time", "lat", "lon"),
    attrs={
        "standard_name": "air_temperature",
        "units": "K",
    },
)
schema.validate(da_3d)
```

## `cf_xarray` accessor check

### `Check.cf_has_standard_names()`

This check requires the [`cf_xarray`](https://cf-xarray.readthedocs.io/)
package and verifies that each listed standard name is resolvable via the
`.cf` accessor:

```python
# pip install cf_xarray
import cf_xarray  # noqa: F401

ds = xr.Dataset({
    "T": (("x",), np.ones(3), {"standard_name": "air_temperature"}),
    "P": (("x",), np.ones(3), {"standard_name": "air_pressure"}),
})

schema = pa.DatasetSchema(
    checks=pa.Check.cf_has_standard_names(
        ("air_temperature", "air_pressure"),
    ),
)
schema.validate(ds)
```

When a standard name cannot be resolved, the check fails:

```python
try:
    pa.DatasetSchema(
        checks=pa.Check.cf_has_standard_names(
            ("air_temperature", "sea_surface_height"),
        ),
    ).validate(ds)
except pa.errors.SchemaError as exc:
    print(exc)
```

!!! note

    If `cf_xarray` is not installed, `Check.cf_has_standard_names()` raises an
    `ImportError` with installation instructions. The lightweight checks
    (`cf_standard_name`, `cf_units`, `cf_has_cell_methods`) never require
    `cf_xarray`.

## Dataset-level CF validation

CF checks work on both `DataArray` and `Dataset` objects.  On a `Dataset`,
the check receives the entire dataset:

```python
ds_cf = xr.Dataset(
    {"temp": (("x",), np.ones(3))},
    attrs={"standard_name": "air_temperature", "units": "K"},
)

pa.DatasetSchema(
    data_vars={"temp": pa.DataVar(dims=("x",))},
    checks=[
        pa.Check.cf_standard_name("air_temperature"),
        pa.Check.cf_units("K"),
    ],
).validate(ds_cf)
```

## Per-variable CF checks

To validate CF attributes on individual data variables, attach checks to the
[`DataVar`](../reference/generated/pandera.api.xarray.components.DataVar.html):

```python
ds_vars = xr.Dataset({
    "temperature": xr.DataArray(
        np.ones((3, 4)),
        dims=("x", "y"),
        attrs={"standard_name": "air_temperature", "units": "K"},
    ),
    "pressure": xr.DataArray(
        np.ones((3, 4)),
        dims=("x", "y"),
        attrs={"standard_name": "air_pressure", "units": "Pa"},
    ),
})

schema = pa.DatasetSchema(
    data_vars={
        "temperature": pa.DataVar(
            dims=("x", "y"),
            checks=[
                pa.Check.cf_standard_name("air_temperature"),
                pa.Check.cf_units("K"),
            ],
        ),
        "pressure": pa.DataVar(
            dims=("x", "y"),
            checks=[
                pa.Check.cf_standard_name("air_pressure"),
                pa.Check.cf_units("Pa"),
            ],
        ),
    },
)
schema.validate(ds_vars)
```

## Combining CF checks with attrs validation

CF checks complement the `attrs` parameter. Use `attrs` for structural
attribute requirements and CF checks for semantic validation:

```python
schema = pa.DataArraySchema(
    attrs={
        "standard_name": "air_temperature",
        "units": "^(K|degC|degF)$",
    },
    checks=[
        pa.Check.cf_standard_name("air_temperature"),
    ],
)

da_combined = xr.DataArray(
    np.ones(3),
    dims="x",
    attrs={"standard_name": "air_temperature", "units": "K"},
)
schema.validate(da_combined)
```

## See also

- [Checks and Parsers](./checks_and_parsers.html) — general check and parser usage
- [DataArraySchema](./data_array_schema.html) — attribute validation modes
- [DatasetSchema](./dataset_schema.html) — dataset attributes and `DataVar` checks
- [Encoding](./encoding.html) — encoding validation (related metadata)
- [CF Conventions](https://cfconventions.org/) — the CF metadata standard
- [cf_xarray documentation](https://cf-xarray.readthedocs.io/) — the
  `cf_xarray` package
