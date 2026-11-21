# DataArray Schemas

[`DataArraySchema`](../reference/generated/pandera.api.xarray.container.DataArraySchema.html) validates a single
`xarray.DataArray`. It is the xarray counterpart of
[`SeriesSchema`](../reference/generated/pandera.api.pandas.array.SeriesSchema.html) — but for arbitrary-rank
labelled arrays rather than 1-D series.

You can also express the same constraints with the declarative
[DataArrayModel](./data_models.html).

## Basic usage

```python
import numpy as np
import xarray as xr
import pandera.xarray as pa

schema = pa.DataArraySchema(
    dtype=np.float64,
    dims=("x", "y"),
    name="temperature",
)

da = xr.DataArray(
    np.random.rand(3, 4),
    dims=("x", "y"),
    name="temperature",
)
schema.validate(da)
```

## Dtype validation

The `dtype` is resolved through NumPy's type hierarchy. You can pass a
Python type, a NumPy dtype, or a string alias:

```python
da_float32 = xr.DataArray(np.zeros(3, dtype=np.float32), dims="x")

pa.DataArraySchema(dtype=float).validate(da)
pa.DataArraySchema(dtype=np.float32).validate(da_float32)
pa.DataArraySchema(dtype="float32").validate(da_float32)
```

If `dtype` is `None`, any dtype is accepted.

## Dimension validation

`dims` enforces dimension names **in order**. `None` entries act as
wildcards that match any name:

```python
pa.DataArraySchema(dims=("x", "y")).validate(da)
pa.DataArraySchema(dims=("x", None)).validate(da)
```

The tuple length also constrains the rank (ndim).

```python
try:
    pa.DataArraySchema(dims=("x", "y", "z")).validate(da)
except pa.errors.SchemaError as exc:
    print(exc)
```

## Sizes and shape

`sizes` is the idiomatic xarray way to constrain dimension lengths.
`shape` does the same thing positionally. They are mutually exclusive.

```python
da_sized = xr.DataArray(
    np.zeros((12, 180, 360)),
    dims=("time", "lat", "lon"),
)

pa.DataArraySchema(
    dims=("time", "lat", "lon"),
    sizes={"lat": 180, "lon": 360},
).validate(da_sized)

pa.DataArraySchema(
    dims=("time", "lat", "lon"),
    shape=(None, 180, 360),
).validate(da_sized)
```

## Coordinate validation

Pass a `dict[str, Coordinate]` to validate coordinate arrays, or a
`list[str]` as shorthand for "these coordinates must exist":

```python
da_with_coords = xr.DataArray(
    np.random.rand(3, 4),
    dims=("x", "y"),
    coords={
        "x": np.arange(3, dtype=np.float64),
        "y": np.arange(4, dtype=np.float64),
        "label": ("x", ["a", "b", "c"]),
    },
)

schema = pa.DataArraySchema(
    dims=("x", "y"),
    coords={
        "x": pa.Coordinate(dtype=np.float64, dimension=True),
        "y": pa.Coordinate(dtype=np.float64, dimension=True),
        "label": pa.Coordinate(dimension=False),
    },
)
schema.validate(da_with_coords)
```

[`Coordinate`](../reference/generated/pandera.api.xarray.components.Coordinate.html) is documented in
detail under [DatasetSchema](./dataset_schema.html).

### Strict coordinates

With `strict_coords=True`, the schema fails if the DataArray has
coordinates not listed in `coords`:

```python
strict_schema = pa.DataArraySchema(
    coords={"x": pa.Coordinate()},
    strict_coords=True,
)

da_x_only = xr.DataArray(
    np.ones(3),
    dims="x",
    coords={"x": np.arange(3, dtype=np.float64)},
)
strict_schema.validate(da_x_only)
```

```python
try:
    strict_schema.validate(da_with_coords)
except pa.errors.SchemaError as exc:
    print(exc)
```

## Attribute validation

`attrs` validates the DataArray's `.attrs` dict. Each value in the schema's
`attrs` dict determines how the corresponding attribute is checked:

- **Literal values** — matched by equality (`==`).
- **Regex patterns** — strings that start with `^` are treated as regular
  expressions and matched against `str(actual_value)` via `re.fullmatch`.
- **Callable predicates** — any callable `(value) -> bool` is invoked with the
  actual attribute value; validation passes when the function returns `True`.
- **Pydantic model** — pass a `pydantic.BaseModel` **class** to validate
  the full attrs dict using pydantic's type system.

### Equality matching

```python
da_attrs = xr.DataArray(
    np.ones(3), dims="x",
    attrs={"units": "K", "standard_name": "air_temperature"},
)

pa.DataArraySchema(
    attrs={"units": "K", "standard_name": "air_temperature"},
).validate(da_attrs)
```

### Regex matching

Use a regex pattern (starting with `^`) to validate an attribute against a
set of acceptable values:

```python
schema = pa.DataArraySchema(
    attrs={"units": "^(K|degC|degF)$"},
)

da_units = xr.DataArray(
    np.ones(3), dims="x",
    attrs={"units": "K"},
)
schema.validate(da_units)
```

```python
da_bad_units = xr.DataArray(
    np.ones(3), dims="x",
    attrs={"units": "meters"},
)

try:
    schema.validate(da_bad_units)
except pa.errors.SchemaError as exc:
    print(exc)
```

### Callable predicates

Pass a function that receives the attribute value and returns a boolean:

```python
schema = pa.DataArraySchema(
    attrs={
        "version": lambda v: isinstance(v, int) and v >= 2,
    },
)

da_v3 = xr.DataArray(
    np.ones(3), dims="x",
    attrs={"version": 3},
)
schema.validate(da_v3)
```

```python
da_v1 = xr.DataArray(
    np.ones(3), dims="x",
    attrs={"version": 1},
)

try:
    schema.validate(da_v1)
except pa.errors.SchemaError as exc:
    print(exc)
```

### Pydantic model

For complex attribute schemas you can pass a `pydantic.BaseModel`
**class** instead of a dict. Pandera delegates validation to pydantic and
converts every pydantic error into a pandera `SchemaError`, so error
collection during lazy validation works seamlessly:

```python
from pydantic import BaseModel, Field as PydanticField

class ArrayAttrs(BaseModel):
    units: str
    standard_name: str
    version: int = PydanticField(ge=2)
```

```python
schema = pa.DataArraySchema(attrs=ArrayAttrs)

da_ok = xr.DataArray(
    np.ones(3), dims="x",
    attrs={"units": "K", "standard_name": "air_temperature", "version": 3},
)
schema.validate(da_ok)
```

When validation fails, the error messages surface the pydantic error details:

```python
da_bad = xr.DataArray(
    np.ones(3), dims="x",
    attrs={"units": "K", "version": 1},  # version < 2, standard_name missing
)

try:
    schema.validate(da_bad, lazy=True)
except pa.errors.SchemaErrors as exc:
    print(exc)
```

All four modes also work on
[`DatasetSchema`](../reference/generated/pandera.api.xarray.container.DatasetSchema.html) — see
[DatasetSchema](./dataset_schema.html).

### Strict attributes

With `strict_attrs=True`, extra attributes cause a validation error.
When `attrs` is a pydantic model class, the set of allowed keys is derived
from the model's fields.

```python
da_extra = xr.DataArray(
    np.ones(3), dims="x",
    attrs={"units": "K", "extra_key": 42},
)

try:
    pa.DataArraySchema(
        attrs={"units": "K"}, strict_attrs=True
    ).validate(da_extra)
except pa.errors.SchemaError as exc:
    print(exc)
```

## Name validation

```python
named_da = xr.DataArray(np.ones(3), dims="x", name="temperature")
pa.DataArraySchema(name="temperature").validate(named_da)
```

The DataArray's `.name` must match exactly.

```python
try:
    pa.DataArraySchema(name="pressure").validate(named_da)
except pa.errors.SchemaError as exc:
    print(exc)
```

## Null values

By default `nullable=False` — any NaN or null value raises a
[`SchemaError`](../reference/generated/pandera.errors.SchemaError.html). Set `nullable=True` to allow them:

```python
da_with_nan = xr.DataArray([1.0, np.nan, 3.0], dims="x")

pa.DataArraySchema(dtype=float, nullable=True).validate(da_with_nan)
```

```python
try:
    pa.DataArraySchema(dtype=float, nullable=False).validate(da_with_nan)
except pa.errors.SchemaError as exc:
    print(exc)
```

## Coercing dtypes

When `coerce=True`, the DataArray is cast to `dtype` before validation:

```python
schema = pa.DataArraySchema(dtype=np.float32, coerce=True)
da_int = xr.DataArray(np.array([1, 2, 3]), dims="x")
validated = schema.validate(da_int)
print(f"original: {da_int.dtype} -> coerced: {validated.dtype}")
```

## Encoding validation

The `encoding` parameter validates the DataArray's `.encoding` dict, which is
populated when reading from netCDF or Zarr:

```python
da_encoded = xr.DataArray(np.ones(3), dims="x")
da_encoded.encoding = {"_FillValue": -999.0, "dtype": "float32"}

pa.DataArraySchema(
    encoding={"_FillValue": -999.0, "dtype": "^float.*"},
).validate(da_encoded)
```

Encoding supports the same matching modes as `attrs` (equality, regex,
callable) plus pydantic models. See [Encoding](./encoding.html) for full details.

## Chunked / array type

Control whether the underlying storage is lazy (Dask) or eager (NumPy):

```python
pa.DataArraySchema(chunked=True)       # must be Dask-backed
pa.DataArraySchema(chunked=False)      # must be eager
pa.DataArraySchema(array_type=np.ndarray)  # must be a numpy array
```

See [Duck Arrays](./duck_arrays.html) for comprehensive Dask integration
documentation, and [Configuration](./configuration.html) for how `chunked` interacts
with validation depth.

## Data-level checks

Use [`Check`](../reference/generated/pandera.api.checks.Check.html) for value-level assertions:

```python
schema = pa.DataArraySchema(
    dtype=np.float64,
    checks=[
        pa.Check(lambda da: float(da.min()) >= 0),
        pa.Check(lambda da: float(da.max()) <= 100),
    ],
)

da_checked = xr.DataArray(np.linspace(0, 50, 10), dims="x")
schema.validate(da_checked)
```

See [Checks and Parsers](./checks_and_parsers.html) for built-in check helpers and details on
how checks interact with lazy / chunked data.

## Parsers

[`Parser`](../reference/generated/pandera.api.parsers.Parser.html) objects run **before** checks and can
transform the array:

```python
schema = pa.DataArraySchema(
    parsers=pa.Parser(lambda da: da.fillna(0)),
    nullable=False,
)

da_nulls = xr.DataArray([1.0, np.nan, 3.0], dims="x")
validated = schema.validate(da_nulls)
validated
```

## Validation options

`schema.validate(da)` accepts several keyword arguments:

- `lazy` — collect all failures into [`SchemaErrors`](../reference/generated/pandera.errors.SchemaErrors.html)
  instead of raising on the first one.
- `head` / `tail` / `sample` — subsample along the first dimension before
  running heavy checks.
- `inplace` — if `True`, coercion mutates the original object.

```python
schema = pa.DataArraySchema(
    dtype=np.float64,
    dims=("x",),
    name="values",
    checks=pa.Check(lambda da: bool((da > 0).all())),
)

da_bad = xr.DataArray([-1, 2, 3], dims="x", name="wrong_name")

try:
    schema.validate(da_bad, lazy=True)
except pa.errors.SchemaErrors as exc:
    print(exc)
```

## See also

- [DatasetSchema](./dataset_schema.html) — [`DatasetSchema`](../reference/generated/pandera.api.xarray.container.DatasetSchema.html) for multi-variable data
- [Data Models](./data_models.html) — class-based [`DataArrayModel`](../reference/generated/pandera.api.xarray.model.DataArrayModel.html)
- [Checks and Parsers](./checks_and_parsers.html) — checks, parsers, lazy validation
- [Encoding](./encoding.html) — encoding validation (netCDF/Zarr metadata)
- [Duck Arrays](./duck_arrays.html) — Dask integration, `chunked`, and `array_type`
- [CF Convention Checks](./cf_conventions.html) — CF convention checks
- [Decorators](./decorators.html) — `check_input`, `check_output`, `check_io`, and `check_types`
- [Configuration](./configuration.html) — [`ValidationDepth`](../reference/generated/pandera.config.ValidationDepth.html),
  [`ValidationScope`](../reference/generated/pandera.config.ValidationScope.html), Dask, environment variables
- [Xarray API reference](../reference/xarray.html) — full API reference for all xarray classes
- [Core API reference](../reference/core.html) — [`PanderaConfig`](../reference/generated/pandera.config.PanderaConfig.html),
  [`ValidationDepth`](../reference/generated/pandera.config.ValidationDepth.html),
  [`ValidationScope`](../reference/generated/pandera.config.ValidationScope.html)
