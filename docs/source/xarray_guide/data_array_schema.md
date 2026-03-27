---
file_format: myst
---

(xarray-data-array-schema)=

# DataArray Schemas

{class}`~pandera.api.xarray.container.DataArraySchema` validates a single
{class}`~xarray.DataArray`. It is the xarray counterpart of
{class}`~pandera.api.pandas.array.SeriesSchema` — but for arbitrary-rank
labelled arrays rather than 1-D series.

You can also express the same constraints with the declarative
{ref}`DataArrayModel <xarray-data-models>`.

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
pa.DataArraySchema(dtype=float)
pa.DataArraySchema(dtype=np.float32)
pa.DataArraySchema(dtype="int64")
```

If `dtype` is `None`, any dtype is accepted.

## Dimension validation

`dims` enforces dimension names **in order**. `None` entries act as
wildcards that match any name:

```python
pa.DataArraySchema(dims=("time", "lat", "lon"))
pa.DataArraySchema(dims=("time", None, None))  # only first dim is checked
```

The tuple length also constrains the rank (ndim).

## Sizes and shape

`sizes` is the idiomatic xarray way to constrain dimension lengths.
`shape` does the same thing positionally. They are mutually exclusive.

```python
pa.DataArraySchema(
    dims=("time", "lat", "lon"),
    sizes={"lat": 180, "lon": 360},  # time is unconstrained
)

pa.DataArraySchema(
    dims=("time", "lat", "lon"),
    shape=(None, 180, 360),
)
```

## Coordinate validation

Pass a `dict[str, Coordinate]` to validate coordinate arrays, or a
`list[str]` as shorthand for "these coordinates must exist":

```python
schema = pa.DataArraySchema(
    dims=("x", "y"),
    coords={
        "x": pa.Coordinate(dtype=np.float64, dimension=True),
        "y": pa.Coordinate(dtype=np.float64, dimension=True),
        "label": pa.Coordinate(dimension=False),
    },
)
```

{class}`~pandera.api.xarray.components.Coordinate` is documented in
detail under {ref}`xarray-dataset-schema`.

### Strict coordinates

With `strict_coords=True`, the schema fails if the DataArray has
coordinates not listed in `coords`:

```python
pa.DataArraySchema(
    coords={"x": pa.Coordinate()},
    strict_coords=True,
)
```

## Attribute validation

`attrs` checks the DataArray's `.attrs` dict for key existence and value
equality:

```python
pa.DataArraySchema(
    attrs={"units": "K", "standard_name": "air_temperature"},
)
```

With `strict_attrs=True`, extra attributes cause a validation error.

## Name validation

```python
pa.DataArraySchema(name="temperature")
```

The DataArray's `.name` must match exactly.

## Null values

By default `nullable=False` — any NaN or null value raises a
{class}`~pandera.errors.SchemaError`. Set `nullable=True` to allow them:

```python
pa.DataArraySchema(dtype=float, nullable=True)
```

## Coercing dtypes

When `coerce=True`, the DataArray is cast to `dtype` before validation:

```python
schema = pa.DataArraySchema(dtype=np.float32, coerce=True)
da_int = xr.DataArray(np.array([1, 2, 3]), dims="x")
validated = schema.validate(da_int)  # now float32
```

## Chunked / array type

Control whether the underlying storage is lazy (Dask) or eager (NumPy):

```python
pa.DataArraySchema(chunked=True)       # must be Dask-backed
pa.DataArraySchema(chunked=False)      # must be eager
pa.DataArraySchema(array_type=np.ndarray)  # must be a numpy array
```

See {ref}`xarray-configuration` for how `chunked` interacts with
validation depth.

## Data-level checks

Use {class}`~pandera.api.checks.Check` for value-level assertions:

```python
schema = pa.DataArraySchema(
    dtype=np.float64,
    checks=[
        pa.Check(lambda da: float(da.min()) >= 0),
        pa.Check(lambda da: float(da.max()) <= 100),
    ],
)
```

See {ref}`xarray-checks-parsers` for built-in check helpers and details on
how checks interact with lazy / chunked data.

## Parsers

{class}`~pandera.api.parsers.Parser` objects run **before** checks and can
transform the array:

```python
schema = pa.DataArraySchema(
    parsers=pa.Parser(lambda da: da.fillna(0)),
    nullable=False,
)
```

## Validation options

`schema.validate(da)` accepts several keyword arguments:

- `lazy` — collect all failures into {class}`~pandera.errors.SchemaErrors`
  instead of raising on the first one.
- `head` / `tail` / `sample` — subsample along the first dimension before
  running heavy checks.
- `inplace` — if `True`, coercion mutates the original object.

```python
try:
    schema.validate(da, lazy=True)
except pa.errors.SchemaErrors as exc:
    print(exc.failure_cases)
```

## See also

- {ref}`xarray-dataset-schema` — `DatasetSchema` for multi-variable data
- {ref}`xarray-data-models` — class-based `DataArrayModel`
- {ref}`xarray-checks-parsers` — checks, parsers, lazy validation
- {ref}`xarray-configuration` — validation depth, Dask, environment variables
