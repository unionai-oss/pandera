---
file_format: mystnb
---

(xarray-error-reporting)=

# Error reports and lazy validation

Pandera’s xarray backend raises the same exception types as other backends:
{class}`~pandera.errors.SchemaError` when validation stops at the first problem,
and {class}`~pandera.errors.SchemaErrors` when you pass `lazy=True` to collect every
failure in one pass. The consolidated summary in `SchemaErrors` is the same
*error report* idea described in {ref}`error-report` and {ref}`lazy-validation`,
adapted to labelled N-dimensional arrays.

This page shows **schema** failures (structure, dtype, dims, metadata) versus
**data** failures (checks, nullability when it is treated as data-scope), how to
read **`lazy=True`** output, and what **`failure_cases`** look like for
{class}`~xarray.DataArray` validation.

## Eager validation: `SchemaError`

By default, `validate()` raises as soon as a rule fails. Structural problems
(wrong dtype, dims, name, missing coordinates, and so on) and data-level
failures (failing {class}`~pandera.api.checks.Check`, nulls when `nullable=False`)
both use `SchemaError`, but they correspond to different **reason codes**
(see {class}`~pandera.errors.SchemaErrorReason`).

### Schema-level failure

```{code-cell} python
import numpy as np
import xarray as xr
import pandera.xarray as pa

schema = pa.DataArraySchema(
    dtype=np.float64,
    dims=("time", "lat"),
    name="temperature",
)

da = xr.DataArray(
    np.zeros((3, 4)),
    dims=("time", "lon"),  # wrong dim name
    name="temperature",
)

try:
    schema.validate(da)
except pa.errors.SchemaError as exc:
    print("reason:", exc.reason_code)
    print("check:", exc.check)
    print("message:", exc)
```

Here `reason_code` is typically `MISMATCH_INDEX` for dimension/shape/coord
mismatches (historical name shared with pandas), or `WRONG_DATATYPE`,
`WRONG_FIELD_NAME`, etc., for other structural issues.

### Data-level failure (`Check`)

```{code-cell} python
schema = pa.DataArraySchema(
    dtype=np.float64,
    dims=("x",),
    checks=pa.Check.in_range(0, 1),
)

da = xr.DataArray([0.0, 2.0, 0.5], dims=("x",))

try:
    schema.validate(da)
except pa.errors.SchemaError as exc:
    print("reason:", exc.reason_code)
    print("check:", exc.check)
```

For element-wise checks, `exc.failure_cases` is often an {class}`~xarray.DataArray`
of the same shape as the data, with **NaN** where values passed and the failing
values kept where the check failed (a masked view of the original array).

```{code-cell} python
try:
    schema.validate(da)
except pa.errors.SchemaError as exc:
    print(exc.failure_cases)
```

## Lazy validation: `SchemaErrors` and `lazy=True`

Pass `lazy=True` to run all applicable checks and raise a single
{class}`~pandera.errors.SchemaErrors`. Its string form is JSON, and the
`.message` attribute holds the same structured dict.

```{code-cell} python
import json

schema = pa.DataArraySchema(
    dtype=np.float64,
    dims=("x",),
    name="values",
    checks=pa.Check.ge(0),
)

da = xr.DataArray([-1.0, 2.0, 3.0], dims="x", name="wrong_name")

try:
    schema.validate(da, lazy=True)
except pa.errors.SchemaErrors as exc:
    print(json.dumps(exc.message, indent=2))
```

### How to read `exc.message`

The summary is a nested dict:

1. **Top level** — `SCHEMA` vs `DATA` (names of {class}`~pandera.api.base.error_handler.ErrorCategory`).
   - **SCHEMA** — structural validation (dtype, dims, coords, attrs, encoding,
     name, `chunked` / `array_type`, strict flags, etc.).
   - **DATA** — data-scope rules such as user `Check` failures and certain
     nullable / duplicate semantics, depending on reason code.

2. **Second level** — reason codes as strings, e.g. `WRONG_FIELD_NAME`,
   `WRONG_DATATYPE`, `MISMATCH_INDEX`, `DATAFRAME_CHECK` (check failures; name
   shared with pandas), `SERIES_CONTAINS_NULLS`, etc.

3. **Entries** — each failure is a small dict with:
   - **`schema`** — the schema’s `name` (or a sensible label).
   - **`column`** — the data variable or coordinate key involved (for a standalone
     {class}`~xarray.DataArray`, this matches the array `name` when set).
   - **`check`** — structural check id (e.g. `"name"`, `"dims"`) or the check
     repr (e.g. `"greater_than_or_equal_to(0)"`).
   - **`error`** — short string describing the failure.

`DATAFRAME_CHECK` appears for generic check pipeline failures on non-dataframe
objects; treat it as “this `Check` failed,” not as a pandas-only concept.

### `error_counts`

```{code-cell} python
try:
    schema.validate(da, lazy=True)
except pa.errors.SchemaErrors as exc:
    print(exc.error_counts)
```

This maps each **reason code** to how many errors of that type were collected.

### `schema_errors` vs `failure_cases` on `SchemaErrors`

- **`exc.schema_errors`** — list of {class}`~pandera.errors.SchemaError`
  instances in collection order. Use this for programmatic access: each item has
  `reason_code`, `check`, `failure_cases`, `message`, `schema`, and `data`
  (may be cleared in lazy mode for large objects — rely on the error fields).

- **`exc.failure_cases`** — a simple **list of strings**, one human-readable
  summary per collected error (aligned with `schema_errors`). Handy for logging;
  for masks and coordinates, inspect each `SchemaError` in `schema_errors`.

```{code-cell} python
try:
    schema.validate(da, lazy=True)
except pa.errors.SchemaErrors as exc:
    for err in exc.schema_errors:
        print(err.reason_code.name, "->", type(err.failure_cases).__name__)
```

For a failing **element-wise** check, `err.failure_cases` is often a
{class}`~xarray.DataArray` with coordinates preserved so you can locate bad
points in label space.

## Datasets

{class}`~pandera.api.xarray.container.DatasetSchema` validation aggregates errors from
dataset-level rules and from each `data_vars` / `coords` slice. Lazy reports
use the same `SCHEMA` / `DATA` grouping; **`column`** identifies the data
variable or coordinate name that failed.

```{code-cell} python
import json

ds_schema = pa.DatasetSchema(
    data_vars={
        "a": pa.DataVar(dtype=np.float64, dims=("x",)),
        "b": pa.DataVar(dtype=np.float64, dims=("x",)),
    },
    dims=("x",),
    sizes={"x": 3},
)

ds = xr.Dataset(
    {
        "a": ("x", [1.0, 2.0, 3.0]),
        "b": ("x", np.array([1, 2, 3], dtype=np.int64)),  # schema expects float64
    },
)

try:
    ds_schema.validate(ds, lazy=True)
except pa.errors.SchemaErrors as exc:
    print(json.dumps(exc.message, indent=2))
```

## Validation depth and what appears in the report

Global {class}`~pandera.config.ValidationDepth` controls which **scopes** run
(e.g. `SCHEMA_ONLY` skips user checks on chunked arrays by default). Errors from
skipped scopes **do not** appear in `SchemaErrors`. See
{ref}`xarray-configuration` and {ref}`xarray-duck-arrays`.

With `SCHEMA_ONLY`, a value that only fails a user `Check` may **pass**
validation (no exception):

```{code-cell} python
from pandera.config import ValidationDepth, config_context

schema = pa.DataArraySchema(
    dims=("x",),
    name="a",
    checks=pa.Check.ge(0),
)
da = xr.DataArray([-1.0], dims="x", name="a")

with config_context(validation_depth=ValidationDepth.SCHEMA_ONLY):
    out = schema.validate(da)
    print("SCHEMA_ONLY: validation returned", type(out).__name__)
```

With the default depth (`SCHEMA_AND_DATA` for eager arrays), the same data
raises and the failure shows up under `DATA`:

```{code-cell} python
try:
    schema.validate(da, lazy=True)
except pa.errors.SchemaErrors as exc:
    print("error_counts:", exc.error_counts)
```

## See also

- {ref}`error-report` — error reports for pandas / PySpark
- {ref}`lazy-validation` — lazy validation concepts (pandas-oriented)
- {ref}`xarray-checks-parsers` — checks, parsers, and lazy validation on xarray
- {ref}`xarray-configuration` — `ValidationDepth` and environment variables
- {ref}`xarray-duck-arrays` — Dask-backed arrays and default schema-only data checks
