# ADR 20260819-01: Use FieldType for static DataFrameModel fields

- Status: Approved
- Date: 2026-08-19

## Context

`DataFrameModel` fields have two deliberately different lives. At runtime,
Pandera replaces the class attribute with a field descriptor whose class-level
value is the column name (`str`). In static analysis, a checker sees the
field's annotation as the attribute type. With native backend annotations such
as Polars' `pl.List`, using a schema field as a column name can therefore
produce a false positive even though the runtime behavior is correct.

The existing mypy plugin can compensate for this in mypy, but a
checker-specific plugin cannot provide the same contract to other checkers
such as `ty`. The public typing API needs a backend-neutral descriptor layer,
distinct from each backend's runtime `pa.Field` function.

## Decision

Pandera exposes the typing-only `pandera.typing.FieldType` descriptor. Its
first generic argument is the dtype consumed by the runtime model parser, and
additional arguments may carry a backend-specific `pa.Field(...)` object and
any dtype parameters:

```python
from pandera.typing import FieldType
import polars as pl
import pandera.polars as pa


class Schema(pa.DataFrameModel):
    values: FieldType[pl.List, pa.Field()]
    nullable_field: FieldType[int | None, pa.Field()]
    optional_field: FieldType[int] | None
```

The assignment form is also supported and keeps value expressions out of type
arguments for static checkers:

```python
class Schema(pa.DataFrameModel):
    field: FieldType[int] = pa.Field(gt=0)
```

`FieldType[T]` models class-level access as `str` while retaining `T` as the
schema dtype. It is typing-only and must not be instantiated; the backend
runtime `pa.Field(...)` function remains the source of field checks and
metadata. An explicit assignment takes precedence over metadata supplied in
`FieldType`.

Field metadata is represented by `FieldType` metadata or the assignment form,
rather than requiring an outer `typing.Annotated` wrapper. Existing
`Annotated` support for backend dtype parameters remains independent of this
descriptor contract.

The shared parser extracts the dtype and field metadata before constructing
backend-specific schema components. All supported DataFrameModel backends
use the same descriptor and preserve the existing field checks, aliases,
requiredness, and nullability behavior.

## Consequences

### Positive

- Generic checkers can accept `Schema.field` where a `str` column name is
  expected without a checker-specific plugin.
- The descriptor name is backend-neutral and does not collide conceptually
  with `pandera.polars.Field`, `pandera.pandas.Field`, or other runtime
  `Field` functions.
- Field metadata can sit beside the dtype in a concise declaration, while
  assignment-based declarations remain available.
- The runtime parser owns one contract across pandas, Polars, Ibis, PyArrow,
  PySpark, and xarray-compatible model infrastructure.

### Negative

- `FieldType` is a Pandera-specific typing construct that users must learn.
- A generic checker may reject a value expression such as `pa.Field()` inside
  a type argument even though Pandera can parse it at runtime; the assignment
  form is the checker-friendly spelling in that case.
- The parser must distinguish dtype parameters from field metadata and must
  preserve explicit assignment precedence.

## Alternatives considered

### Add a backend-neutral FieldType descriptor

The original proposal introduced a typing-only descriptor named `Column`,
but `FieldType` is more future-proof alongside xarray and TensorDict support
and clearly separates the typing layer from backend runtime `Field` objects.

### Require a checker-specific plugin

An analyzer-specific plugin could teach one checker about Pandera's runtime
descriptor, but it would not provide the same contract to `ty` and other
checkers. The public descriptor keeps the bridge backend-neutral and plugin
free.

### Use only `typing.Annotated`

`Annotated[T, ...]` preserves the underlying `T` for generic static analysis,
so it cannot by itself prevent the false positive when a model attribute is
used as a column name. It remains useful for dtype metadata, but it is not the
static descriptor layer.

### Keep only assignment-based `Field()` declarations

Assignment remains supported and is the most checker-friendly spelling, but
it does not provide the concise metadata-bearing annotation proposed by the
maintainer. `FieldType` supports both forms.

## Implementation status

The implementation preserves the `FieldType[T]` no-plugin static contract,
`FieldType` metadata and assignment precedence, and the presence/nullability
rules in ADR 20260819-02 across supported DataFrameModel backends.

## References

- [Pydantic fields documentation](https://docs.pydantic.dev/latest/concepts/fields/)
- [Pandera PR #2255](https://github.com/pandera-dev/pandera/pull/2255)
