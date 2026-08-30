# ADR 20260819-02: Separate column presence from value nullability

- Status: Approved
- Date: 2026-08-19

## Context

Pandera has two different concepts that are easy to conflate:

- whether a declared column must be present in the input data; and
- whether values in a present column may be null.

The annotation contract should make the distinction visible. `None` inside a
dtype describes value nullability, while an outer `| None` on the typing-only
`FieldType` descriptor describes optional field presence. `Field(required=...)`
and `Field(nullable=...)` remain explicit overrides.

## Decision

DataFrameModel fields use the following meanings:

```python
from pandera.typing import FieldType


class Schema(pa.DataFrameModel):
    required: int
    nullable_values: int | None
    optional_presence: FieldType[int] | None
```

When the static descriptor is needed for every field, the equivalent forms
are:

```python
class StaticSchema(pa.DataFrameModel):
    required: FieldType[int]
    nullable_values: FieldType[int | None]
    optional_presence: FieldType[int] | None
```

Field metadata can be supplied as additional `FieldType` metadata or through
assignment:

```python
class MetadataSchema(pa.DataFrameModel):
    checked: FieldType[int, pa.Field(gt=0)]
    assigned: FieldType[int] = pa.Field(description="assigned")
```

The meanings are:

| Annotation | Schema meaning |
| --- | --- |
| `T` or `FieldType[T]` | Required column; non-nullable by default |
| `T \| None` or `FieldType[T \| None]` | Required column whose values may be null |
| `FieldType[T] \| None` | Column may be absent from the input data |

`pandera.typing.FieldType[T]` is typing-only and does not replace the runtime
backend `pa.Field(...)` function. Explicit `required` and `nullable` values
override inference independently, and an explicit assignment takes precedence
over metadata supplied in `FieldType`.

The annotation parser retains outer-union optionality for generic model
containers used by runtime decorators. It separately records field-presence
optionality for schema annotations so that `check_types` can accept an
optional model argument without making a required schema field optional.

`Field(required=...)` is optional so that an omitted value does not override
the annotation's inferred presence. When supplied, it takes precedence over
the annotation. Likewise, an explicit `Field(nullable=...)` takes precedence
over nullability inferred from `T | None`.

The established `Optional[Series[T]]` spelling remains supported for
backwards compatibility and continues to mean optional column presence. This
legacy wrapper is distinct from a bare `T | None` annotation. Optional index
annotations remain invalid where the backend has historically rejected them.

## Consequences

### Positive

- Presence and value nullability can be stated independently.
- `FieldType[T]` gives generic checkers the class-level `str` descriptor
  contract without a checker-specific plugin.
- Field metadata remains concise and backend-neutral.
- Existing `Series[T]` declarations and explicit `Field(nullable=...)`
  overrides remain supported.

### Negative

- Users must distinguish a bare `T | None` from `FieldType[T] | None` and the
  legacy `Optional[Series[T]]` spelling.
- The `FieldType` implementation must track whether `required` and `nullable`
  were explicitly supplied so omitted values do not override inference.
- Backend parsers must apply the same rules consistently.

## Alternatives considered

### Use Annotated metadata for presence

This was the earlier metadata-carrier design, but the maintainer-approved
`FieldType` descriptor is more concise and also models the class-level access
needed by generic static checkers. `Annotated` remains available for existing
dtype-parameter syntax, but is not required for field metadata here.

### Use an earlier descriptor and an outer union for presence

The original proposal used `Column[T | None]` for nullable values and
`Column[T] | None` for optional presence. The future-proof `FieldType` name
preserves that descriptor idea while differentiating it from backend runtime
`Field` functions.

### Make nullable metadata the only nullable spelling

This keeps all metadata in `Field`, but loses the conventional type-level
meaning of `T | None`. The annotation form remains the primary way to express
value nullability, with explicit `Field(nullable=...)` as an override.

### Use only required metadata and keep T | None as presence optional

This would preserve the old interpretation for bare dtype annotations, but
would make `None` mean different things depending on whether it describes a
dtype or a field declaration. The contract reserves bare `T | None` for
nullable values and uses `FieldType[T] | None` for optional presence.

## Implementation status

Tests cover required fields, nullable values, optional presence, explicit
nullable/required overrides, `FieldType` metadata, assignment precedence, the
legacy `Optional[Series[T]]` compatibility path, and optional runtime model
arguments.

## References

- [Pydantic fields documentation](https://docs.pydantic.dev/latest/concepts/fields/)
- [Pandera PR #2255](https://github.com/pandera-dev/pandera/pull/2255)
