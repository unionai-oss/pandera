# ADR 20260819-02: Separate column presence from value nullability

- Status: Approved
- Date: 2026-08-19

## Context

Pandera has two different concepts that are easy to conflate:

- whether a declared column must be present in the input data; and
- whether values in a present column may be null.

The annotation contract should make the distinction visible. `None` inside
`FieldType[...]` describes value nullability, while an outer `| None` on the
annotation describes optional field presence. `Field(required=...)` and
`Field(nullable=...)` remain explicit overrides.

Pandera has interpreted an outer `| None` as optional column presence since
the `DataFrameModel` API was introduced, for both `Optional[Series[T]]` and
bare `Optional[T]` annotations. Redefining a bare `T | None` as a required,
nullable column silently changed the meaning of existing schemas
([#2457](https://github.com/unionai-oss/pandera/issues/2457)), so the outer
union keeps its historical presence meaning for every annotation form.

## Decision

DataFrameModel fields use the following meanings:

```python
from pandera.typing import FieldType


class Schema(pa.DataFrameModel):
    required: int
    nullable_values: FieldType[int | None]
    optional_presence: FieldType[int] | None
    legacy_optional_presence: int | None
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
| `FieldType[T \| None]` | Required column whose values may be null |
| `T \| None` or `FieldType[T] \| None` | Column may be absent from the input data |

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
over nullability inferred from `FieldType[T | None]`.

The established `Optional[Series[T]]` and bare `Optional[T]` spellings remain
supported and continue to mean optional column presence. Optional index
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

- Value nullability cannot be expressed by a bare `T | None` annotation; it
  requires `FieldType[T | None]` or an explicit `Field(nullable=True)`.
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

This keeps all metadata in `Field`, but loses the type-level spelling of
nullability entirely. `FieldType[T | None]` remains the annotation-level way
to express value nullability, with explicit `Field(nullable=...)` as an
override.

### Reinterpret a bare T | None as nullable values

This was the contract released in `0.33.0`. It reads naturally, but it
silently flipped every pre-existing `col: T | None` declaration from optional
to required, breaking user schemas without a deprecation path
([#2457](https://github.com/unionai-oss/pandera/issues/2457)). The outer
union is therefore reserved for presence in every annotation form, and the
dtype position inside `FieldType[...]` carries nullability.

## Implementation status

Tests cover required fields, nullable values, optional presence, explicit
nullable/required overrides, `FieldType` metadata, assignment precedence, the
legacy `Optional[Series[T]]` compatibility path, and optional runtime model
arguments.

## References

- [Pydantic fields documentation](https://docs.pydantic.dev/latest/concepts/fields/)
- [Pandera PR #2255](https://github.com/pandera-dev/pandera/pull/2255)
