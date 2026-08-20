# ADR 20260819-01: Use a typing-only `Column[T]` for DataFrameModel fields

- Status: Proposed
- Date: 2026-08-19
- Approval: Pending an approval comment on the implementing pull request

## Context

`DataFrameModel` annotations currently describe the schema dtype, for example:

```python
class UserSchema(pa.DataFrameModel):
    user_id: int
```

At runtime, Pandera replaces the annotated class attribute with a `FieldInfo`
descriptor. Accessing `UserSchema.user_id` therefore returns the column name as
a `str`. Static checkers, however, generally interpret the annotation itself
as the class attribute type. With a native backend type such as
`pl.List`, `ty` consequently reports a false positive when the class attribute
is used as a column name.

The existing mypy plugin corrects this for DataFrameModel fields, but that is
checker-specific. SQLAlchemy's `Mapped[T]` provides an established precedent
for an explicit, typing-only generic descriptor whose runtime attribute is
replaced by an ORM-managed object.

## Decision

Pandera will introduce a backend-neutral public typing construct:

```python
from pandera.typing import Column


class UserSchema(pa.DataFrameModel):
    user_id: Column[int] = pa.Field()


column_name: str = UserSchema.user_id
```

`Column[T]` will have these semantics:

1. `T` is the schema's dtype annotation and remains available to Pandera's
   runtime model parser.
2. Class-level access to a required `Column[T]` is typed as `str`.
3. The typing definition is descriptor-shaped but typing-only; runtime
   `DataFrameModel` construction continues to install and use `FieldInfo`.
4. `Column` is distinct from `pa.Column`, which is a runtime schema component.
   It belongs in `pandera.typing` and is intended specifically for
   `DataFrameModel` annotations.
5. `Field()` and `Annotated` remain the mechanisms for field constraints and
   metadata.

The parser will recognize `Column[T]` through shared annotation handling across
DataFrameModel backends. Existing bare annotations, `Series[T]` annotations,
and the mypy plugin remain supported.

## Consequences

### Positive

- The new spelling models Pandera's runtime descriptor explicitly.
- The same public annotation can work with mypy and ty without requiring a
  checker plugin.
- The schema dtype remains in the annotation rather than being moved into
  `Field()` metadata.
- Existing users do not need to migrate from currently supported annotation
  forms.

### Negative

- Pandera gains a second concept named `Column`: the typing marker and the
  runtime schema component. The separate import path and documentation must
  make this distinction clear.
- Every DataFrameModel backend must recognize the new marker consistently;
  relying on a backend's incidental dtype fallback is not sufficient.
- The typing-only descriptor is a static contract and does not itself change
  runtime descriptor behavior.

## Alternatives considered

### Extend the mypy plugin

This solves mypy but cannot provide the same behavior to ty, which has no
Pandera plugin integration. It also makes the public type contract depend on a
particular checker.

### Alias `Column[T]` to `Series[T]`

This would reuse an existing descriptor, but `Series[T]` describes a data
series rather than a DataFrameModel field name. A dedicated marker gives the
runtime and static meanings a clearer boundary.

### Use `Annotated[str, T]`

This makes the class attribute type `str`, but does not naturally preserve the
schema dtype as the primary type argument or express the separate optionality
semantics in ADR 0002.

### Add `dataclass_transform` to DataFrameModel

`dataclass_transform` is useful for synthesized constructors and field
specifiers. It does not change the type of an annotated class attribute and is
not sufficient for this problem.

## Approval and implementation gate

This ADR remains `Proposed` until an approval comment is added to the
implementing pull request. Implementation and status changes should preserve
the acceptance requirements above and the optionality rules in ADR 0002.

## References

- [SQLAlchemy declarative typing documentation](https://docs.sqlalchemy.org/en/20/orm/declarative_tables.html)
- [Pydantic fields documentation](https://docs.pydantic.dev/latest/concepts/fields/)
- [Pandera PR #2255](https://github.com/unionai-oss/pandera/pull/2255)
