# ADR 20260819-02: Separate column presence from value nullability

- Status: Proposed
- Date: 2026-08-19
- Approval: Pending an approval comment on the implementing pull request

## Context

Pandera has two different concepts that are easy to conflate:

- whether a declared column must be present in the input data; and
- whether values in a present column may be null.

Today, an outer optional annotation such as
`Series[int] | None` makes the column not required, while
`Field(nullable=True)` controls value nullability. The runtime `FieldInfo`
descriptor still returns the declared column name as a string in either case.

SQLAlchemy's `Mapped[Optional[T]]` is a useful precedent for putting value
nullability inside the wrapper. However, its class-level value is an
`InstrumentedAttribute[T | None]`, not the literal column name. Pandera must
keep that distinction because `DataFrameModel` class access returns the name
itself.

## Decision

The new `Column` syntax will keep the two concepts distinct:

```python
class Schema(pa.DataFrameModel):
    required: Column[int]
    nullable_values: Column[int | None]
    optional_presence: Column[int] | None
```

The meanings are:

| Annotation | Schema meaning | Class-level typing |
| --- | --- | --- |
| `Column[T]` | Required column; non-nullable by default | `str` |
| `Column[T | None]` | Present column whose values may be null | `str` |
| `Column[T] | None` | Column may be absent from the input data | `str | None` |

The parser will unwrap an inner optional type when processing
`Column[T | None]` and use it to establish value nullability. An explicit
`Field(nullable=...)` setting remains the runtime configuration mechanism and
must take precedence when supplied. If the current `False` default cannot
distinguish an omitted setting from an explicit `nullable=False`, the
implementation must preserve that distinction before adding this behavior.

The outer-union spelling is intentional. Both mypy and ty derive
`str | None` from `Column[T] | None` using ordinary descriptor and union
semantics. Python's standard typing model has no portable conditional type
function that can make `Column[T]` resolve to `str` while making
`Column[Optional[T]]` resolve to `str | None`.

The existing `Optional[Series[T]]` spelling remains supported and continues to
mean optional column presence. It is not reinterpreted as value nullability.

## Consequences

### Positive

- Users can state presence and value nullability independently.
- The spelling follows the useful part of SQLAlchemy's convention without
  pretending that Pandera's class-level string has the same runtime role as an
  ORM expression.
- Required-column class references remain directly usable where `str` is
  expected.
- Optional-column references conservatively require handling `None` in static
  code, even though the declaration descriptor itself has a runtime name.

### Negative

- Users must learn the distinction between inner and outer optionality.
- `Column[T] | None` is a conservative static representation of optional
  presence; runtime `FieldInfo` access continues to return the declared name.
- Backend parsers need consistent support for inner optional types and
  explicit `Field(nullable=...)` precedence.

## Alternatives considered

### Make `Column[Optional[T]]` produce `str | None`

This is attractive syntactically, but it requires a conditional mapping from
the generic argument to the descriptor's return type. Standard typing cannot
express that mapping consistently for both mypy and ty. Self-type overloads
were checked as a prototype: ty and mypy select incompatible overloads.

### Add a separate `OptionalColumn[T]` marker

This could model an optional class-level result directly, but it introduces a
second field marker and is less concise than the ordinary outer union already
understood by both checkers.

### Use only `Field(nullable=True)`

This remains supported, but does not provide a type-level way to express
nullable values. `Column[T | None]` supplies that information while preserving
the existing explicit `Field` configuration channel.

## Approval and implementation gate

This ADR remains `Proposed` until an approval comment is added to the
implementing pull request. Before implementation is considered complete, the
TDD suite must cover runtime schema construction, class-level runtime values,
and no-plugin mypy and ty checks for all three annotation forms.

## References

- [SQLAlchemy declarative typing documentation](https://docs.sqlalchemy.org/en/20/orm/declarative_tables.html)
- [Pandera PR #2255](https://github.com/unionai-oss/pandera/pull/2255)
