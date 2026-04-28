# Optimized Data Synthesis Strategies Spec

> **Status:** Draft
> **Scope:** `pandera/strategies/pandas_strategies.py`,
> `pandera/strategies/xarray_strategies.py`,
> `pandera/strategies/base_strategies.py`,
> `pandera/api/extensions.py` (additive), `pandera/api/checks.py` (additive),
> `pandera/backends/pandas/builtin_checks.py`
> **Author:** pandera maintainers

---

## 1. Motivation

`pandera`'s data synthesis layer is built on top of
[`hypothesis`](https://hypothesis.readthedocs.io/) and is responsible for
generating dataframes, series, indexes and (post #1929) xarray containers
that satisfy a schema's type and `Check` constraints. It powers the
user-facing `Schema.strategy()` and `Schema.example()` APIs and the
`pa.Field(...)` data generation behaviour described in the
[Data Synthesis Strategies guide](../docs/source/data_synthesis_strategies.md).

The current implementation in `pandera/strategies/pandas_strategies.py`
follows a "chained strategy" pattern (see
[`field_element_strategy`](../pandera/strategies/pandas_strategies.py)) where
each `Check` contributes a hypothesis strategy on top of the strategy produced
by the previous check:

```226:870:pandera/strategies/pandas_strategies.py
def gt_strategy(...):
    if strategy is None:
        strategy = pandas_dtype_strategy(
            pandera_dtype,
            min_value=min_value,
            exclude_min=True if is_float(pandera_dtype) else None,
        )
    return strategy.filter(partial(operator.lt, min_value))
```

```853:870:pandera/strategies/pandas_strategies.py
for check in checks:
    check_strategy = (
        check.strategy
        if check.strategy is not None
        else STRATEGY_DISPATCHER.get((check.name, pd.Series), None)
    )
    if check_strategy is not None:
        elements = check_strategy(
            pandera_dtype, elements, **check.statistics
        )
```

This design has known problems:

1. **Filter chaining is slow.** Combining e.g. `Check.gt(0)`, `Check.lt(100)`,
   `Check.notin([10, 20])`, `Check.isin(range(0, 100))` produces a strategy
   that draws an unbounded value of the dtype, then runs four
   `.filter(...)`s. Hypothesis health-checks (`filter_too_much`,
   `data_too_large`) trigger frequently, and the interpreter spends most of
   its time discarding draws.
2. **`Unsatisfiable` errors arise gratuitously.** Because each strategy is
   built independently and then `.filter`-chained, we lose the ability to
   intersect constraints (e.g. take the tightest of multiple
   lower bounds) before handing them to `hypothesis.extra.numpy.from_dtype`.
   In practice, even satisfiable check stacks raise `Unsatisfiable` because
   the base strategy was constructed too loosely.
3. **First-check-wins semantics leak into the user docs.**
   [`check-strategy-chaining`](../docs/source/data_synthesis_strategies.md)
   currently advises users to put the most restrictive constraint *first*.
   This is an implementation artifact and should not be a user-facing
   concern.
4. **Order-dependent base strategy.** `gt_strategy`, `ge_strategy`,
   `lt_strategy`, `le_strategy`, `in_range_strategy`, `isin_strategy`,
   `eq_strategy`, `str_*_strategy`, etc. each have a "no parent strategy"
   path that calls `pandas_dtype_strategy` with one or two `min_value`/
   `max_value`/`pattern` kwargs, and a separate "parent strategy" path that
   `.filter`s. This branch logic is duplicated across nine functions and
   keeps growing as we add checks.
5. **xarray gets nothing.** `xarray_strategies.py` accepts a `checks`
   argument but ignores it (`# (unused) reserved for future check-aware
   generation`).

The goal of this spec is to refactor pandera's strategy layer so that
**all built-in `Check` constraints from a column/series/index/data-array
are first reduced into a single `FieldConstraints` value, and the hypothesis
strategy is constructed in one shot from those merged constraints**, while
keeping the user-facing API and the public extension API intact.

---

## 2. Goals & Non-goals

### 2.1 Goals

1. Eliminate `.filter` chaining for *built-in* checks. Built-in `Check`s
   listed in `pandera/backends/pandas/builtin_checks.py` (and the future
   xarray equivalents) must compile down to a single hypothesis strategy
   call wherever possible.
2. Aggregate compatible constraints across multiple checks of the same kind
   (e.g. `Check.gt(0) & Check.gt(5)` → `min_value=5, exclude_min=True`).
3. Preserve user-facing API:
   - `Schema.strategy(...)`, `Schema.example(...)` signatures and return
     types unchanged.
   - `pa.Check.<name>(...)` factories unchanged.
   - `pa.Check(strategy=fn)` constructor still accepted, with the same
     `(pandera_dtype, strategy=None, **statistics) -> SearchStrategy` calling
     convention.
4. Preserve public extension API:
   - `pandera.api.extensions.register_check_method(..., strategy=fn)` keeps
     working.
   - `pandera.strategies.pandas_strategies.register_check_strategy(fn)` keeps
     working.
   - `pandera.strategies.base_strategies.STRATEGY_DISPATCHER` keeps working.
5. Provide a *new, optional* extension hook
   (`register_check_constraint`) that lets check authors emit a
   `FieldConstraints` value instead of (or in addition to) a hypothesis
   strategy. This is the path that gets the speedup.
6. Apply the same refactor to `xarray_strategies.py`, including making it
   actually consume `checks` (currently a TODO).
7. Maintain or improve test coverage. All existing tests in
   `tests/strategies/` and `tests/pandas/test_extensions.py` must pass
   without modification.

### 2.2 Non-goals

- Changing the validation semantics of any `Check`.
- Introducing a new dataframe backend for strategies (polars/pyspark/ibis
  remain out of scope; they have no strategy support today).
- Replacing `hypothesis` as the underlying engine.
- Auto-deriving strategies for arbitrary user-supplied `lambda` checks
  (in-line custom checks). The fall-back filter behaviour for those is
  preserved as-is.

---

## 3. Background — current architecture

### 3.1 Where strategies are wired up

```
pandera/strategies/
├── __init__.py              # docstring only
├── base_strategies.py       # SearchStrategy stub, STRATEGY_DISPATCHER, decorators
├── pandas_strategies.py     # eq/ne/gt/ge/lt/le/in_range/isin/notin/str_*
│                            # field_element_strategy, series_strategy,
│                            # column_strategy, index_strategy,
│                            # dataframe_strategy, multiindex_strategy
└── xarray_strategies.py     # data_array_strategy, dataset_strategy
```

Strategies are attached to `Check` classes through two mechanisms:

1. **Built-in checks** (`pandera/backends/pandas/builtin_checks.py`) use
   ```python
   @register_builtin_check(strategy=st.gt_strategy, ...)
   def greater_than(...): ...
   ```
   `register_builtin_check` writes the strategy into
   `STRATEGY_DISPATCHER[(name, pd.Series)]` and `[(name, pd.DataFrame)]`.

2. **User-defined checks** use either:
   - `pa.Check(check_fn, strategy=my_strategy)` constructor, or
   - `pa.extensions.register_check_method(..., strategy=my_strategy)`, which
     wraps the strategy with `register_check_strategy(strategy_fn)`
     (defined in `pandas_strategies.py`).

### 3.2 How a column strategy is built today

`field_element_strategy(pandera_dtype, strategy=None, *, checks=None)` walks
the `checks` list and folds each one onto the running strategy:

```853:870:pandera/strategies/pandas_strategies.py
for check in checks:
    check_strategy = (
        check.strategy
        if check.strategy is not None
        else STRATEGY_DISPATCHER.get((check.name, pd.Series), None)
    )
    if check_strategy is not None:
        elements = check_strategy(
            pandera_dtype, elements, **check.statistics
        )
    elif check.element_wise:
        elements = undefined_check_strategy(elements, check)
```

`series_strategy` then wraps `elements` in `pdst.series(...)`,
`column_strategy` wraps it in `pdst.column(...)`, and `dataframe_strategy`
calls `make_row_strategy(col, checks)` which performs the same per-check
fold for dataframe-level row strategies.

### 3.3 Each built-in `*_strategy` has two branches

For example:

```514:534:pandera/strategies/pandas_strategies.py
def gt_strategy(
    pandera_dtype, strategy=None, *, min_value,
):
    if strategy is None:
        strategy = pandas_dtype_strategy(
            pandera_dtype,
            min_value=min_value,
            exclude_min=True if is_float(pandera_dtype) else None,
        )
    return strategy.filter(partial(operator.lt, min_value))
```

The `strategy is None` branch is the "fast" path (one `from_dtype` call).
The `strategy is not None` branch unconditionally appends a `.filter`. When
multiple builtin checks are present, **only the first** check hits the fast
path; the rest filter.

This is the crux of the inefficiency the spec addresses.

---

## 4. Proposed architecture

### 4.1 Core abstraction: `FieldConstraints`

We introduce a new pandera-internal value type that holds the *intersection*
of all numeric/string/membership constraints derived from a list of
`Check`s, before any hypothesis strategy is constructed.

It lives in a new module:

```
pandera/strategies/constraints.py
```

```python
# pandera/strategies/constraints.py

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

UNSET: Any = object()  # sentinel distinct from None


@dataclass(frozen=True)
class FieldConstraints:
    """Aggregated, dtype-agnostic constraints for a single field.

    A field is a 1-D structure: a pandas Series / Index / Column, an
    xarray.DataArray, or a single data variable in an xarray.Dataset.

    All attributes default to ``UNSET``. Merging two ``FieldConstraints``
    intersects their constraints (tightest wins).
    """

    # Numeric / orderable
    min_value: Any = UNSET           # inclusive lower bound
    max_value: Any = UNSET           # inclusive upper bound
    exclude_min: bool = False
    exclude_max: bool = False

    # Membership
    isin: frozenset | None = None    # allowed values
    notin: frozenset = field(default_factory=frozenset)

    # Equality
    eq: Any = UNSET                  # exact value (collapses to single-value)
    # Note: there is no separate `ne` field; Check.ne(value) lowers to
    # notin=frozenset([value]). Same channel, same merge semantics.

    # String
    regex_fullmatch: tuple[str, ...] = ()   # AND of full-match patterns
    regex_search: tuple[str, ...] = ()      # AND of substring patterns
    str_min_len: int | None = None
    str_max_len: int | None = None
    str_exact_len: int | None = None

    # Floats
    allow_nan: bool = False
    allow_infinity: bool = False

    # Catch-all: opaque per-check predicates that we could not aggregate.
    # These are applied as `.filter(...)` at the very end and only on the
    # final element strategy (NOT chained per-check).
    residual_filters: tuple[tuple[str, "Predicate"], ...] = ()

    # Optional adapter-supplied callbacks of type
    # (FieldConstraints) -> FieldConstraints, run after merge() and
    # before compile_field_strategy(). Lets a constraint adapter rewrite
    # itself with knowledge of what sibling checks contributed (e.g.
    # upgrading a residual_filter to an isin set when bounds are known).
    # See §4.4.1 step 4 for a worked example.
    post_merge_hooks: tuple["PostMergeHook", ...] = ()

    def merge(self, other: "FieldConstraints") -> "FieldConstraints":
        """Intersect two FieldConstraints.

        Raises ConstraintConflictError when constraints are jointly
        unsatisfiable (e.g. min_value=5 and max_value=3, or
        eq=10 and notin={10}).
        """
        ...


class ConstraintConflictError(ValueError):
    """Raised when two constraints cannot be jointly satisfied."""


Predicate = Any  # callable producing bool, kept for residual filters
PostMergeHook = Any  # FieldConstraints -> FieldConstraints
```

`FieldConstraints.merge` is a pure function whose semantics are explicit and
testable. The intersection rules are:

| Field | Merge rule |
|---|---|
| `min_value` | `max(left.min_value, right.min_value)` (with `UNSET` ignoring) |
| `max_value` | `min(left.max_value, right.max_value)` |
| `exclude_min` | `OR` (if both are equal-valued, take strictest) |
| `exclude_max` | `OR` |
| `isin` | set intersection (`None` means unconstrained) |
| `notin` | set union |
| `eq` | conflict unless both equal; collapses bounds; if `isin` is set, `eq` must be a member, otherwise `ConstraintConflictError` |
| `regex_fullmatch` | tuple concatenation; AND of patterns |
| `regex_search` | tuple concatenation |
| `str_min_len` | `max(...)` |
| `str_max_len` | `min(...)` |
| `str_exact_len` | conflict unless both equal |
| `allow_nan`/`allow_infinity` | `AND` (any check disallowing wins) |
| `residual_filters` | tuple concatenation |
| `post_merge_hooks` | tuple concatenation; applied left-to-right after `merge` is fully complete, before `compile_field_strategy` |

If after merging `min_value > max_value`, `isin == frozenset()`,
`eq` is also in `notin`, etc., we raise
`ConstraintConflictError`. The caller (the schema strategy) translates
this into a `SchemaDefinitionError` with the same message style as today's
`Unsatisfiable`-rewrap, so users get an actionable error at strategy
construction time rather than after `hypothesis` exhausts itself.

### 4.2 Built-in check → constraint adapters

For each built-in check, a small adapter converts the check's
`statistics` dict to a `FieldConstraints` instance. These live alongside the
old `*_strategy` functions in `pandera/strategies/pandas_strategies.py` (or,
preferably, a new sibling module `pandas_constraints.py`).

```python
# pandera/strategies/pandas_constraints.py

from .constraints import FieldConstraints, UNSET

def eq_constraint(*, value) -> FieldConstraints:
    return FieldConstraints(eq=value)

def ne_constraint(*, value) -> FieldConstraints:
    return FieldConstraints(notin=frozenset([value]))

def gt_constraint(*, min_value) -> FieldConstraints:
    return FieldConstraints(min_value=min_value, exclude_min=True)

def ge_constraint(*, min_value) -> FieldConstraints:
    return FieldConstraints(min_value=min_value, exclude_min=False)

def lt_constraint(*, max_value) -> FieldConstraints:
    return FieldConstraints(max_value=max_value, exclude_max=True)

def le_constraint(*, max_value) -> FieldConstraints:
    return FieldConstraints(max_value=max_value, exclude_max=False)

def in_range_constraint(
    *, min_value, max_value, include_min=True, include_max=True,
) -> FieldConstraints:
    return FieldConstraints(
        min_value=min_value,
        max_value=max_value,
        exclude_min=not include_min,
        exclude_max=not include_max,
    )

def isin_constraint(*, allowed_values) -> FieldConstraints:
    return FieldConstraints(isin=frozenset(allowed_values))

def notin_constraint(*, forbidden_values) -> FieldConstraints:
    return FieldConstraints(notin=frozenset(forbidden_values))

def str_matches_constraint(*, pattern) -> FieldConstraints:
    return FieldConstraints(regex_fullmatch=(pattern,))

def str_contains_constraint(*, pattern) -> FieldConstraints:
    return FieldConstraints(regex_search=(pattern,))

def str_startswith_constraint(*, string) -> FieldConstraints:
    return FieldConstraints(regex_fullmatch=(rf"\A(?:{string}).*\Z",))

def str_endswith_constraint(*, string) -> FieldConstraints:
    return FieldConstraints(regex_fullmatch=(rf"\A.*(?:{string})\Z",))

def str_length_constraint(
    *, min_value=None, max_value=None, exact_value=None,
) -> FieldConstraints:
    return FieldConstraints(
        str_min_len=min_value,
        str_max_len=max_value,
        str_exact_len=exact_value,
    )
```

These functions are *registered* against built-in checks via a new
constraint dispatcher (described in §4.4).

### 4.3 Compiling `FieldConstraints` to a hypothesis strategy

A new function `compile_field_strategy` in
`pandera/strategies/pandas_strategies.py` takes a dtype and a
`FieldConstraints` and returns a single `SearchStrategy` with no chained
filters (modulo `residual_filters`).

**Design decision (see §9.1):** `compile_field_strategy` delegates all
dtype-specific bridging to the existing `pandas_dtype_strategy(...)`
helper. It only computes which kwargs to forward; it does not duplicate
the datetime/timedelta/datetime-tz/complex/numpy/from_dtype dispatch
logic that already lives in `pandas_dtype_strategy`.

```python
def compile_field_strategy(
    pandera_dtype, constraints: FieldConstraints,
) -> SearchStrategy:
    """Build a hypothesis strategy from merged constraints in one go.

    All dtype-specific logic (datetime tz, complex, time resolutions,
    surrogate handling) is delegated to ``pandas_dtype_strategy``.
    This function is responsible only for translating the merged
    ``FieldConstraints`` into kwargs/parent-strategy arguments.
    """

    # 1. Equality short-circuits everything; collapse to a single value.
    if constraints.eq is not UNSET:
        if constraints.eq in constraints.notin:
            raise ConstraintConflictError(
                f"eq={constraints.eq!r} conflicts with notin={constraints.notin!r}"
            )
        return pandas_dtype_strategy(pandera_dtype, st.just(constraints.eq))

    # 2. Membership with bounds: prune allowed set against bounds/notin
    # *before* sampling, so we never .filter on the strategy.
    if constraints.isin is not None:
        allowed = constraints.isin - constraints.notin
        if constraints.min_value is not UNSET:
            op = operator.lt if constraints.exclude_min else operator.le
            allowed = {v for v in allowed if op(constraints.min_value, v)}
        if constraints.max_value is not UNSET:
            op = operator.gt if constraints.exclude_max else operator.ge
            allowed = {v for v in allowed if op(constraints.max_value, v)}
        if not allowed:
            raise ConstraintConflictError(
                "isin/notin/bounds intersection is empty"
            )
        # Delegate dtype bridging to pandas_dtype_strategy.
        return pandas_dtype_strategy(
            pandera_dtype, st.sampled_from(sorted(allowed)),
        )

    # 3. Strings: route to the regex/length compiler (§9.2).
    if _is_string_dtype(pandera_dtype):
        return _compile_string_strategy(pandera_dtype, constraints)

    # 4. Numeric / temporal: build the kwargs dict for pandas_dtype_strategy.
    # pandas_dtype_strategy already knows how to translate (min_value,
    # max_value, exclude_min, exclude_max, allow_nan, allow_infinity)
    # into npst.from_dtype / numpy_time_dtypes / numpy_complex_dtypes.
    #
    # Two dtype-gating subtleties (mirroring today's behaviour in
    # gt_strategy / lt_strategy):
    #
    #   - exclude_min / exclude_max: hypothesis only supports these for
    #     floats and complex. For integer/temporal dtypes we lower them
    #     to a closed-bound representation (min_value=v, exclude=True
    #     becomes min_value=v+1, exclude=False) so the kwargs we forward
    #     are always within st.integers / numpy_time_dtypes' contract.
    #
    #   - allow_nan / allow_infinity: only meaningful for floats and
    #     complex. For other dtypes we omit them entirely; pandas_dtype
    #     _strategy already drops them via compat_kwargs, but we keep
    #     compile_field_strategy's output minimal.
    kwargs: dict = {}
    if constraints.min_value is not UNSET:
        min_v, excl_min = _close_bound(
            pandera_dtype, constraints.min_value,
            constraints.exclude_min, side="min",
        )
        kwargs["min_value"] = min_v
        if is_float(pandera_dtype) or is_complex(pandera_dtype):
            kwargs["exclude_min"] = excl_min
    if constraints.max_value is not UNSET:
        max_v, excl_max = _close_bound(
            pandera_dtype, constraints.max_value,
            constraints.exclude_max, side="max",
        )
        kwargs["max_value"] = max_v
        if is_float(pandera_dtype) or is_complex(pandera_dtype):
            kwargs["exclude_max"] = excl_max
    if is_float(pandera_dtype) or is_complex(pandera_dtype):
        kwargs["allow_nan"] = constraints.allow_nan
        kwargs["allow_infinity"] = constraints.allow_infinity

    strat = pandas_dtype_strategy(pandera_dtype, **kwargs)

    # 5. notin (only if isin path was not taken): single membership filter.
    if constraints.notin:
        forbidden = constraints.notin
        strat = strat.filter(lambda v, f=forbidden: v not in f)

    # 6. Residual opaque predicates from custom checks that did not
    # provide a constraint adapter. There is at most one filter per
    # such check, never per built-in check.
    for _name, predicate in constraints.residual_filters:
        strat = strat.filter(predicate)

    return strat
```

`_compile_string_strategy` is specified in §9.2: it folds every
`regex_fullmatch` and `regex_search` pattern into a single
`st.from_regex(...)` call via lookahead-AND, applies merged length
bounds, and falls back to a single `.filter(...)` only when a pattern
fails `re.compile`.

### 4.4 New extension hook: `register_check_constraint`

We introduce a *new* registration mechanism that is additive — old
strategies keep working. A new dispatcher, parallel to
`STRATEGY_DISPATCHER`, is added to `base_strategies.py`:

```python
# pandera/strategies/base_strategies.py (additive)

from collections.abc import Callable

CONSTRAINT_DISPATCHER: dict[tuple[str, type], Callable] = {}
"""(check_name, data_type) -> callable producing a FieldConstraints."""
```

`register_builtin_check` is extended with an optional `constraint=` kwarg:

```python
# pandera/api/extensions.py

def register_builtin_check(
    fn=None,
    strategy: Callable | None = None,
    constraint: Callable | None = None,   # NEW
    _check_cls: type = Check,
    aliases=None,
    **outer_kwargs,
):
    ...
    if strategy is not None:
        for dt in data_types:
            STRATEGY_DISPATCHER[(name, dt)] = strategy
    if constraint is not None:                       # NEW
        for dt in data_types:
            CONSTRAINT_DISPATCHER[(name, dt)] = constraint
    ...
```

`register_check_method` (the public extensions API) is extended with the
same optional `constraint=` kwarg:

```python
def register_check_method(
    check_fn=None, *, statistics=None, supported_types=None,
    check_type="vectorized", strategy=None, constraint=None,  # NEW
):
    ...
```

`pandera/strategies/pandas_strategies.py` gets a sibling decorator to
`register_check_strategy`:

```python
def register_check_constraint(constraint_fn):
    """Decorate a Check method with a constraint function.

    The constraint_fn receives the check's statistics as kwargs and
    returns a FieldConstraints. When present, the strategy builder will
    prefer the constraint over check.strategy and merge it with siblings.
    """
    def _decorator(class_method):
        @wraps(class_method)
        def _wrapper(cls, *args, **kwargs):
            check = class_method(cls, *args, **kwargs)
            if check.statistics is None:
                raise AttributeError(...)
            check.constraint = constraint_fn       # NEW Check attribute
            return check
        return _wrapper
    return _decorator
```

`Check.__init__` gains a new attribute `self.constraint = constraint`
(default `None`). `Check.constraint` is the *only* new optional argument
on `pa.Check(...)`. Every existing call site keeps working.

### 4.4.1 Worked example: a custom `divisible_by` check

To make the contract concrete, the following is a realistic
end-to-end example showing (1) how a third-party library author
writes a constraint adapter, (2) how it composes with built-in
constraints on the same column, and (3) the exact
`FieldConstraints` value and hypothesis strategy that get produced
versus what happens on `main` today.

#### The use case

A user is validating financial transaction amounts (in cents) and
wants every value to be a positive multiple of 5 cents, less than
$1000. Today they would write:

```python
import pandera.pandas as pa
from pandera.api.checks import Check

schema = pa.DataFrameSchema({
    "amount_cents": pa.Column(int, checks=[
        Check.gt(0),
        Check.lt(100_000),
        Check(
            lambda s: (s % 5) == 0,
            name="divisible_by_5",
            error="must be a multiple of 5 cents",
        ),
    ]),
})

schema.example(size=100)   # slow; can hit Unsatisfiable
```

On `main`, this generates an unbounded `int`, applies a
`>0` filter, an `<100_000` filter, *and* a `(x % 5) == 0` filter.
hypothesis discards roughly 80% of draws on the modulus filter
alone, and the `Unsatisfiable` health check fires for `size > 50`
on a fresh seed.

#### Step 1 — define the check and constraint adapter (library code)

```python
# myproject/checks.py
import pandera.pandas as pa
from pandera.api.extensions import register_check_method
from pandera.strategies.constraints import FieldConstraints


def _divisible_by_constraint(*, divisor: int) -> FieldConstraints:
    """Adapter: emit a FieldConstraints describing 'multiple of N'.

    'divisible by N' is not expressible as min/max/isin/regex, but
    it IS still a single predicate per field. The right place for
    it is the residual_filters channel: it gets folded into the
    base strategy as exactly one .filter call, never chained with
    other built-in checks' filters.

    A more sophisticated adapter could narrow the strategy further
    by intersecting with bounds when both are present (see Step 4).
    """
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    return FieldConstraints(
        residual_filters=(
            (f"divisible_by({divisor})", lambda x, d=divisor: x % d == 0),
        ),
    )


@register_check_method(
    statistics=["divisor"],
    check_type="element_wise",
    constraint=_divisible_by_constraint,   # NEW opt-in path
)
def divisible_by(value, *, divisor: int) -> bool:
    return (value % divisor) == 0
```

The library author registered the check via the *existing*
`register_check_method` decorator and passed the *new* optional
`constraint=` kwarg. No `strategy=` is supplied, so the legacy
`STRATEGY_DISPATCHER` entry is empty and the constraint path is
the only path the new aggregator considers.

A user who is unable or unwilling to use `register_check_method`
(for example, because they want to attach the adapter to an
existing in-line `pa.Check(...)`) can do the same thing
directly:

```python
from pandera.strategies.constraints import FieldConstraints

modulus_check = pa.Check(
    lambda s: (s % 5) == 0,
    name="divisible_by_5",
    error="must be a multiple of 5 cents",
)
modulus_check.constraint = lambda **_: _divisible_by_constraint(divisor=5)
```

`Check.constraint` is a public attribute and can be set after
construction; this is the migration path for users who have
in-line `pa.Check(...)` instances they don't control the
construction of.

#### Step 2 — use it alongside built-in checks

The schema definition is unchanged from the user's perspective:

```python
# myproject/schema.py
from myproject.checks import divisible_by   # noqa: F401 (registers Check.divisible_by)

schema = pa.DataFrameSchema({
    "amount_cents": pa.Column(int, checks=[
        Check.gt(0),
        Check.lt(100_000),
        Check.divisible_by(divisor=5),
    ]),
})
```

#### Step 3 — what the aggregator produces

`field_element_strategy` walks the three checks and consults each
one's constraint source (`check.constraint`, then
`CONSTRAINT_DISPATCHER[(check.name, pd.Series)]`):

| Check | Constraint source | Contributes to `FieldConstraints` |
|---|---|---|
| `Check.gt(0)` | built-in `gt_constraint` (§4.2) | `min_value=0, exclude_min=True` |
| `Check.lt(100_000)` | built-in `lt_constraint` (§4.2) | `max_value=100_000, exclude_max=True` |
| `Check.divisible_by(5)` | user `_divisible_by_constraint` (above) | `residual_filters=(("divisible_by(5)", <lambda>),)` |

After `merge`, the accumulated `FieldConstraints` is:

```python
FieldConstraints(
    min_value=0,
    max_value=100_000,
    exclude_min=True,
    exclude_max=True,
    residual_filters=(
        ("divisible_by(5)", <lambda x: x % 5 == 0>),
    ),
)
```

`compile_field_strategy(pa.Int(), constraints)` produces:

```python
# Effectively (one from_dtype call + one residual filter):
strat = pandas_dtype_strategy(
    pa.Int(),
    min_value=0, max_value=100_000,
    exclude_min=True, exclude_max=True,
    allow_nan=False, allow_infinity=False,
)
strat = strat.filter(lambda x: x % 5 == 0)   # the only filter
```

Compared with `main`:

| | `main` | this spec |
|---|---|---|
| `npst.from_dtype` calls | 1 (unbounded int) | 1 (bounded `(0, 100_000)`) |
| `.filter` nodes | 3 (`>0`, `<100_000`, `% 5`) | 1 (`% 5`) |
| Discard rate per draw | ~80% | ~80% (modulus is genuinely 1-in-5) |
| `Unsatisfiable` at `size=100` | frequent | none observed |
| Wall-clock for `example(size=100)` | seconds, often errors | tens of milliseconds |

The win is *not* that the modulus filter goes away — it can't, the
constraint is genuinely 1-in-5 — but that the bound filters fold
into the `from_dtype` call, the bounded space is 5× denser in
valid integers, and hypothesis's shrinker has a much smaller
search space to begin with.

#### Step 4 — a stronger adapter that aggregates with bounds

A library author who wants to push further can write a smarter
adapter that *narrows the bounds* against the modulus when both
are present. For this we expose a new convenience hook on
`FieldConstraints`:

```python
from dataclasses import replace
from pandera.strategies.constraints import FieldConstraints, UNSET


def _round_up(value: int, divisor: int, strict: bool) -> int:
    """Smallest multiple of ``divisor`` that is > value (strict) or >= value."""
    base = value + 1 if strict else value
    rem = base % divisor
    return base if rem == 0 else base + (divisor - rem)


def _round_down(value: int, divisor: int, strict: bool) -> int:
    """Largest multiple of ``divisor`` that is < value (strict) or <= value."""
    base = value - 1 if strict else value
    return base - (base % divisor)


def _divisible_by_constraint_v2(*, divisor: int) -> FieldConstraints:
    return FieldConstraints(
        residual_filters=(
            (f"divisible_by({divisor})", lambda x, d=divisor: x % d == 0),
        ),
        # Hint to the compiler: when (min_value, max_value) are also
        # present after merge, prefer st.sampled_from(range(...))
        # over from_dtype + filter.
        post_merge_hooks=(
            lambda c, d=divisor: (
                replace(
                    c,
                    isin=frozenset(range(
                        _round_up(c.min_value, d, c.exclude_min),
                        _round_down(c.max_value, d, c.exclude_max) + 1,
                        d,
                    )),
                    residual_filters=tuple(
                        rf for rf in c.residual_filters
                        if not rf[0].startswith("divisible_by(")
                    ),
                )
                if c.min_value is not UNSET
                and c.max_value is not UNSET
                else c
            ),
        ),
    )
```

`post_merge_hooks` is an optional channel on `FieldConstraints`
(default empty tuple) that runs *after* the `merge` step and
*before* `compile_field_strategy`, giving an adapter the chance
to rewrite the merged constraint set with knowledge of what
sibling checks contributed. With this v2 adapter, the example
above compiles to:

```python
strat = pandas_dtype_strategy(
    pa.Int(),
    st.sampled_from(sorted(range(5, 100_000, 5))),
)
# zero filters; every draw is valid by construction.
```

This pattern — start with a residual filter, optionally upgrade to
a stronger constraint via `post_merge_hooks` once siblings are
known — is the recommended idiom for non-decomposable predicates.
It is opt-in: the simpler v1 adapter is also fully supported and
already a strict win over the legacy `.filter`-only path.

#### Step 5 — interaction with another user-defined constraint

Two user-defined constraint adapters compose the same way as
built-ins. Suppose a second team adds a `Check.percentile_clipped`
check that constrains values to lie within the empirical
1st–99th percentile of a reference distribution loaded at module
import time:

```python
@register_check_method(
    statistics=["ref_low", "ref_high"],
    check_type="vectorized",
    constraint=_percentile_clipped_constraint,   # NEW opt-in path
)
def percentile_clipped(series, *, ref_low, ref_high):
    return series.between(ref_low, ref_high)


def _percentile_clipped_constraint(*, ref_low, ref_high) -> FieldConstraints:
    return FieldConstraints(min_value=ref_low, max_value=ref_high)
```

Stacked on the same column:

```python
pa.Column(int, checks=[
    Check.gt(0),                               # min_value=0,    excl
    Check.lt(100_000),                         # max_value=100k, excl
    Check.divisible_by(divisor=5),             # residual + (v2: isin)
    Check.percentile_clipped(ref_low=50, ref_high=90_000),  # min=50, max=90k
])
```

`merge` intersects the bounds (taking `max(0, 50) = 50` for
`min_value` and `min(100_000, 90_000) = 90_000` for `max_value`),
preserves the modulus residual filter, and — under the v2 adapter
— produces:

```python
FieldConstraints(
    min_value=50,
    max_value=90_000,
    exclude_min=True,
    exclude_max=True,
    isin=frozenset(range(55, 90_000, 5)),  # rewritten by post_merge_hook
    residual_filters=(),                    # hook removed it
)
```

which `compile_field_strategy` lowers to a single
`st.sampled_from(...)` call. **Two independent third-party
authors, neither aware of the other, get the maximally tight
strategy by composing through `FieldConstraints`** — no
`.filter`-chain, no `Unsatisfiable`, no ordering dependency. This
is the property that the refactor exists to enable.

#### Step 6 — what happens if a user adapter conflicts with a built-in

If `Check.percentile_clipped(ref_low=200_000, ref_high=300_000)`
is added to the same column, `merge` produces
`min_value=200_000, max_value=100_000` and immediately raises
`ConstraintConflictError`, which `field_element_strategy`
re-raises as `SchemaDefinitionError` (§6) with a message that
names both contributing checks:

```
SchemaDefinitionError: Cannot construct a data-generation strategy
for column 'amount_cents' with checks
[gt(0), lt(100000), divisible_by(divisor=5),
 percentile_clipped(ref_low=200000, ref_high=300000)]:
constraints are jointly unsatisfiable
(min_value=200000 > max_value=100000;
 contributed by 'percentile_clipped' and 'lt').
```

The user gets an actionable error at `Schema.strategy()` /
`Schema.example()` call time, not several seconds later from
`hypothesis.errors.Unsatisfiable`.

### 4.5 New core dispatcher: `field_element_strategy` v2

The function in `pandas_strategies.py` is rewritten to *bucket* checks into
"constraint-providing" and "filter-only", aggregate the former, and emit a
single strategy:

```python
def field_element_strategy(pandera_dtype, strategy=None, *, checks=None):
    if strategy is not None:
        # Legacy path: caller supplied a base strategy. We keep the old
        # behaviour for backwards compatibility (e.g. user is composing
        # strategies manually).
        return _legacy_field_element_strategy(
            pandera_dtype, strategy, checks=checks,
        )

    checks = list(checks or [])
    constraint_acc = FieldConstraints()
    legacy_strategies: list[tuple[Check, Callable]] = []
    residuals: list[tuple[str, Callable]] = []

    for check in checks:
        # 1. Prefer explicit constraint adapter (new API).
        constraint_fn = (
            getattr(check, "constraint", None)
            or CONSTRAINT_DISPATCHER.get((check.name, pd.Series))
        )
        if constraint_fn is not None:
            constraint_acc = constraint_acc.merge(
                constraint_fn(**check.statistics)
            )
            continue

        # 2. Legacy strategy (built-in or user-supplied via Check(strategy=)).
        legacy_strategy = (
            check.strategy
            or STRATEGY_DISPATCHER.get((check.name, pd.Series))
        )
        if legacy_strategy is not None:
            legacy_strategies.append((check, legacy_strategy))
            continue

        # 3. Element-wise opaque predicate.
        if check.element_wise:
            residuals.append((check.name, check._check_fn))

        # vectorized opaque checks are handled by series/dataframe layer

    if residuals:
        constraint_acc = replace(
            constraint_acc,
            residual_filters=constraint_acc.residual_filters + tuple(residuals),
        )

    elements = compile_field_strategy(pandera_dtype, constraint_acc)

    # Apply legacy strategies as filter-chains *after* the merged base.
    # This preserves the documented behaviour for users who currently
    # rely on Check(strategy=...). Emit a DeprecationWarning when a
    # legacy strategy that advertises base-mode support is being forced
    # into chained mode by the presence of built-in constraints (§9.3).
    has_aggregated_constraints = (
        constraint_acc != FieldConstraints()
    )
    for check, legacy_strategy in legacy_strategies:
        if (
            has_aggregated_constraints
            and _strategy_supports_base_mode(legacy_strategy)
        ):
            _warn_legacy_strategy_chained_once(check, legacy_strategy)
        elements = legacy_strategy(
            pandera_dtype, elements, **check.statistics
        )

    return elements
```

This is the only place where filter chaining can still happen. With all
built-in checks migrated to constraints (§4.6), filter chaining for
built-ins disappears entirely, and the legacy path is reserved for
user-supplied checks that don't (yet) provide a constraint adapter.

`_strategy_supports_base_mode` and `_warn_legacy_strategy_chained_once`
are helpers defined alongside `field_element_strategy` in
`pandas_strategies.py`; their full specification is in §9.3.

### 4.6 Migrating built-in checks

`pandera/backends/pandas/builtin_checks.py` already declares strategies via
`@register_builtin_check(strategy=...)`. We add `constraint=` to each entry:

```python
@register_builtin_check(
    aliases=["gt"],
    strategy=st.gt_strategy,             # KEEP for backward compat
    constraint=cn.gt_constraint,         # NEW, preferred
    error="greater_than({min_value})",
)
def greater_than(data, min_value): ...
```

Built-in checks to migrate (full list, all in `builtin_checks.py`):
`equal_to`, `not_equal_to`, `greater_than`, `greater_than_or_equal_to`,
`less_than`, `less_than_or_equal_to`, `in_range`, `isin`, `notin`,
`str_matches`, `str_contains`, `str_startswith`, `str_endswith`,
`str_length`. (`unique_values_eq` has no strategy today and is not in
scope.)

After migration, `field_element_strategy` for any column whose checks are
all built-ins emits exactly one hypothesis strategy call, with no
`.filter` in the chain.

### 4.7 Backwards compatibility matrix

| User-facing surface | Status under this spec |
|---|---|
| `Schema.strategy(size=...)` / `Schema.example(...)` | unchanged |
| `pa.Check.gt(0)`, `pa.Check.isin(...)`, etc. | unchanged behaviour, faster generation |
| `pa.Check(check_fn, strategy=fn)` | unchanged; `fn` is invoked as today on the merged base strategy |
| `pa.Check(check_fn)` with no strategy (in-line custom) | unchanged; falls through to `undefined_check_strategy` filter as today |
| `pa.extensions.register_check_method(strategy=fn)` | unchanged; `fn` continues to receive `(pandera_dtype, strategy, **stats)` |
| `pa.extensions.register_check_method(constraint=fn)` | **new** opt-in for faster generation |
| `pandera.strategies.pandas_strategies.register_check_strategy(fn)` | unchanged |
| `pandera.strategies.pandas_strategies.register_check_constraint(fn)` | **new** |
| `pandera.strategies.pandas_strategies.gt_strategy` (and all `*_strategy` peers) | unchanged signatures, still importable |
| `pandera.strategies.base_strategies.STRATEGY_DISPATCHER` | unchanged |
| `pandera.strategies.base_strategies.CONSTRAINT_DISPATCHER` | **new** |
| `Check.strategy` attribute | unchanged |
| `Check.constraint` attribute | **new**, default `None` |
| Documented "first check is the base strategy" semantics | softened in docs but not broken: when only built-ins are used, ordering becomes irrelevant; when a user supplies `Check(strategy=...)` it still runs after the merged built-ins. A `DeprecationWarning` is raised once-per-process per `(check.name, fn id)` when a `Check(strategy=fn)` whose `fn` advertises base-mode support (i.e. its `strategy` parameter has a `None` default) is forced into chained mode by sibling built-in checks. See §9.3. |

Existing tests in `tests/strategies/test_strategies.py` and
`tests/pandas/test_extensions.py` exercise (a) the per-`*_strategy`
function signatures, (b) `register_check_strategy` on a custom check, and
(c) end-to-end schema strategies. **All of these must continue to pass
without modification**; that is the bar for "no API change".

### 4.8 Container-level strategies

`series_strategy`, `column_strategy`, `index_strategy`,
`dataframe_strategy`, and `multiindex_strategy` change in only one way:
they delegate to the new `field_element_strategy` (which already exists,
just with a new internal implementation) instead of doing per-check fold
logic themselves. Specifically:

- `series_strategy` and `index_strategy` already call
  `field_element_strategy`; no API change. The "vectorized check fallback"
  loop after the strategy is built (lines 935–943) stays.
- `column_strategy` already calls `field_element_strategy`; no change.
- `dataframe_strategy.make_row_strategy` is rewritten to use the same
  bucketing logic for dataframe-level checks, replacing the explicit
  per-check fold at lines 1094–1118. Constraint adapters registered at
  `(check_name, pd.DataFrame)` receive the row dict (`dict[str, Any]`)
  not a single element, and produce a `RowConstraints` value (a thin
  parallel to `FieldConstraints` that holds per-column overrides plus
  a residual row-level predicate channel). For the v1 of this refactor,
  no built-in check uses the `(name, pd.DataFrame)` constraint slot —
  every built-in is column-scoped — so `RowConstraints` exists only as
  the extension point. The legacy `STRATEGY_DISPATCHER[(name,
  pd.DataFrame)]` path remains the default for dataframe-level checks
  and will be migrated lazily in a follow-up. This keeps the v1 PR
  series strictly column-scoped and avoids re-litigating the
  row-vs-column-scope semantics of dataframe-level checks.

`null_field_masks`, `null_dataframe_masks`, `set_pandas_index`,
`convert_dtype`, `convert_dtypes`, `numpy_time_dtypes`,
`numpy_complex_dtypes`, `pandas_dtype_strategy`, surrogate handling
(`_remove_surrogates`, `_str_no_surrogates`) remain unchanged.

### 4.9 Caveats and known limitations

The refactor is deliberately scoped; a few things it does *not* do, and
a few sharp edges callers should be aware of:

1. **In-line `pa.Check(lambda ...)` checks remain filter-based.** The
   only way for an opaque user predicate to participate in the
   aggregator is to either (a) attach a `constraint=` adapter to the
   `Check`, or (b) provide one via `register_check_method(constraint=)`
   / `register_check_constraint`. Without one, the predicate falls
   through to `residual_filters` and the legacy
   `undefined_check_strategy` warning continues to fire. This is the
   §2.2 non-goal made explicit.
2. **Hypothesis `from_regex` lookahead is a generative heuristic.**
   §9.2's lookahead-AND combiner produces a strictly more
   *specifiable* strategy than today's chained `.filter`s, but
   hypothesis's regex generator handles lookaheads via internal
   rejection sampling. For pathological combinations (e.g. five
   mutually-restrictive `str_contains` patterns) the combined
   `from_regex` may itself be slow. The spec's claim is that this is
   *no worse than today* and almost always faster; it is not a
   guarantee that lookahead is free. Benchmarks in §10.3.3 quantify
   this on a representative case.
3. **`compile_field_strategy` is element-scoped.** It produces a
   strategy for a *single value* of the field's dtype. Container-level
   concerns (`unique=True`, `nullable=True`, index alignment, dataframe
   row count) are still handled by the existing
   `series_strategy` / `column_strategy` / `dataframe_strategy`
   wrappers and are unaffected by the constraint aggregator.
4. **`unique_values_eq` is intentionally out of scope.** It has no
   `strategy=` today and has set-of-allowed-values semantics that are
   most cleanly expressed as `isin(...)` plus a uniqueness flag at the
   container level; folding it into `FieldConstraints` would conflate
   element-level and series-level semantics.
5. **Dataframe-level constraint adapters are deferred.** §4.8 keeps
   `(name, pd.DataFrame)` checks on the legacy `STRATEGY_DISPATCHER`
   path for v1. The constraint aggregator is column-scoped only.
6. **`Check.constraint` is a public, mutable attribute.** Setting it
   after `Schema.strategy()` has been called for that check has no
   effect (the strategy is constructed at `.strategy()` time, not at
   Check definition time, but cached at the schema level by the
   container layer). Users wiring constraints via the
   "set after construction" idiom in §4.4.1 step 1 should do so at
   schema-definition time, not lazily.
7. **No automatic bridging of `pa.Check.<name>(strategy=...)` keyword
   to `constraint=`.** Even when a check's `strategy=fn` could in
   principle be expressed as a constraint adapter, the spec does not
   try to derive one. The migration path is explicit (write the
   adapter and pass `constraint=fn`) which keeps semantics legible.

---

## 5. xarray strategies

`pandera/strategies/xarray_strategies.py` accepts `checks` arguments on
`data_array_strategy` and `data_array_schema_strategy` but currently
ignores them:

```142:155:pandera/strategies/xarray_strategies.py
:param checks: (unused) reserved for future check-aware generation.
```

The same `FieldConstraints` machinery is applied here:

1. A new function `compile_dataarray_element_strategy(np_dtype, constraints)`
   wraps `npst.from_dtype(np_dtype, **kwargs_from_constraints)` analogously
   to `compile_field_strategy`.
2. `data_array_strategy(... checks=...)` aggregates the `checks` list into a
   `FieldConstraints`, then passes the resulting per-element strategy as
   `elements=` to `npst.arrays(...)`. This is a single composed strategy,
   not a chain of `.filter`s on the arrays themselves.
3. `dataset_strategy(data_vars=...)` accepts an optional `checks` key per
   data var (already supported by the schema layer; see
   `dataset_schema_strategy` in `xarray_strategies.py`) and routes them
   through the same compile step.
4. `data_array_schema_strategy` and `dataset_schema_strategy` populate the
   `checks` argument from `schema.checks` (and per-`DataVar.checks`)
   instead of dropping them.

User-facing API on `DataArraySchema.strategy` / `DatasetSchema.strategy`
remains unchanged; the only difference is that checks now actually
constrain the generated data.

---

## 6. Error handling

`ConstraintConflictError` is caught in `field_element_strategy` and
re-raised as `pandera.errors.SchemaDefinitionError` with a message of the
form:

```
Cannot construct a data-generation strategy for column 'x' with checks
[gt(10), lt(5)]: constraints are jointly unsatisfiable
(min_value=10 > max_value=5).
```

This replaces the much-later `hypothesis.errors.Unsatisfiable` that
currently surfaces from the same situation. The behaviour for
*satisfiable but hard-to-sample* constraint stacks (e.g. an inline
`Check(lambda s: s.isin({"foo","bar"}))` on a `str` column) is unchanged:
they still go through `undefined_check_strategy` with its existing
warning.

---

## 7. Implementation plan

The rollout is split into small PR-sized chunks so that each step is
independently reviewable and shippable.

1. **Constraints scaffolding (no behaviour change).**
   - Add `pandera/strategies/constraints.py` with `FieldConstraints`,
     `UNSET`, `ConstraintConflictError`, and `merge`.
   - Unit-test `merge` exhaustively in `tests/strategies/test_constraints.py`.
2. **Constraint dispatcher (no behaviour change).**
   - Add `CONSTRAINT_DISPATCHER` to `base_strategies.py`.
   - Add `constraint=` kwarg to `register_builtin_check` and to
     `register_check_method` (no migrations yet).
   - Add `register_check_constraint` decorator to `pandas_strategies.py`.
   - Add `Check.constraint` attribute, default `None`.
3. **Compile step (no behaviour change yet).**
   - Add `compile_field_strategy(pandera_dtype, constraints)` to
     `pandas_strategies.py`.
   - Unit-test it with hand-built `FieldConstraints` values.
4. **Switch `field_element_strategy` to bucketing.**
   - Replace the body with the new logic from §4.5.
   - Constraint dispatcher is empty at this point, so behaviour is
     identical to today (every check still goes down the legacy strategy
     path).
   - Run the full `tests/strategies/` suite — must pass unmodified.
5. **Migrate built-in checks one family at a time.**
   - Numeric: `eq`, `ne`, `gt`, `ge`, `lt`, `le`, `in_range` → add
     `constraint=` arg. After each migration, all existing tests must
     still pass and a new "no-filter assertion" test verifies the strategy
     contains zero `.filter` operators in the resulting strategy graph
     for those check stacks.
   - Membership: `isin`, `notin`.
   - String: `str_matches`, `str_contains`, `str_startswith`,
     `str_endswith`, `str_length`. The string migration also lands
     `_compile_string_strategy` and the lookahead-AND combiner from §9.2.
6. **Apply same machinery to `dataframe_strategy.make_row_strategy`.**
7. **Deprecation warning for forced-chained legacy strategies (§9.3).**
   - Add `_strategy_supports_base_mode` and
     `_warn_legacy_strategy_chained_once` helpers.
   - Wire the warning into `field_element_strategy` and
     `make_row_strategy`.
   - Add tests covering each non-warning case (no built-ins present;
     `fn` does not support base mode; user migrated to `constraint=`)
     and the positive case.
8. **xarray.**
   - Wire `checks` into `data_array_strategy` and `dataset_strategy`.
   - Add `tests/strategies/test_xarray_strategies.py` cases that previously
     could not pass because checks were ignored.
9. **Docs.**
   - Update [`data_synthesis_strategies.md`](../docs/source/data_synthesis_strategies.md):
     - Soften the "first check wins" guidance under
       `(check-strategy-chaining)=` to: "When all your checks are
       built-ins, ordering is irrelevant. When mixing custom
       `Check(strategy=...)` with built-ins, custom strategies run after
       the merged built-ins."
     - Document `register_check_constraint` and
       `register_check_method(constraint=...)`.
     - Document the §9.3 `DeprecationWarning` and the migration path to
       constraint adapters.
   - Add a "Performance notes" subsection comparing before/after.
10. **Optional follow-up (out of scope for first PR sequence):**
    migrate `Check(strategy=...)` users in the wider community by writing
    constraint adapters for popular custom checks.

---

## 8. Testing strategy

- All existing tests in `tests/strategies/test_strategies.py` and
  `tests/pandas/test_extensions.py` must pass unmodified.
- New `tests/strategies/test_constraints.py`:
  - `merge` is associative and commutative for compatible constraints.
  - `merge` raises `ConstraintConflictError` for each conflict class
    (numeric inversion, empty `isin`, `eq` ∈ `notin`, conflicting
    `str_exact_len`).
  - Round-trip: every built-in check's constraint adapter, fed its own
    `statistics`, produces a `FieldConstraints` whose `compile_field_strategy`
    output draws values that pass the original check.
- New `tests/strategies/test_no_filter_chain.py`:
  - For each combination of built-in checks (numeric, string, membership),
    introspect the resulting hypothesis `SearchStrategy`'s `.wrapped_strategy`
    chain and assert no `FilteredStrategy` is present (or assert at most
    one for `notin`).
  - Specifically for strings (§9.2): assert that
    `Check.str_startswith("foo") & Check.str_endswith("bar")
    & Check.str_length(min_value=10)` produces a strategy whose regex
    portion contains zero `FilteredStrategy` nodes (one trailing length
    filter is acceptable).
  - Microbenchmark (smoke, not perf-gated): generating 100 rows with
    `Check.gt(0) & Check.lt(1e10) & Check.notin([-100,-10,0])` should
    finish well under the previous baseline (the example used in our docs
    today).
- New `tests/strategies/test_legacy_strategy_warning.py` (covers §9.3):
  - Asserts the `DeprecationWarning` fires exactly once per
    `(check.name, fn id)` pair when a `Check(strategy=fn)` whose `fn`
    has a `strategy=None` parameter is mixed with built-in checks.
  - Asserts no warning when only the legacy custom check is present
    (no built-ins → legacy ordering still works).
  - Asserts no warning when `fn`'s `strategy` parameter has no default
    (pure chained-mode strategy).
  - Asserts no warning when the user migrated to `Check(constraint=fn)`
    or `register_check_constraint(...)`.
- New `tests/strategies/test_xarray_checks.py`:
  - `data_array_schema_strategy` with `Check.gt(0)` produces only
    positive values.
  - `dataset_schema_strategy` with per-DataVar checks honours them.

---

## 9. Resolved design decisions

The three open questions from earlier drafts of this spec have been
resolved as follows. They are listed here so the rationale stays visible
to readers; the bodies of §4.3, §4.5, and §4.7 already reflect these
choices.

### 9.1 Delegate non-trivial dtypes to `pandas_dtype_strategy`

**Decision:** `compile_field_strategy` does **not** re-implement
hypothesis bridging for datetime, datetime-with-tz, timedelta, complex,
or category dtypes. It builds a single kwargs dict from the merged
`FieldConstraints` and hands it to the existing
`pandas_dtype_strategy(pandera_dtype, **kwargs)` helper, which already
dispatches to `numpy_time_dtypes` / `numpy_complex_dtypes` /
`_datetime_strategy` / `npst.from_dtype` as appropriate. This keeps the
spec narrowly focused on the constraint-aggregation problem and
guarantees that no datetime/tz/complex behaviour regresses.

The kwargs translation table is:

| `FieldConstraints` field | `pandas_dtype_strategy` kwarg |
|---|---|
| `min_value` (numeric/temporal, not UNSET) | `min_value`, with closed-bound translation for integer/temporal dtypes (see §4.3 step 4) |
| `max_value` (numeric/temporal, not UNSET) | `max_value`, with closed-bound translation for integer/temporal dtypes |
| `exclude_min` | forwarded as `exclude_min` only when `is_float(pandera_dtype) or is_complex(pandera_dtype)`; otherwise lowered into `min_value` |
| `exclude_max` | forwarded as `exclude_max` only for float/complex; otherwise lowered into `max_value` |
| `allow_nan` | forwarded only for float/complex (the dtypes hypothesis supports it on); omitted otherwise |
| `allow_infinity` | same dtype gating as `allow_nan` |
| `eq` (not UNSET) | passed as `st.just(value)` parent strategy |
| `isin` (not None) | passed as `st.sampled_from(sorted(allowed))` parent |
| `notin` (after `isin` resolution) | applied as a single trailing `.filter` |
| string fields | routed to `_compile_string_strategy` (§9.2) |

`_close_bound(pandera_dtype, value, exclude, side)` is a small helper
defined alongside `compile_field_strategy`. For float/complex dtypes
it returns `(value, exclude)` unchanged. For integer dtypes it returns
`(value + 1, False)` when `side == "min" and exclude`, `(value - 1,
False)` when `side == "max" and exclude`, else `(value, False)`. For
datetime/timedelta dtypes it lowers exclusive bounds by one nanosecond
(matching pandas resolution) so that `numpy_time_dtypes`'s closed-bound
contract is honoured. This mirrors the dtype-gating already encoded in
today's `gt_strategy` / `lt_strategy` (`True if is_float(pandera_dtype)
else None`) and avoids passing kwargs that hypothesis would reject for
non-float dtypes.

`compile_field_strategy` therefore contains roughly the same control
flow as `pandas_dtype_strategy` itself, but operates one level above:
it decides which kwargs to forward, and then calls
`pandas_dtype_strategy` exactly once.

### 9.2 String regex merging via lookahead-AND, with filter fallback

**Decision:** `_compile_string_strategy` produces a single
`st.from_regex(combined_pattern, fullmatch=True)` call whenever every
pattern in `regex_fullmatch + regex_search` compiles cleanly under
`re.compile(...)`. The combined pattern uses zero-width lookahead
conjunction:

```python
def _combine_patterns(
    fullmatch_patterns: tuple[str, ...],
    search_patterns: tuple[str, ...],
) -> tuple[str, bool] | None:
    """Return (combined_pattern, fullmatch) or None if not combinable."""
    parts: list[str] = []
    for p in fullmatch_patterns:
        try:
            re.compile(p)
        except re.error:
            return None
        parts.append(rf"(?=\A(?:{p})\Z)")
    for p in search_patterns:
        try:
            re.compile(p)
        except re.error:
            return None
        parts.append(rf"(?=.*(?:{p}))")
    if not parts:
        return None
    # Anchored placeholder so hypothesis still has a body to expand on.
    combined = "".join(parts) + r".*"
    return combined, True
```

`_compile_string_strategy` then:

1. Calls `_combine_patterns(...)`. If it returns a combined pattern,
   builds `st.from_regex(combined, fullmatch=True)` (this is the
   single-strategy fast path).
2. Falls back to `st.from_regex(first_pattern).filter(re.compile(rest).search)`
   for any pattern that did not compile (preserving today's behaviour
   bit-for-bit for those edge cases).
3. Applies `str_min_len` / `str_max_len` / `str_exact_len` by passing
   them as `min_size` / `max_size` to the underlying `st.text(...)` when
   no patterns are present, or by wrapping the regex strategy in a
   `.filter(lambda s: lo <= len(s) <= hi)` when both regex and length
   constraints are present (length filter is a single composed call,
   not chained).
4. Routes the result through the existing surrogate-stripping map
   (`_remove_surrogates` for pyarrow-backed string dtypes) — unchanged.

Rationale: lookahead-AND is strictly faster than chained `.filter`s
because hypothesis's regex generator can shrink and bias toward valid
strings; the filter fallback exists only for genuinely uncombinable
inputs (invalid regex, which today already raises in
`re.compile(pattern).fullmatch`).

A unit test in `tests/strategies/test_constraints.py` asserts that for
`Check.str_startswith("foo") & Check.str_endswith("bar") & Check.str_length(min_value=10)`
the resulting strategy graph contains zero `FilteredStrategy` nodes
for the regex portion (one length filter is acceptable).

### 9.3 Deprecation warning for legacy mixed strategy ordering

**Decision:** When a column has both built-in checks (which now go
through the constraint aggregator and produce the merged base strategy)
**and** at least one `Check(strategy=fn)` whose `fn` has a `strategy`
parameter with a `None` default (i.e. `fn` advertises itself as
"can be a base strategy"), `field_element_strategy` emits a
`DeprecationWarning` of the form:

```
The 'strategy' kwarg on Check(check_fn=..., strategy=<fn>) is being
invoked as a chained strategy because built-in checks are also present
on this column and now produce the merged base strategy in a single
hypothesis call. If <fn> relied on running as the base strategy in this
context, migrate it to a constraint adapter via
`pandera.strategies.pandas_strategies.register_check_constraint(...)`
or pass `constraint=...` to `register_check_method(...)`. This warning
will become an error in pandera 1.0.
```

Detection logic in `field_element_strategy`:

```python
import inspect

def _strategy_supports_base_mode(fn) -> bool:
    """True if fn has a 'strategy' parameter with a None default."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    param = sig.parameters.get("strategy")
    return param is not None and param.default is None
```

The warning is emitted at most once per `(check.name, fn id)` pair
per process (using a module-level `set` cache) to avoid noise in
hypothesis's many-draws loop. The warning is *not* emitted when:

- The column has no built-in checks (legacy ordering still works).
- The user-supplied strategy does not advertise base-mode support
  (i.e. `fn(pandera_dtype, strategy, **stats)` where `strategy` has no
  default — pure chained-mode strategy, unaffected).
- The user has migrated to `Check(constraint=...)` or
  `register_check_constraint(...)` (the constraint path is taken first
  and the legacy strategy path is skipped entirely).

Tests in `tests/strategies/test_strategies.py` (new) cover all three
non-warning cases plus the positive case.

---

## 10. Benchmarking plan

The performance argument for this refactor needs to be quantified, not
asserted. This section specifies the benchmarks that gate the work,
where they live, how they are run, and what numbers must move in which
direction for the change to be considered successful.

### 10.1 Infrastructure

Pandera already uses [Airspeed Velocity (`asv`)](https://asv.readthedocs.io)
for performance benchmarks, with results published at
[pandera-asv-logs](https://github.com/pandera-dev/pandera-asv-logs) and
benchmark sources in
[`asv_bench/benchmarks/`](../asv_bench/benchmarks/) (see
[`asv_bench/README.md`](../asv_bench/README.md) for the existing
workflow). We will extend this infrastructure rather than introduce a
new one.

A new file `asv_bench/benchmarks/strategies.py` is added containing the
benchmark classes specified in §10.3. The naming conventions (`time_*`,
`mem_*`, `peakmem_*`, `setup`, `setup_cache`) and import style follow
the existing
[`asv_bench/benchmarks/dataframe_schema.py`](../asv_bench/benchmarks/dataframe_schema.py)
and `series_schema.py`.

For *in-PR* comparison (since `asv` is designed for committed history,
not a single PR diff), we additionally add a `pytest-benchmark`-driven
smoke comparison that runs in CI on the strategies-refactor PR and
reports a summary table in the PR body. This is a one-off CI artifact,
not a permanent dashboard.

### 10.2 Methodology

#### 10.2.1 What we measure

Three orthogonal axes:

1. **Wall-clock generation time.** End-to-end time to produce a sample
   of size `N` from `Schema.example(size=N)`. Primary metric.
2. **Strategy graph shape.** Number of `FilteredStrategy` nodes in the
   compiled `SearchStrategy` tree returned by
   `Schema.strategy(size=N)`. A structural metric that does not depend
   on machine speed and gives a deterministic regression signal even
   when wall-clock is noisy. Implemented as a small recursive walker
   in `asv_bench/benchmarks/_strategy_introspection.py`.
3. **Hypothesis health-check trigger rate.** Count of
   `hypothesis.errors.FailedHealthCheck` and `Unsatisfiable` raises
   over a fixed budget of draws (e.g. 100 attempts at
   `Schema.example(size=100)`). A high-signal indicator of whether the
   refactor actually fixed the "Unsatisfiable" pain documented in §1.

We deliberately *do not* benchmark `Schema.validate(...)` here — the
refactor changes only the synthesis path and the existing
`dataframe_schema.py` benchmarks already cover validation.

#### 10.2.2 How we compare

For each benchmark scenario in §10.3:

- **Baseline:** `main` at the commit immediately before step 4 of the
  implementation plan (§7), i.e. the state where bucketing is in place
  but no built-in checks have been migrated yet — this is the
  "behaviour identical to today" checkpoint and lets us measure the
  pure constraint-aggregation speedup without conflating it with the
  bucketing refactor itself.
- **Treatment:** the tip of each migration step in §7 (numeric →
  membership → string → dataframe → xarray). Each step gets its own
  comparison row in the PR summary table.
- Report wall-clock as the median of 30 runs with `min_run_count=5`
  and `warmup_time=0.1` (the `asv` defaults) per scenario. Filter-node
  counts and health-check triggers are exact, not statistical.

#### 10.2.3 Reproducibility

- All benchmarks pin `hypothesis.seed(42)` and pass `max_examples=N`
  to `Schema.example`'s underlying `find` to remove RNG variance from
  the wall-clock measurement.
- `asv` configuration (Python version, pandas/numpy/hypothesis pins)
  matches the existing `pandera-asv-logs` matrix.
- Each benchmark `setup` rebuilds its schema from scratch so cached
  strategies don't leak between runs.

### 10.3 Benchmark scenarios

The scenarios are chosen to exercise the specific slow paths called
out in §1, plus the regression baselines that must *not* regress.

#### 10.3.1 Built-in single-check baseline (regression guard)

Goal: ensure the refactor does not regress simple cases where there
is nothing to aggregate.

```python
# asv_bench/benchmarks/strategies.py
class SingleCheckBaseline:
    params = ["gt", "isin", "str_matches", "in_range"]
    param_names = ["check_kind"]

    def setup(self, check_kind):
        self.schema = _schema_with_single_check(check_kind)

    def time_example_size_100(self, check_kind):
        self.schema.example(size=100)

    def time_strategy_construct(self, check_kind):
        self.schema.strategy(size=100)
```

**Success criterion:** wall-clock within ±10% of baseline. Strategy
graph: same number of filter nodes (allowed to *decrease*, not
increase).

#### 10.3.2 Aggregated numeric checks (primary win)

The case from
[`data_synthesis_strategies.md`](../docs/source/data_synthesis_strategies.md):

```python
class AggregatedNumericChecks:
    params = [1, 2, 4, 8]
    param_names = ["n_redundant_bounds"]

    def setup(self, n):
        bounds = [Check.gt(i) for i in range(n)] + [Check.lt(1e10)]
        self.schema = pa.DataFrameSchema({
            "col": pa.Column(float, checks=bounds + [
                Check.notin([-100, -10, 0]),
            ]),
        })

    def time_example_size_100(self, n):
        self.schema.example(size=100)

    def track_filter_node_count(self, n):
        return _count_filtered_strategies(self.schema.strategy(size=100))

    def track_unsatisfiable_rate(self, n):
        return _count_unsatisfiable_in_budget(self.schema, draws=100)
```

**Success criteria:**
- Wall-clock at `n=8`: ≥ **5×** faster than baseline.
- `track_filter_node_count` at `n=8`: ≤ **1** (allowed: the single
  trailing `notin` filter from `compile_field_strategy` step 5).
  Baseline today: ≥ `n + 2` filter nodes.
- `track_unsatisfiable_rate`: 0 in the new code; baseline may be > 0
  for `n ≥ 4`.

#### 10.3.3 Multi-check string columns (regex merging, §9.2)

```python
class StringRegexMerging:
    def setup_cache(self):
        return None

    def setup(self):
        self.schema = pa.DataFrameSchema({
            "col": pa.Column(str, checks=[
                Check.str_startswith("foo"),
                Check.str_endswith("bar"),
                Check.str_length(min_value=10, max_value=30),
            ]),
        })

    def time_example_size_100(self):
        self.schema.example(size=100)

    def track_regex_filter_node_count(self):
        return _count_filtered_strategies(
            self.schema.strategy(size=100), kind="regex",
        )
```

**Success criteria:**
- Wall-clock: ≥ **3×** faster than baseline (today this scenario
  frequently triggers `filter_too_much`).
- `track_regex_filter_node_count`: **0** (the merged
  `st.from_regex(...)` produced by `_compile_string_strategy` should
  have no regex-side filters; one length filter is acceptable and
  counted separately).

#### 10.3.4 Membership intersection (`isin` ∩ bounds ∩ `notin`)

```python
class MembershipIntersection:
    def setup(self):
        self.schema = pa.DataFrameSchema({
            "col": pa.Column(int, checks=[
                Check.isin(range(0, 1000)),
                Check.gt(100),
                Check.lt(900),
                Check.notin([200, 300, 400]),
            ]),
        })

    def time_example_size_100(self):
        self.schema.example(size=100)
```

**Success criterion:** wall-clock ≥ **10×** faster than baseline.
This is the case where today's `.filter`-chain on `st.sampled_from`
plus three more filters is pathological; the new path prunes the
allowed set up-front and emits a single `st.sampled_from(sorted(allowed))`.

#### 10.3.5 Wide DataFrame schema (per-column compounding)

```python
class WideSchema:
    params = [10, 50, 200]
    param_names = ["n_columns"]

    def setup(self, n_columns):
        cols = {
            f"col_{i}": pa.Column(float, checks=[
                Check.gt(0), Check.lt(100), Check.notin([50]),
            ])
            for i in range(n_columns)
        }
        self.schema = pa.DataFrameSchema(cols)

    def time_example_size_100(self, n_columns):
        self.schema.example(size=100)

    def peakmem_example_size_100(self, n_columns):
        self.schema.example(size=100)
```

**Success criteria:**
- Wall-clock at `n_columns=200`: ≥ **3×** faster than baseline.
- `peakmem_example_size_100`: within ±15% of baseline (the refactor
  shouldn't bloat memory; if anything, fewer composed strategy
  objects should reduce it).

#### 10.3.6 Joint-unsatisfiable construction time

Goal: verify the §6 promise that joint-unsatisfiable check stacks
fail *fast* at strategy-construction time instead of going through
hypothesis's exhaustion loop.

```python
class UnsatisfiableConstruction:
    def setup(self):
        self.schema = pa.DataFrameSchema({
            "col": pa.Column(int, checks=[
                Check.gt(100), Check.lt(50),  # empty intersection
            ]),
        })

    def time_construct_strategy(self):
        try:
            self.schema.strategy(size=100)
        except SchemaDefinitionError:
            pass

    def time_attempt_example(self):
        try:
            self.schema.example(size=100)
        except SchemaDefinitionError:
            pass
```

**Success criteria:**
- `time_construct_strategy`: ≤ **5 ms** (essentially constant; just
  the `merge` cost). Baseline: comparable (no work happens until
  draw).
- `time_attempt_example`: ≤ **20 ms** under the new code (raises
  `SchemaDefinitionError` immediately). Baseline: typically
  several seconds before `Unsatisfiable` is raised. **This is a
  qualitative win**, not a percentage; we report the actual wall-clock
  delta for both.

#### 10.3.7 Mixed legacy + built-in checks (§9.3 deprecation path)

```python
class MixedLegacyAndBuiltin:
    def setup(self):
        def custom_strategy(pandera_dtype, strategy=None, *, value):
            if strategy is None:
                return st.just(value)
            return strategy.filter(lambda x: x == value)

        self.schema = pa.DataFrameSchema({
            "col": pa.Column(int, checks=[
                Check.gt(0),                                  # built-in
                Check.lt(100),                                # built-in
                Check(lambda s: s == 42, strategy=custom_strategy),  # legacy
            ]),
        })

    def time_example_size_100(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.schema.example(size=100)
```

**Success criterion:** wall-clock ≥ **2×** faster than baseline.
Verifies that even when a legacy `Check(strategy=fn)` is present,
collapsing the built-in checks into a single base strategy still pays
off versus the per-check chain today. Also serves as a runtime
regression test for the deprecation warning path.

#### 10.3.8 xarray data array with checks (§5)

```python
class XarrayDataArrayChecks:
    def setup(self):
        import pandera.xarray as pax
        self.schema = pax.DataArraySchema(
            dtype="float64",
            dims=("time", "lat", "lon"),
            sizes={"time": 12, "lat": 30, "lon": 30},
            checks=[Check.gt(0), Check.lt(1e6), Check.notin([0])],
        )

    def time_example(self):
        self.schema.example()

    def track_filter_node_count(self):
        return _count_filtered_strategies(self.schema.strategy())
```

**Success criteria:**
- This scenario does not exist on `main` (xarray ignores `checks`),
  so we report only the absolute wall-clock and structural numbers
  for the new code, plus a manual check that the generated array
  satisfies the constraints (which is impossible on `main`).
- `track_filter_node_count`: ≤ 1.

### 10.4 Reporting

A new file `asv_bench/benchmarks/_summary.py` provides a CLI that
reads two `asv` result trees (baseline and treatment) and renders a
markdown table to stdout:

```
| Scenario | Metric | Baseline | New | Δ | Pass? |
|---|---|---|---|---|---|
| AggregatedNumericChecks(n=8) | time_example_size_100 | 4.20 s | 0.62 s | 6.8× | ✓ |
| AggregatedNumericChecks(n=8) | filter_node_count | 10 | 1 | -9 | ✓ |
| StringRegexMerging | time_example_size_100 | 1.10 s | 0.18 s | 6.1× | ✓ |
| StringRegexMerging | regex_filter_node_count | 3 | 0 | -3 | ✓ |
| ... | | | | | |
```

The CI job for the strategies-refactor PR posts this table as a
sticky comment. The thresholds in §10.3 are the `Pass?` column's
acceptance criteria; any row that fails fails the CI job.

### 10.5 Long-term tracking

After merge, the new `strategies.py` benchmarks become a permanent
part of the
[pandera-asv-logs dashboard](https://pandera-dev.github.io/pandera-asv-logs/),
following the existing flow documented in
[`asv_bench/README.md`](../asv_bench/README.md):

```bash
asv run ALL --config asv_bench/asv.conf.json
asv publish --config asv_bench/asv.conf.json
asv gh-pages --rewrite --config asv_bench/asv.conf.json
```

This guards against future regressions in the synthesis layer and
gives downstream users (especially in the docs case `Check.gt(0) &
Check.lt(1e10) & Check.notin([-100,-10,0])`) a public, citable
performance number.

### 10.6 Out of scope for the benchmark suite

- Benchmarking polars, pyspark, or ibis synthesis: those backends do
  not have strategies today and are out of scope for this refactor
  (§2.2).
- Benchmarking `Schema.validate(...)`: covered by the existing
  `dataframe_schema.py` and `series_schema.py` benchmarks; this
  refactor does not touch validation.
- Comparing against alternative property-based testing engines
  (`crosshair`, `pythonfuzz`, etc.): hypothesis remains the only
  supported engine.

---

## 11. Summary

This refactor introduces a single new value type (`FieldConstraints`),
a single new dispatcher (`CONSTRAINT_DISPATCHER`), a single new public
extension hook (`register_check_constraint` /
`register_check_method(constraint=...)`), and rewrites
`field_element_strategy` to aggregate constraints before constructing the
hypothesis strategy. All existing user code, custom strategies, and
extension entry points keep working unchanged. The wins are:

- One `npst.from_dtype` / `st.sampled_from` / `st.from_regex` call per
  field instead of N filters.
- Joint-unsatisfiability raised as a clean `SchemaDefinitionError` at
  strategy-construction time, not as `Unsatisfiable` after `hypothesis`
  exhausts itself.
- xarray actually honours `Check`s in synthesis.
- Check ordering stops being a performance concern for built-ins.
