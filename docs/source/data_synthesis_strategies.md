---
file_format: mystnb
---

% pandera documentation for synthesizing data

```{currentmodule} pandera
```

(data-synthesis-strategies)=

# Data Synthesis Strategies

*new in 0.6.0*

`pandera` provides a utility for generating synthetic data purely from
pandera schema or schema component objects. Under the hood, the schema metadata
is collected to create a data-generating strategy using
[hypothesis](https://hypothesis.readthedocs.io/en/latest/), which is a
property-based testing library.

:::{note}
The data synthesis feature requires a pandera installation with the
`strategies` dependency set:

`pip install 'pandera[strategies]'`

The `hypotheses` dependency set is for pandera hypothesis checks and does
not install the `hypothesis` library.
:::

## Basic Usage

Once you've defined a schema, it's easy to generate examples:

```{code-cell} python
import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {
        "column1": pa.Column(int, pa.Check.eq(10)),
        "column2": pa.Column(float, pa.Check.eq(0.25)),
        "column3": pa.Column(str, pa.Check.eq("foo")),
    }
)
schema.example(size=3)
```

Note that here we've constrained the specific values in each column using
{class}`~pandera.api.checks.Check` s  in order to make the data generation process
deterministic for documentation purposes.

## Usage in Unit Tests

The `example` method is available for all schemas and schema components, and
is primarily meant to be used interactively. It *could* be used in a script to
generate test cases, but `hypothesis` recommends against doing this and
instead using the `strategy` method to create a `hypothesis` strategy
that can be used in `pytest` unit tests.

```{code-cell} python
import hypothesis

def processing_fn(df):
    return df.assign(column4=df.column1 * df.column2)

@hypothesis.given(schema.strategy(size=5))
def test_processing_fn(dataframe):
    result = processing_fn(dataframe)
    assert "column4" in result
```

The above example is trivial, but you get the idea! Schema objects can create
a `strategy` that can then be collected by a [pytest](https://docs.pytest.org/en/latest/)
runner. We could also run the tests explicitly ourselves, or run it as a
`unittest.TestCase`. For more information on testing with hypothesis, see the
[hypothesis quick start guide](https://hypothesis.readthedocs.io/en/latest/quickstart.html#running-tests).

A more practical example involves using
{ref}`schema transformations<dataframe-schema-transformations>`. We can modify
the function above to make sure that `processing_fn` actually outputs the
correct result:

```{code-cell} python
out_schema = schema.add_columns({"column4": pa.Column(float)})

@pa.check_output(out_schema)
def processing_fn(df):
    return df.assign(column4=df.column1 * df.column2)

@hypothesis.given(schema.strategy(size=5))
def test_processing_fn(dataframe):
    processing_fn(dataframe)
```

Now the `test_processing_fn` simply becomes an execution test, raising a
{class}`~pandera.errors.SchemaError` if `processing_fn` doesn't add
`column4` to the dataframe.

## Strategies and Examples from DataFrame Models

You can also use the {ref}`class-based API<dataframe-models>` to generate examples.
Here's the equivalent dataframe model for the above examples:

```{code-cell} python
from pandera.typing import Series, DataFrame

class InSchema(pa.DataFrameModel):
    column1: Series[int] = pa.Field(eq=10)
    column2: Series[float] = pa.Field(eq=0.25)
    column3: Series[str] = pa.Field(eq="foo")

class OutSchema(InSchema):
    column4: Series[float]

@pa.check_types
def processing_fn(df: DataFrame[InSchema]) -> DataFrame[OutSchema]:
    return df.assign(column4=df.column1 * df.column2)

@hypothesis.given(InSchema.strategy(size=5))
def test_processing_fn(dataframe):
    processing_fn(dataframe)
```

## Checks as Constraints

As you may have noticed in the first example, {class}`~pandera.api.checks.Check` s
further constrain the data synthesized from a strategy. Without checks, the
`example` method would simply generate any value of the specified type. You
can specify multiple checks on a column and `pandera` should be able to
generate valid data under those constraints.

```{code-cell} python
schema_multiple_checks = pa.DataFrameSchema({
    "column1": pa.Column(
        float, checks=[
            pa.Check.gt(0),
            pa.Check.lt(1e10),
            pa.Check.notin([-100, -10, 0]),
        ]
    )
})

for _ in range(5):
    # generate 10 rows of the dataframe
    sample_data = schema_multiple_checks.example(size=3)

    # validate the sampled data
    schema_multiple_checks(sample_data)
```

One caveat here is that it's up to you to define a set of checks that are
jointly satisfiable. If not, an `Unsatisfiable` exception will be raised:

```{code-cell} python
:tags: [raises-exception]

import hypothesis

schema_multiple_checks = pa.DataFrameSchema({
    "column1": pa.Column(
        float, checks=[
            # nonsensical constraints
            pa.Check.gt(0),
            pa.Check.lt(-10),
        ]
    )
})

try:
    schema_multiple_checks.example(size=3)
except Exception as e:
    print(e)
```

(check-strategy-chaining)=

### Check Strategy Chaining

When all the checks on a column are *built-in* checks (`gt`, `lt`,
`isin`, `notin`, `eq`, `ne`, `in_range`, `str_*`, etc.), pandera now
aggregates them into a single set of constraints and asks `hypothesis`
to generate values that satisfy them in one shot. There is no
*base-strategy-vs-chained-strategy* distinction in this path, so:

- Ordering of built-in checks does not matter — `Check.gt(0) & Check.lt(100)`
  generates the same distribution as `Check.lt(100) & Check.gt(0)`.
- Bounds intersect (the tightest wins): `Check.gt(0) & Check.gt(5)` is
  equivalent to a single `min_value=5, exclude_min=True` lower bound.
- `isin` / `notin` / `eq` and the numeric bounds are intersected as
  sets, so an unsatisfiable combination raises a
  {class}`~pandera.errors.SchemaDefinitionError` at strategy-construction
  time rather than after `hypothesis` exhausts itself.

When you mix a custom `Check(strategy=...)` with built-ins, the custom
strategy still runs *after* the merged built-in base. If your custom
strategy advertises base-mode support (its `strategy` parameter has
``None`` as the default), pandera emits a `DeprecationWarning` pointing
you at the new {ref}`constraint adapter API<custom-constraints>` so you
can migrate it onto the same single-shot generation path.

### In-line Custom Checks

One of the strengths of `pandera` is its flexibility with regard to defining
custom checks on the fly:

```{code-cell} python
schema_inline_check = pa.DataFrameSchema({
    "col": pa.Column(str, pa.Check(lambda s: s.isin({"foo", "bar"})))
})
```

One of the disadvantages of this is that the fallback strategy is to simply
apply the check to the generated data, which can be highly inefficient. In this
case, `hypothesis` will generate strings and try to find examples of strings
that are in the set `{"foo", "bar"}`, which will be very slow and most likely
raise an `Unsatisfiable` exception. To get around this limitation, you can
register custom checks and define strategies that correspond to them.

(custom-strategies)=

## Defining Custom Strategies via the `strategy` kwarg

The {class}`~pandera.api.checks.Check` constructor exposes a `strategy`
keyword argument that allows you to define a data synthesis strategy that can
work as a *base strategy* or *chained strategy*. For example, suppose you define
a custom check that makes sure values in a column are in some specified range.

```{code-cell} python
check = pa.Check(lambda x: x.between(0, 100))
```

You can then define a strategy for this check with:

```{code-cell} python
import pandera.strategies.pandas_strategies as st

def in_range_strategy(pandera_dtype, strategy=None):
    if strategy is None:
        # handle base strategy case
        return st.floats(min_value=min_val, max_value=max_val).map(
            # the map isn't strictly necessary, but shows an example of
            # using the pandera_dtype argument
            strategies.to_numpy_dtype(pandera_dtype).type
        )

    # handle chained strategy case
    return strategy.filter(lambda val: 0 <= val <= 10)

check = pa.Check(lambda x: x.between(0, 100), strategy=in_range_strategy)
```

Notice that the `in_range_strategy` function takes two arguments: `pandera_dtype`,
and `strategy`. `pandera_dtype` is required, since this is almost always
required information when generating data. The `strategy` argument is optional,
where the default case assumes a *base strategy*, where the check is specified
as the first one in the list of checks specified at the column- or dataframe- level.

## Defining Custom Strategies via Check Registration

All built-in {class}`~pandera.api.checks.Check` s are associated with a data
synthesis strategy. You can define your own data synthesis strategies by using
the {ref}`extensions API<extensions>` to register a custom check function with
a corresponding strategy.

(custom-constraints)=

## Defining Custom *Constraint Adapters*

The recommended path for any new check is to write a **constraint
adapter** instead of (or in addition to) a `strategy=` callable. A
constraint adapter takes the check's statistics as keyword arguments
and returns a {class}`~pandera.strategies.constraints.FieldConstraints`
value describing the *intersection of valid values*. pandera's
synthesis layer then merges the adapter's output with sibling
constraints from other checks and lowers the merged result to a single
`hypothesis` strategy — no `.filter` chaining, no rejection sampling.

The `FieldConstraints` value type lets adapters compose:

- Numeric bounds (`min_value`, `max_value`, `exclude_min`,
  `exclude_max`) intersect tightest-wins.
- Membership (`isin`, `notin`, `eq`) is set-intersected.
- String regex patterns (`regex_fullmatch`, `regex_search`) and
  length bounds (`str_min_len`, `str_max_len`, `str_exact_len`)
  combine into a single `from_regex(...)` call where possible.

### Attaching an adapter to a single `Check` instance

For one-off checks the easiest path is the new `constraint=` kwarg on
`Check`:

```python
import pandera.pandas as pa
from pandera.strategies.constraints import FieldConstraints

def positive_lt_constraint(*, value):
    return FieldConstraints(
        min_value=0, max_value=value,
        exclude_min=True, exclude_max=True,
    )

check = pa.Check(
    lambda s: (s > 0) & (s < 5),
    constraint=positive_lt_constraint,
    statistics={"value": 5},
)
```

### Registering an adapter for a custom check

For checks registered with `register_check_method`, pass `constraint=`
alongside the existing `strategy=` arg (you can pass either or both):

```python
from pandera.api.extensions import register_check_method
from pandera.strategies.constraints import FieldConstraints

def my_check_constraint(*, value):
    return FieldConstraints(notin=frozenset({value}))

@register_check_method(constraint=my_check_constraint, statistics=["value"])
def not_eq(pandas_obj, *, value):
    return pandas_obj != value
```

### Migration: from `strategy=` to `constraint=`

When a `Check(strategy=fn)` is mixed with built-in checks, pandera now
issues a `DeprecationWarning` of the form:

```
The 'strategy' kwarg on Check(check_fn=..., strategy=<fn>) is being
invoked as a chained strategy because built-in checks are also present
on this column and now produce the merged base strategy in a single
hypothesis call. ...
```

The fix is to write a `FieldConstraints`-returning adapter for the
same check and pass it as `constraint=fn`. The legacy `strategy=`
path keeps working until pandera 1.0; the warning fires at most once
per `(check.name, fn id)` pair.

## Performance Notes

The Stage 5 / Stage 8 refactor makes data generation considerably
faster for schemas that mix multiple built-in checks per column:

- A column with `Check.gt(0) & Check.lt(1_000) & Check.notin([100, 500])`
  used to compose four `.filter(...)` nodes on top of an unbounded
  numeric strategy; it now lowers to a single `from_dtype(min_value=0,
  max_value=1000)` call plus one trailing `notin` filter.
- `Check.str_startswith("foo") & Check.str_endswith("bar")` no longer
  generates random strings and rejects most of them — the patterns are
  structurally merged into a single anchored regex
  (`\A(?:foo).*(?:bar)\Z`) that `hypothesis` drives directly.
- xarray strategies now respect `checks` arguments (they used to be
  silently ignored); the same constraint aggregation machinery powers
  `data_array_strategy` and `dataset_strategy`.
