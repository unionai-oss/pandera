# Extensions

*new in 0.6.0*

## Registering Custom Check Methods

One of the strengths of `pandera` is its flexibility in enabling you to
defining in-line custom checks on the fly:

```python
import pandera.pandas as pa

# checks elements in a column/dataframe
element_wise_check = pa.Check(lambda x: x < 0, element_wise=True)

# applies the check function to a dataframe/series
vectorized_check = pa.Check(lambda series_or_df: series_or_df < 0)
```

However, there are two main disadvantages of schemas with inline custom checks:

1. they are not serializable with the [IO interface](./schema_inference.html#persisting-a-schema).
2. you can't use them to [synthesize data](./data_synthesis_strategies.html)
   because the checks are not associated with a `hypothesis` strategy.

`pandera` now offers a way to register custom checks so that they're
available in the [`Check`](./reference/generated/pandera.api.checks.Check.html) class as a check method. Here
let's define a custom method that checks whether a pandas object contains
elements that lie within two values.

```python
import pandera.pandas as pa
import pandera.extensions as extensions
import pandas as pd

@extensions.register_check_method(statistics=["min_value", "max_value"])
def is_between(pandas_obj, *, min_value, max_value):
    return (min_value <= pandas_obj) & (pandas_obj <= max_value)

schema = pa.DataFrameSchema({
    "col": pa.Column(int, pa.Check.is_between(min_value=1, max_value=10))
})

data = pd.DataFrame({"col": [1, 5, 10]})
schema.validate(data)
```

As you can see, a custom check's first argument is a pandas series or dataframe
by default (more on that later), followed by keyword-only arguments, specified
with the `*` syntax.

The [`register_check_method`](./reference/generated/pandera.extensions.html) requires you to
explicitly name the check `statistics` via the keyword argument, which are
essentially the constraints placed by the check on the pandas data structure.

## Specifying a Check Strategy { #extension-check-strategy }

To specify a check strategy with your custom check, you'll need to install the
[strategies extension](./index.html#install). First let's look at a trivially simple
example, where the check verifies whether a column is equal to a certain value:

```python
def custom_equals(pandas_obj, *, value):
    return pandas_obj == value
```

The corresponding strategy for this check would be:

```python
from typing import Optional
import hypothesis
import pandera.strategies.pandas_strategies as st

def equals_strategy(
    pandera_dtype: pa.DataType,
    strategy: Optional[st.SearchStrategy] = None,
    *,
    value,
):
    if strategy is None:
        return st.pandas_dtype_strategy(
            pandera_dtype, strategy=hypothesis.strategies.just(value),
        )
    return strategy.filter(lambda x: x == value)
```

As you may notice, the `pandera` strategy interface has two positional arguments
followed by keyword-only arguments that match the check function keyword-only
check statistics. The `pandera_dtype` positional argument is useful for
ensuring the correct data type. In the above example, we're using the
[`pandas_dtype_strategy`](./reference/generated/pandera.strategies.pandas_strategies.html) strategy to
make sure the generated `value` is of the correct data type.

The optional `strategy` argument allows us to use the check strategy as a
*base strategy* or a *chained strategy*. There's a detail that we're
responsible for implementing in the strategy function body: we need to handle
two cases to account for [strategy chaining](./data_synthesis_strategies.html#check-strategy-chaining):

1. when the strategy function is being used as a *base strategy*, i.e. when
   `strategy` is `None`
2. when the strategy function is being chained from a previously-defined
   strategy, i.e. when `strategy` is not `None`.

!!! note

    The `strategy=` parameter is preserved for backward compatibility. New
    code should prefer the `constraint=` parameter (described in
    [Defining Custom *Constraint Adapters*](./data_synthesis_strategies.html#custom-constraints)), which lets your check participate in
    pandera's single-shot constraint aggregator. When a `Check(strategy=fn)`
    is mixed with built-in checks, pandera emits a `DeprecationWarning`
    pointing to the migration path.

Finally, to register the custom check with the strategy, use the
[`register_check_method`](./reference/generated/pandera.extensions.html) decorator:

```python
@extensions.register_check_method(
    statistics=["value"], strategy=equals_strategy
)
def custom_equals(pandas_obj, *, value):
    return pandas_obj == value
```

Let's unpack what's going in here. The `custom_equals` function only has
a single statistic, which is the `value` argument, which we've also specified
in [`register_check_method`](./reference/generated/pandera.extensions.html). This means that the
associated check strategy must match its keyword-only arguments.

Going back to our `is_between` function example, here's what the strategy
would look like:

```python
def in_between_strategy(
    pandera_dtype: pa.DataType,
    strategy: Optional[st.SearchStrategy] = None,
    *,
    min_value,
    max_value
):
    if strategy is None:
        return st.pandas_dtype_strategy(
            pandera_dtype,
            min_value=min_value,
            max_value=max_value,
            exclude_min=False,
            exclude_max=False,
        )
    return strategy.filter(lambda x: min_value <= x <= max_value)

@extensions.register_check_method(
    statistics=["min_value", "max_value"],
    strategy=in_between_strategy,
)
def is_between_with_strat(pandas_obj, *, min_value, max_value):
    return (min_value <= pandas_obj) & (pandas_obj <= max_value)
```

## Check Types

The extensions module also supports registering
[element-wise](./checks.html#vectorized-vs-element-wise-checks) and [groupby](./checks.html#column-check-groups)
checks.

### Element-wise Checks

```python
@extensions.register_check_method(
    statistics=["val"],
    check_type="element_wise",
)
def element_wise_equal_check(element, *, val):
    return element == val
```

Note that the first argument of `element_wise_equal_check` is a single
element in the column or dataframe.

### Groupby Checks

In this groupby check, we're verifying that the values of one column for
`group_a` are, on average, greater than those of `group_b`:

```python
from typing import Dict

@extensions.register_check_method(
    statistics=["group_a", "group_b"],
    check_type="groupby",
)
def groupby_check(dict_groups: Dict[str, pd.Series], *, group_a, group_b):
    return dict_groups[group_a].mean() > dict_groups[group_b].mean()

data = pd.DataFrame({
    "values": [20, 10, 1, 15],
    "groups": list("xxyy"),
})

schema = pa.DataFrameSchema({
    "values": pa.Column(
        int,
        pa.Check.groupby_check(group_a="x", group_b="y", groupby="groups"),
    ),
    "groups": pa.Column(str),
})

schema.validate(data)
```

## Registered Custom Checks with the Class-based API { #class-based-api-dataframe-checks }

Since registered checks are part of the [`Check`](./reference/generated/pandera.api.checks.Check.html) namespace,
you can also use custom checks with the [class-based API](./dataframe_models.html):

```python
from pandera.typing import Series

class Schema(pa.DataFrameModel):
    col1: Series[str] = pa.Field(custom_equals="value")
    col2: Series[int] = pa.Field(is_between={"min_value": 0, "max_value": 10})

data = pd.DataFrame({
    "col1": ["value"] * 5,
    "col2": range(5)
})

Schema.validate(data)
```

DataFrame checks can be attached by using the [Config](./dataframe_models.html#config) class. Any field names that
do not conflict with existing fields of [`BaseConfig`](./reference/generated/pandera.api.pandas.model_config.BaseConfig.html) and do not start
with an underscore (`_`) are interpreted as the name of registered checks. If the value
is a tuple or dict, it is interpreted as the positional or keyword arguments of the check, and
as the first argument otherwise.

For example, to register zero, one, and two statistic dataframe checks one could do the following:

```python
import pandera.pandas as pa
import pandera.extensions as extensions
import numpy as np
import pandas as pd


@extensions.register_check_method()
def is_small(df):
    return sum(df.shape) < 1000


@extensions.register_check_method(statistics=["fraction"])
def total_missing_fraction_less_than(df, *, fraction: float):
    return (1 - df.count().sum().item() / df.apply(len).sum().item()) < fraction


@extensions.register_check_method(statistics=["col_a", "col_b"])
def col_mean_a_greater_than_b(df, *, col_a: str, col_b: str):
    return df[col_a].mean() > df[col_b].mean()


from pandera.typing import Series


class Schema(pa.DataFrameModel):
    col1: Series[float] = pa.Field(nullable=True, ignore_na=False)
    col2: Series[float] = pa.Field(nullable=True, ignore_na=False)

    class Config:
        is_small = ()
        total_missing_fraction_less_than = 0.6
        col_mean_a_greater_than_b = {"col_a": "col2", "col_b": "col1"}


data = pd.DataFrame({
    "col1": [float('nan')] * 3 + [0.5, 0.3, 0.1],
    "col2": np.arange(6.),
})

Schema.validate(data)
```
