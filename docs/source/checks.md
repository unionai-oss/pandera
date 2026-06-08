---
file_format: mystnb
---

% pandera documentation for Checks

```{currentmodule} pandera
```

(checks)=

# Validating with Checks

Checks are one of the fundamental constructs of pandera. They allow you to
specify properties about dataframes, columns, indexes, and series objects, which
are applied after data type validation/coercion and the core pandera checks
are applied to the data to be validated.

```{important}
You can learn more about how data type validation works
{ref}`dtype-validation`.
```

## Checking column properties

{class}`~pandera.api.checks.Check` objects accept a function as a required argument, which is
expected to take a `pa.Series` input and output a `boolean` or a `Series`
of boolean values. For the check to pass, all of the elements in the boolean
series must evaluate to `True`, for example:

```{code-cell} python
import pandera.pandas as pa
import pandas as pd

check_lt_10 = pa.Check(lambda s: s <= 10)

schema = pa.DataFrameSchema({"column1": pa.Column(int, check_lt_10)})
schema.validate(pd.DataFrame({"column1": range(10)}))
```

Multiple checks can be applied to a column:

```{code-cell} python
schema = pa.DataFrameSchema({
    "column2": pa.Column(str, [
        pa.Check(lambda s: s.str.startswith("value")),
        pa.Check(lambda s: s.str.split("_", expand=True).shape[1] == 2)
    ]),
})
```

## Built-in Checks

For common validation tasks, built-in checks are available in `pandera`.

```{code-cell} python
import pandera.pandas as pa

schema = pa.DataFrameSchema({
    "small_values": pa.Column(float, pa.Check.less_than(100)),
    "one_to_three": pa.Column(int, pa.Check.isin([1, 2, 3])),
    "phone_number": pa.Column(str, pa.Check.str_matches(r'^[a-z0-9-]+$')),
})
```

See the {class}`~pandera.api.checks.Check` API reference for a complete list of built-in checks.

(elementwise-checks)=

## Vectorized vs. Element-wise Checks

By default, {class}`~pandera.api.checks.Check` objects operate on `pd.Series`
objects. If you want to make atomic checks for each element in the Column, then
you can provide the `element_wise=True` keyword argument:

```{code-cell} python
import pandas as pd
import pandera.pandas as pa

schema = pa.DataFrameSchema({
    "a": pa.Column(
        int,
        checks=[
            # a vectorized check that returns a bool
            pa.Check(lambda s: s.mean() > 5, element_wise=False),

            # a vectorized check that returns a boolean series
            pa.Check(lambda s: s > 0, element_wise=False),

            # an element-wise check that returns a bool
            pa.Check(lambda x: x > 0, element_wise=True),
        ]
    ),
})
df = pd.DataFrame({"a": [4, 4, 5, 6, 6, 7, 8, 9]})
schema.validate(df)
```

`element_wise == False` by default so that you can take advantage of the
speed gains provided by the `pd.Series` API by writing vectorized
checks.

(grouping)=

## Handling Null Values

By default, `pandera` drops null values before passing the objects to
validate into the check function. For `Series` objects null elements are
dropped (this also applies to columns), and for `DataFrame` objects, rows
with any null value are dropped.

If you want to check the properties of a pandas data structure while preserving
null values, specify `Check(..., ignore_na=False)` when defining a check.

Note that this is different from the `nullable` argument in {class}`~pandera.api.pandas.components.Column`
objects, which simply checks for null values in a column.

(column-check-groups)=

## Column Check Groups

{class}`~pandera.api.pandas.components.Column` checks support grouping by a different column so that you
can make assertions about subsets of the column of interest. This
changes the function signature of the {class}`~pandera.api.checks.Check` function so that its
input is a dict where keys are the group names and values are subsets of the
series being validated.

Specifying `groupby` as a column name, list of column names, or
callable changes the expected signature of the {class}`~pandera.api.checks.Check`
function argument to:

`Callable[Dict[Any, pd.Series] -> Union[bool, pd.Series]`

where the dict keys are the discrete keys in the `groupby` columns.

In the example below we define a {class}`~pandera.api.pandas.container.DataFrameSchema` with column checks
for `height_in_feet` using a single column, multiple columns, and a more
complex groupby function that creates a new column `age_less_than_15` on the
fly.

```{code-cell} python
import pandas as pd
import pandera.pandas as pa

schema = pa.DataFrameSchema({
    "height_in_feet": pa.Column(
        float, [
            # groupby as a single column
            pa.Check(
                lambda g: g[False].mean() > 6,
                groupby="age_less_than_20"),

            # define multiple groupby columns
            pa.Check(
                lambda g: g[(True, "F")].sum() == 9.1,
                groupby=["age_less_than_20", "sex"]),

            # groupby as a callable with signature:
            # (DataFrame) -> DataFrameGroupBy
            pa.Check(
                lambda g: g[(False, "M")].median() == 6.75,
                groupby=lambda df: (
                    df.assign(age_less_than_15=lambda d: d["age"] < 15)
                    .groupby(["age_less_than_15", "sex"]))),
        ]),
    "age": pa.Column(int, pa.Check(lambda s: s > 0)),
    "age_less_than_20": pa.Column(bool),
    "sex": pa.Column(str, pa.Check(lambda s: s.isin(["M", "F"])))
})

df = (
    pd.DataFrame({
        "height_in_feet": [6.5, 7, 6.1, 5.1, 4],
        "age": [25, 30, 21, 18, 13],
        "sex": ["M", "M", "F", "F", "F"]
    })
    .assign(age_less_than_20=lambda x: x["age"] < 20)
)

schema.validate(df)
```

(wide-checks)=

## Wide Checks

`pandera` is primarily designed to operate on long-form data (commonly known
as [tidy data](https://vita.had.co.nz/papers/tidy-data.pdf)), where each row
is an observation and each column is an attribute associated with an
observation.

However, `pandera` also supports checks on wide-form data to operate across
columns in a `DataFrame`. For example, if you want to make assertions about
`height` across two groups, the tidy dataset and schema might look like this:

```{code-cell} python
import pandas as pd
import pandera.pandas as pa


df = pd.DataFrame({
    "height": [5.6, 6.4, 4.0, 7.1],
    "group": ["A", "B", "A", "B"],
})

schema = pa.DataFrameSchema({
    "height": pa.Column(
        float,
        pa.Check(lambda g: g["A"].mean() < g["B"].mean(), groupby="group")
    ),
    "group": pa.Column(str)
})

schema.validate(df)
```

Whereas the equivalent wide-form schema would look like this:

```{code-cell} python
df = pd.DataFrame({
    "height_A": [5.6, 4.0],
    "height_B": [6.4, 7.1],
})

schema = pa.DataFrameSchema(
    columns={
        "height_A": pa.Column(float),
        "height_B": pa.Column(float),
    },
    # define checks at the DataFrameSchema-level
    checks=pa.Check(
        lambda df: df["height_A"].mean() < df["height_B"].mean()
    )
)

schema.validate(df)
```

You can see that when checks are supplied to the `DataFrameSchema` `checks`
key-word argument, the check function should expect a pandas `DataFrame` and
should return a `bool`, a `Series` of booleans, or a `DataFrame` of
boolean values.

## Raise Warning Instead of Error on Check Failure

In some cases, you might want to raise a warning and continue execution
of your program. The `Check` and `Hypothesis` classes and their built-in
methods support the keyword argument `raise_warning`, which is `False`
by default. If set to `True`, the check will warn with a `SchemaWarning` instead
of raising a `SchemaError` exception.

:::{note}
Use this feature carefully! If the check is for informational purposes and
not critical for data integrity then use `raise_warning=True`. However,
if the assumptions expressed in a `Check` are necessary conditions to
considering your data valid, do not set this option to true.
:::

One scenario where you'd want to do this would be in a data pipeline that
does some preprocessing, checks for normality in certain columns, and writes
the resulting dataset to a table. In this case, you want to see if your
normality assumptions are not fulfilled by certain columns, but you still
want the resulting table for further analysis.

```{code-cell} python
import warnings

import numpy as np
import pandas as pd
import pandera.pandas as pa

from scipy.stats import normaltest


np.random.seed(1000)

df = pd.DataFrame({
    "var1": np.random.normal(loc=0, scale=1, size=1000),
    "var2": np.random.uniform(low=0, high=10, size=1000),
})

normal_check = pa.Hypothesis(
    test=normaltest,
    samples="normal_variable",
    # null hypotheses: sample comes from a normal distribution. The
    # relationship function checks if we cannot reject the null hypothesis,
    # i.e. the p-value is greater or equal to alpha.
    relationship=lambda stat, pvalue, alpha=0.05: pvalue >= alpha,
    error="normality test",
    raise_warning=True,
)

schema = pa.DataFrameSchema(
    columns={
        "var1": pa.Column(checks=normal_check),
        "var2": pa.Column(checks=normal_check),
    }
)

# catch and print warnings
with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.simplefilter("always")
    validated_df = schema(df)
    for warning in caught_warnings:
        print(warning.message)
```

## Registering Custom Checks

`pandera` now offers an interface to register custom checks functions so
that they're available in the {class}`~pandera.api.checks.Check` namespace. See
{ref}`the extensions<extensions>` document for more information.
