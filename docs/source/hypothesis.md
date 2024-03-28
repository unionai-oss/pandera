---
file_format: mystnb
---

% pandera documentation for Hypothesis Testing

```{currentmodule} pandera
```

(hypothesis)=

# Hypothesis Testing

`pandera` enables you to perform statistical hypothesis tests on your data.

:::{note}
The hypothesis feature requires a pandera installation with `hypotheses`
dependency set. See the {ref}`installation<installation>` instructions for
more details.
:::

## Overview

The {class}`~pandera.api.hypotheses.Hypothesis` class defines built in methods,
which can be called as in this example of a two-sample t-test:

```{code-cell} python
import pandas as pd
import pandera as pa

from pandera import Column, DataFrameSchema, Check, Hypothesis

from scipy import stats

df = (
    pd.DataFrame({
        "height_in_feet": [6.5, 7, 6.1, 5.1, 4],
        "sex": ["M", "M", "F", "F", "F"]
    })
)

schema = DataFrameSchema({
    "height_in_feet": Column(
        float, [
            Hypothesis.two_sample_ttest(
                sample1="M",
                sample2="F",
                groupby="sex",
                relationship="greater_than",
                alpha=0.05,
                equal_var=True),
    ]),
    "sex": Column(str)
})

try:
    schema.validate(df)
except pa.errors.SchemaError as exc:
    print(exc)
```

You can also define custom hypotheses by passing in functions to the
`test` and `relationship` arguments.

The `test` function takes as input one or multiple array-like objects
and should return a `stat`, which is the test statistic, and `pvalue` for
assessing statistical significance. It also takes key-word arguments supplied
by the `test_kwargs` dict when initializing a `Hypothesis` object.

The `relationship` function should take all of the outputs of `test` as
positional arguments, in addition to key-word arguments supplied by the
`relationship_kwargs` dict.

Here's an implementation of the two-sample t-test that uses the
[scipy implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html):

```{code-cell} python
def two_sample_ttest(array1, array2):
    # the "height_in_feet" series is first grouped by "sex" and then
    # passed into the custom `test` function as two separate arrays in the
    # order specified in the `samples` argument.
    return stats.ttest_ind(array1, array2)


def null_relationship(stat, pvalue, alpha=0.01):
    return pvalue / 2 >= alpha


schema = DataFrameSchema({
    "height_in_feet": Column(
        float, [
            Hypothesis(
                test=two_sample_ttest,
                samples=["M", "F"],
                groupby="sex",
                relationship=null_relationship,
                relationship_kwargs={"alpha": 0.05}
            )
    ]),
    "sex": Column(str, checks=Check.isin(["M", "F"]))
})

schema.validate(df)
```

## Wide Hypotheses

`pandera` is primarily designed to operate on long-form data (commonly known
as [tidy data](https://vita.had.co.nz/papers/tidy-data.pdf)), where each row
is an observation and columns are attributes associated with the observation.

However, `pandera` also supports hypothesis testing on wide-form data to
operate across columns in a `DataFrame`.

For example, if you want to make assertions about `height` across two groups,
the tidy dataset and schema might look like this:

```{code-cell} python
import pandas as pd
import pandera as pa

from pandera import Check, DataFrameSchema, Column, Hypothesis

df = pd.DataFrame({
    "height": [5.6, 7.5, 4.0, 7.9],
    "group": ["A", "B", "A", "B"],
})

schema = DataFrameSchema({
    "height": Column(
        float, Hypothesis.two_sample_ttest(
            "A", "B",
            groupby="group",
            relationship="less_than",
            alpha=0.05
        )
    ),
    "group": Column(str, Check(lambda s: s.isin(["A", "B"])))
})

schema.validate(df)
```

The equivalent wide-form schema would look like this:

```{code-cell} python
import pandas as pd
import pandera as pa

from pandera import DataFrameSchema, Column, Hypothesis

df = pd.DataFrame({
    "height_A": [5.6, 4.0],
    "height_B": [7.5, 7.9],
})

schema = DataFrameSchema(
    columns={
        "height_A": Column(float),
        "height_B": Column(float),
    },
    # define checks at the DataFrameSchema-level
    checks=Hypothesis.two_sample_ttest(
        "height_A", "height_B",
        relationship="less_than",
        alpha=0.05
    )
)

schema.validate(df)
```
