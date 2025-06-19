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
import pandera.pandas as pa


from scipy import stats

df = (
    pd.DataFrame({
        "height_in_feet": [6.5, 7, 6.1, 5.1, 4],
        "sex": ["M", "M", "F", "F", "F"]
    })
)

schema = pa.DataFrameSchema({
    "height_in_feet": pa.Column(
        float, [
            pa.Hypothesis.two_sample_ttest(
                sample1="M",
                sample2="F",
                groupby="sex",
                relationship="greater_than",
                alpha=0.05,
                equal_var=True),
    ]),
    "sex": pa.Column(str)
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


schema = pa.DataFrameSchema({
    "height_in_feet": pa.Column(
        float, [
            pa.Hypothesis(
                test=two_sample_ttest,
                samples=["M", "F"],
                groupby="sex",
                relationship=null_relationship,
                relationship_kwargs={"alpha": 0.05}
            )
    ]),
    "sex": pa.Column(str, checks=pa.Check.isin(["M", "F"]))
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
import pandera.pandas as pa


df = pd.DataFrame({
    "height": [5.6, 7.5, 4.0, 7.9],
    "group": ["A", "B", "A", "B"],
})

schema = pa.DataFrameSchema({
    "height": pa.Column(
        float, pa.Hypothesis.two_sample_ttest(
            "A", "B",
            groupby="group",
            relationship="less_than",
            alpha=0.05
        )
    ),
    "group": pa.Column(str, pa.Check(lambda s: s.isin(["A", "B"])))
})

schema.validate(df)
```

The equivalent wide-form schema would look like this:

```{code-cell} python
import pandas as pd
import pandera.pandas as pa


df = pd.DataFrame({
    "height_A": [5.6, 4.0],
    "height_B": [7.5, 7.9],
})

schema = pa.DataFrameSchema(
    columns={
        "height_A": pa.Column(float),
        "height_B": pa.Column(float),
    },
    # define checks at the DataFrameSchema-level
    checks=pa.Hypothesis.two_sample_ttest(
        "height_A", "height_B",
        relationship="less_than",
        alpha=0.05
    )
)

schema.validate(df)
```
