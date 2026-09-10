---
file_format: mystnb
---

```{currentmodule} pandera
```

(scaling-dask)=

# Data Validation with Dask

*new in 0.8.0*

[Dask](https://docs.dask.org/en/latest/dataframe.html) is a distributed
compute framework that offers a pandas-like dataframe API.
You can use pandera to validate {py:func}`~dask.dataframe.DataFrame`
and {py:func}`~dask.dataframe.Series` objects directly. First, install
`pandera` with the `dask` extra:

```bash
pip install 'pandera[dask]'
```

Then you can use pandera schemas to validate dask dataframes. In the example
below we'll use the {ref}`class-based API <dataframe-models>` to define a
{py:class}`~pandera.api.pandas.model.DataFrameModel` for validation.

```{code-cell} python
import dask.dataframe as dd
import pandas as pd
import pandera.pandas as pa

from pandera.typing.dask import DataFrame, Series


class Schema(pa.DataFrameModel):
    state: Series[str]
    city: Series[str]
    price: Series[int] = pa.Field(in_range={"min_value": 5, "max_value": 20})


ddf = dd.from_pandas(
    pd.DataFrame(
        {
            'state': ['FL','FL','FL','CA','CA','CA'],
            'city': [
                'Orlando',
                'Miami',
                'Tampa',
                'San Francisco',
                'Los Angeles',
                'San Diego',
            ],
            'price': [8, 12, 10, 16, 20, 18],
        }
    ),
    npartitions=2
)
pandera_ddf = Schema(ddf)
pandera_ddf
```

As you can see, passing the dask dataframe into `Schema` will produce
another dask dataframe which hasn't been evaluated yet. What this means is
that pandera will only validate when the dask graph is evaluated.

```{code-cell} python
pandera_ddf.compute()
```

You can also use the {py:func}`~pandera.check_types` decorator to validate
dask dataframes at runtime:

```{code-cell} python
@pa.check_types
def function(ddf: DataFrame[Schema]) -> DataFrame[Schema]:
    return ddf[ddf["state"] == "CA"]

function(ddf).compute()
```

And of course, you can use the object-based API to validate dask dataframes:

```{code-cell} python
schema = pa.DataFrameSchema({
    "state": pa.Column(str),
    "city": pa.Column(str),
    "price": pa.Column(int, pa.Check.in_range(min_value=5, max_value=20))
})
schema(ddf).compute()
```
