# Data Validation with Dask

*new in 0.8.0*

[Dask](https://docs.dask.org/en/latest/dataframe.html) is a distributed
compute framework that offers a pandas-like dataframe API.
You can use pandera to validate `dask.dataframe.DataFrame`
and `dask.dataframe.Series` objects directly. First, install
`pandera` with the `dask` extra:

```bash
pip install 'pandera[dask]'
```

Then you can use pandera schemas to validate dask dataframes. In the example
below we'll use the [class-based API](./dataframe_models.html) to define a
[`DataFrameModel`](./reference/generated/pandera.api.pandas.model.DataFrameModel.html) for validation.

```python
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

```python
pandera_ddf.compute()
```

You can also use the [`check_types`](./reference/generated/pandera.decorators.check_types.html) decorator to validate
dask dataframes at runtime:

```python
@pa.check_types
def function(ddf: DataFrame[Schema]) -> DataFrame[Schema]:
    return ddf[ddf["state"] == "CA"]

function(ddf).compute()
```

And of course, you can use the object-based API to validate dask dataframes:

```python
schema = pa.DataFrameSchema({
    "state": pa.Column(str),
    "city": pa.Column(str),
    "price": pa.Column(int, pa.Check.in_range(min_value=5, max_value=20))
})
schema(ddf).compute()
```
