---
file_format: mystnb
---

```{currentmodule} pandera
```

(scaling-pyspark)=

# Data Validation with Pyspark Pandas

*new in 0.10.0*

[Pyspark](https://spark.apache.org/docs/3.2.0/api/python/index.html) is a
distributed compute framework that offers a pandas drop-in replacement dataframe
implementation via the [pyspark.pandas API](https://spark.apache.org/docs/3.2.0/api/python/reference/pyspark.pandas/index.html) .
You can use pandera to validate {py:func}`~pyspark.pandas.DataFrame`
and {py:func}`~pyspark.pandas.Series` objects directly. First, install
`pandera` with the `pyspark` extra:

```bash
pip install 'pandera[pyspark]'
```

Then you can use pandera schemas to validate pyspark dataframes. In the example
below we'll use the {ref}`class-based API <dataframe-models>` to define a
{py:class}`~pandera.api.pandas.model.DataFrameModel` for validation.

```{code-cell} python
import pyspark.pandas as ps
import pandas as pd
import pandera.pandas as pa

from pandera.typing.pyspark import DataFrame, Series


class Schema(pa.DataFrameModel):
    state: Series[str]
    city: Series[str]
    price: Series[int] = pa.Field(in_range={"min_value": 5, "max_value": 20})


# create a pyspark.pandas dataframe that's validated on object initialization
df = DataFrame[Schema](
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
)
print(df)
```

You can also use the {py:func}`~pandera.check_types` decorator to validate
pyspark pandas dataframes at runtime:

```{code-cell} python
@pa.check_types
def function(df: DataFrame[Schema]) -> DataFrame[Schema]:
    return df[df["state"] == "CA"]

print(function(df))
```

And of course, you can use the object-based API to validate dask dataframes:

```{code-cell} python
schema = pa.DataFrameSchema({
    "state": pa.Column(str),
    "city": pa.Column(str),
    "price": pa.Column(int, pa.Check.in_range(min_value=5, max_value=20))
})
schema(df)
```
