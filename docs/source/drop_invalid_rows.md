---
file_format: mystnb
---

```{currentmodule} pandera
```

(drop-invalid-rows)=

# Dropping Invalid Rows

*New in version 0.16.0*

If you wish to use the validation step to remove invalid data, you can pass the
`drop_invalid_rows=True` argument to the `schema` object on creation. On `schema.validate()`,
if a data-level check fails, then that row which caused the failure will be removed from the dataframe
when it is returned.

`drop_invalid_rows` will prevent data-level schema errors being raised and will instead
remove the rows which causes the failure.

This functionality is available on `DataFrameSchema`, `SeriesSchema`, `Column`,
as well as `DataFrameModel` schemas.

**Note** that this functionality works by identifying the index or multi-index of the failing rows.
If the index is not unique on the dataframe, this could result in incorrect rows being dropped.

Dropping invalid rows with {class}`~pandera.api.pandas.container.DataFrameSchema`:

```{code-cell} python
import pandas as pd
import pandera.pandas as pa


df = pd.DataFrame({"counter": [1, 2, 3]})
schema = pa.DataFrameSchema(
    {"counter": pa.Column(int, checks=[pa.Check(lambda x: x >= 3)])},
    drop_invalid_rows=True,
)

schema.validate(df, lazy=True)
```

Dropping invalid rows with {class}`~pandera.api.pandas.array.SeriesSchema`:

```{code-cell} python
import pandas as pd
import pandera.pandas as pa


series = pd.Series([1, 2, 3])
schema = pa.SeriesSchema(
    int,
    checks=[pa.Check(lambda x: x >= 3)],
    drop_invalid_rows=True,
)

schema.validate(series, lazy=True)
```

Dropping invalid rows with {class}`~pandera.api.pandas.components.Column`:

```{code-cell} python
import pandas as pd
import pandera.pandas as pa


df = pd.DataFrame({"counter": [1, 2, 3]})
schema = pa.Column(
    int,
    name="counter",
    drop_invalid_rows=True,
    checks=[pa.Check(lambda x: x >= 3)]
)

schema.validate(df, lazy=True)
```

Dropping invalid rows with {class}`~pandera.api.pandas.model.DataFrameModel`:

```{code-cell} python
import pandas as pd
import pandera.pandas as pa


class MySchema(pa.DataFrameModel):
    counter: int = pa.Field(in_range={"min_value": 3, "max_value": 5})

    class Config:
        drop_invalid_rows = True


MySchema.validate(
    pd.DataFrame({"counter": [1, 2, 3, 4, 5, 6]}), lazy=True
)
```

```{note}
In order to use `drop_invalid_rows=True`, `lazy=True` must
be passed to the `schema.validate()`. {ref}`lazy-validation` enables all schema
errors to be collected and raised together, meaning all invalid rows can be dropped together.
This provides clear API for ensuring the validated dataframe contains only valid data.
```
