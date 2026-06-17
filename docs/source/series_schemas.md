---
file_format: mystnb
---

% pandera documentation for seriesschemas

```{currentmodule} pandera
```

(seriesschemas)=

# Series Schemas

The {class}`~pandera.api.pandas.array.SeriesSchema` class allows for the validation of pandas
`Series` objects, and are very similar to {ref}`columns<column>` and
{ref}`indexes<index>` described in {ref}`DataFrameSchemas<DataFrameSchemas>`.

```{code-cell} python
import pandas as pd
import pandera.pandas as pa

schema = pa.SeriesSchema(
    str,
    checks=[
        pa.Check(lambda s: s.str.startswith("foo")),
        pa.Check(lambda s: s.str.endswith("bar")),
        pa.Check(lambda x: len(x) > 3, element_wise=True)
    ],
    nullable=False,
    unique=False,
    name="my_series")

validated_series = schema.validate(
    pd.Series(["foobar", "foobar", "foobar"], name="my_series")
)

validated_series
```
