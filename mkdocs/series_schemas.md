# Series Schemas

The [SeriesSchema](./reference/generated/pandera.api.pandas.array.SeriesSchema.html) class allows for the validation of pandas
`Series` objects, and are very similar to [columns](./dataframe_schemas.html#column-validation) and
[indexes](./dataframe_schemas.html#index-validation) described in [DataFrameSchemas](./dataframe_schemas.html).

```python
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
