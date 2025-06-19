---
file_format: mystnb
---

(error-report)=

# Error Reports

*new in 0.19.0*

The pandera error report is a generalised machine-readable summary of failures
which occurred during schema validation. It is available for both `pysparksql` and
`pandas` objects.

By default, error reports are generated for both schema and data level validation,
but more granular control over schema or data only validations is available.

This is achieved by introducing configurable settings using environment variables
that allow you to control execution at three different levels:

1. `SCHEMA_ONLY`: perform schema validations only. It checks that data conforms
   to the schema definition, but does not perform any data-level validations on dataframe.
2. `DATA_ONLY`: perform data-level validations only. It validates that data
   conforms to the defined `checks`, but does not validate the schema.
3. `SCHEMA_AND_DATA`: (**default**) perform both schema and data level
   validations. It runs most exhaustive validation and could be compute intensive.

You can override default behaviour by setting an environment variable from terminal
before running the `pandera` process as:

```bash
export PANDERA_VALIDATION_DEPTH=SCHEMA_ONLY
```

This will be picked up by `pandera` to only enforce SCHEMA level validations.

## Error reports with `pandas`

To create an error report with pandas, you must specify `lazy=True` to allow all errors
to be aggregated and raised together as a `SchemaErrors`.

```{code-cell} python
import pandas as pd
import pandera.pandas as pa
import json

pandas_schema = pa.DataFrameSchema(
    {
        "color": pa.Column(str, pa.Check.isin(["red", "green", "blue"])),
        "length": pa.Column(int, pa.Check.gt(10)),
    }
)
data = [("red", 4), ("blue", 11), ("purple", 15), ("green", 39)]

df = pd.DataFrame(
    {
        "color": ["red", "blue", "purple", "green"],
        "length": [4, 11, 15, 39],
    }
)

try:
    pandas_schema.validate(df, lazy=True)
except pa.errors.SchemaErrors as e:
    print(json.dumps(e.message, indent=2))
```

## Error reports with `pyspark.sql`

Accessing the error report on a validated `pyspark` dataframe can be done via the
`errors` attribute on the `pandera` accessor.

```{code-cell} python
import pandera.pyspark as pa
import pyspark.sql.types as T
import json

from decimal import Decimal
from pyspark.sql import SparkSession
from pandera.pyspark import DataFrameModel

spark = SparkSession.builder.getOrCreate()

class PysparkPanderSchema(DataFrameModel):
    color: T.StringType() = pa.Field(isin=["red", "green", "blue"])
    length: T.IntegerType() = pa.Field(gt=10)

data = [("red", 4), ("blue", 11), ("purple", 15), ("green", 39)]

spark_schema = T.StructType(
    [
        T.StructField("color", T.StringType(), False),
        T.StructField("length", T.IntegerType(), False),
    ],
)

df = spark.createDataFrame(data, spark_schema)
df_out = PysparkPanderSchema.validate(check_obj=df)

print(json.dumps(dict(df_out.pandera.errors), indent=4))
```
