---
file_format: mystnb
---

% pandera documentation for datafusion

```{currentmodule} pandera.datafusion
```

(datafusion)=

# Data Validation with DataFusion

[DataFusion](https://datafusion.apache.org/python/) is an Arrow-native query
engine. Its `datafusion.DataFrame` is a lazy query plan with an Arrow schema,
and pandera supports validating it directly so you get dtype checking, value
checks and class-based models without leaving the plan.

## Installation

```bash
pip install 'pandera[datafusion]'
```

Validation is performed by pandera's
{ref}`narwhals backend <narwhals-backend>`, which reaches DataFusion through
the [narwhals-datafusion](https://pypi.org/project/narwhals-datafusion/)
plugin. Both are installed alongside `datafusion`.

## `DataFrameSchema`

Import the DataFusion entry point and define a schema as you would for any
other backend:

```{code-cell} python
import pyarrow
from datafusion import SessionContext

import pandera.datafusion as pa

schema = pa.DataFrameSchema(
    {
        "state": pa.Column(str),
        "city": pa.Column(str),
        "price": pa.Column(int, pa.Check.in_range(5, 20)),
    }
)

ctx = SessionContext()
df = ctx.from_arrow(
    pyarrow.table(
        {
            "state": ["FL", "FL", "CA", "CA"],
            "city": ["Orlando", "Miami", "Los Angeles", "San Francisco"],
            "price": [8, 12, 10, 16],
        }
    )
)

schema.validate(df)
```

`validate` returns a `datafusion.DataFrame`, so schemas drop into an existing
query pipeline without changing types.

## `DataFrameModel`

```{code-cell} python
class Schema(pa.DataFrameModel):
    state: str
    city: str
    price: int = pa.Field(in_range={"min_value": 5, "max_value": 20})


Schema.validate(df)
```

Annotate function signatures with `pandera.typing.datafusion.DataFrame` and
use {func}`~pandera.decorators.check_types` to validate inputs and outputs:

```{code-cell} python
from pandera.typing.datafusion import DataFrame


@pa.check_types
def transform(df: DataFrame[Schema]) -> DataFrame[Schema]:
    return df


transform(df)
```

## Validation depth

As with a `polars.LazyFrame` or an `ibis.Table`, running a data-level check on
a `datafusion.DataFrame` executes it. By default pandera therefore runs only
schema-level checks (column presence and data types), which read the schema
without executing anything.

To also run data-level checks, set `PANDERA_VALIDATION_DEPTH=SCHEMA_AND_DATA`
or use {func}`~pandera.config.config_context`:

```{code-cell} python
from pandera.config import ValidationDepth, config_context

invalid_df = ctx.from_arrow(
    pyarrow.table({"state": ["FL"], "city": ["Miami"], "price": [100]})
)

with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
    try:
        schema.validate(invalid_df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        print(exc.failure_cases)
```

The failing rows are reported as a `pyarrow.Table`, DataFusion's native
exchange format.

The remaining examples on this page run data-level checks:

```{code-cell} python
pa.set_config(validation_depth=ValidationDepth.SCHEMA_AND_DATA)
```

## Supported data types

DataFusion data types are Arrow data types. Columns accept native pyarrow
types, Python builtins, and their string aliases — all resolve to the same
underlying pandera datatype:

```{code-cell} python
pa.DataFrameSchema(
    {
        "a": pa.Column(pyarrow.int64()),
        "b": pa.Column(int),
        "c": pa.Column("int64"),
    }
)
```

Parametrized pyarrow types such as `pyarrow.timestamp("us")` and
`pyarrow.list_(pyarrow.int32())` are supported.

`pa.Column(str)` accepts every Arrow string type (`string`, `large_string` and
`string_view`), so it also matches the strings produced by SQL casts.

## Custom checks

Check functions receive a
{class}`~pandera.api.datafusion.types.DataFusionData` container holding the
native `datafusion.DataFrame` and the column key, mirroring `PolarsData` and
`IbisData` on the other backends. Return a `datafusion.Expr` that evaluates to
a boolean for each row:

```{code-cell} python
from datafusion import col, lit

schema = pa.DataFrameSchema(
    {"price": pa.Column(int, pa.Check(lambda data: col(data.key) > lit(0)))}
)
schema.validate(df)
```

A check function taking two positional arguments receives the native
dataframe and the key directly, i.e. `check_fn(df, key)`.

Dataframe-level checks receive the same container with the key set to `"*"`:

```{code-cell} python
schema = pa.DataFrameSchema(
    {"price": pa.Column(int)},
    checks=pa.Check(lambda data: col("price") < lit(100)),
)
schema.validate(df)
```

The expression may also be an aggregate, which is evaluated to a single
pass/fail result:

```{code-cell} python
from datafusion import functions as f

def max_price_below_100(data):
    return f.max(col(data.key)) < lit(100)


schema = pa.DataFrameSchema(
    {"price": pa.Column(int, pa.Check(max_price_below_100))}
)
schema.validate(df)
```

A check may also return a Python `bool`, for example
`df.filter(col(key) <= lit(0)).count() == 0`.

## Limitations

These follow from a `datafusion.DataFrame` being a lazy query plan.

- **`element_wise=True` checks are not supported.** Use a vectorized check
  instead.
- **`tail=` is not supported** in `validate`; use `head=` to validate a subset
  of rows.
- **Custom checks cannot return a `datafusion.DataFrame`.** Return a
  `datafusion.Expr` or a Python `bool`.
- **`lazy=True` may collect the whole dataframe** to report failures of
  nullability and custom expression checks.
- **A missing-column error runs the plan** to preview the first rows in its
  message.
- **Data format conversion reads through a new `SessionContext`**, not the one
  the caller is using.
