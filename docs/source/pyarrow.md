---
file_format: mystnb
---

% pandera documentation for pyarrow

```{currentmodule} pandera.pyarrow
```

(pyarrow)=

# Data Validation with PyArrow

*new in 0.33.0*

[PyArrow](https://arrow.apache.org/docs/python/) tables are a common exchange
format between systems, and `pyarrow.Table` carries its own schema. Pandera
supports validating them directly so you get dtype checking, value checks and
class-based models on top of that schema.

## Installation

```bash
pip install 'pandera[pyarrow]'
```

Validation is performed by pandera's
{ref}`narwhals backend <narwhals-backend>`, so `narwhals` is installed
alongside `pyarrow`.

## `DataFrameSchema`

Import the pyarrow entry point and define a schema as you would for any other
backend:

```{code-cell} python
import pyarrow
import pandera.pyarrow as pa

schema = pa.DataFrameSchema(
    {
        "state": pa.Column(str),
        "city": pa.Column(str),
        "price": pa.Column(int, pa.Check.in_range(5, 20)),
    }
)

table = pyarrow.table(
    {
        "state": ["FL", "FL", "CA", "CA"],
        "city": ["Orlando", "Miami", "Los Angeles", "San Francisco"],
        "price": [8, 12, 10, 16],
    }
)

schema.validate(table)
```

`validate` returns a `pyarrow.Table`, so schemas drop into an existing pipeline
without changing types.

## `DataFrameModel`

```{code-cell} python
class Schema(pa.DataFrameModel):
    state: str
    city: str
    price: int = pa.Field(in_range={"min_value": 5, "max_value": 20})


Schema.validate(table)
```

Annotate function signatures with `pandera.typing.pyarrow.Table` and use
{func}`~pandera.decorators.check_types` to validate inputs and outputs:

```{code-cell} python
from pandera.typing.pyarrow import Table


@pa.check_types
def transform(df: Table[Schema]) -> Table[Schema]:
    return df


transform(table)
```

## Supported data types

Columns accept native pyarrow types, python builtins, and their string
aliases — all resolve to the same underlying pandera datatype:

```{code-cell} python
pa.DataFrameSchema(
    {
        "a": pa.Column(pyarrow.int64()),
        "b": pa.Column(int),
        "c": pa.Column("int64"),
    }
)
```

Parametrized pyarrow types such as `pyarrow.timestamp("us")`,
`pyarrow.decimal128(10, 2)` and `pyarrow.list_(pyarrow.int32())` are supported.

## Custom checks

Check functions receive a {class}`~pandera.api.pyarrow.types.PyArrowData`
container holding the native table and the column key, mirroring `PolarsData`
and `IbisData` on the other backends:

```{code-cell} python
import pyarrow.compute as pc

schema = pa.DataFrameSchema(
    {"price": pa.Column(int, pa.Check(lambda data: pc.greater(data.table[data.key], 0)))}
)
schema.validate(table)
```

A check function taking two positional arguments receives the native table and
the key directly, i.e. `check_fn(table, key)`.

## Validation depth

Unlike a `polars.LazyFrame` or an `ibis.Table`, a `pyarrow.Table` is always
fully materialized in memory, so pandera runs both schema-level and data-level
checks by default. Set `PANDERA_VALIDATION_DEPTH=SCHEMA_ONLY` (or use
{func}`~pandera.config.config_context`) to skip data-level checks.

## Limitations

- **`coerce=True` is not applied.** Coercion is not yet implemented in the
  narwhals backend that serves pyarrow; a `SchemaWarning` is emitted and a
  dtype mismatch is reported as a `WRONG_DATATYPE` error instead.
- **Data synthesis strategies are not supported**, matching the polars and ibis
  backends.
