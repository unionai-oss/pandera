---
file_format: mystnb
---

% pandera documentation for Parsers

```{currentmodule} pandera
```

(parsers)=

# Parsers

*new in 0.19.0*

Parsers allow you to do some custom preprocessing on dataframes, columns, and
series objects. This is useful when you want to normalize, clip, or clean
data values before applying validation checks.

With parsers, you can codify and reuse preprocessing logic as part of the schema.
Note that this feature is optional, meaning that you can always do preprocessing
before calling `schema.validate` with the native dataframe API:

```{code-cell} python
import pandas as pd
import pandera as pa

schema = pa.DataFrameSchema({"a": pa.Column(int, pa.Check.ge(0))})
data = pd.DataFrame({"a": [1, 2, -1]})

# clip negative values
data["a"] = data["a"].clip(lower=0)
schema.validate(data)
```

Let's encode the preprocessing step as a parser:

```{code-cell} python
schema = pa.DataFrameSchema({
    "a": pa.Column(
        int,
        parsers=pa.Parser(lambda s: s.clip(lower=0)),
        checks=pa.Check.ge(0),
    )
})

data = pd.DataFrame({"a": [1, 2, -1]})
schema.validate(data)
```

```{important}
This feature is only supported with the `pandas` validation backend.
```

You can specify both dataframe-level and column-level parsers are available, where
dataframe-level parsers are performed before column-level parsers. With parsers
and checks, the validation process consists of the following steps:

1. dataframe-level parsing
2. column-level parsing
3. dataframe-level checks
4. column-level and index-level checks


## Parsing columns

{class}`~pandera.api.parsers.Parser` objects accept a function as a required
argument, which is expected to take a `Series` input and output a parsed
`Series`, for example:

```{code-cell} python
import numpy as np


schema = pa.DataFrameSchema({
    "sqrt_values": pa.Column(parsers=pa.Parser(lambda s: np.sqrt(s)))
})
schema.validate(pd.DataFrame({"sqrt_values": [1., 2., 3.]}))
```

Multiple parsers can be applied to a column:

```{important}
The order of `parsers` is preserved at validation time.
```

```{code-cell} python
schema = pa.DataFrameSchema({
    "string_numbers": pa.Column(
        str,
        parsers=[
            pa.Parser(lambda s: s.str.zfill(10)),
            pa.Parser(lambda s: s.str[2:]),
        ]
    ),
})

schema.validate(pd.DataFrame({"string_numbers": ["12345", "67890"]}))
```

## Parsing the dataframe

For any dataframe-wide preprocessing logic, you can specify the `parsers`
kwarg in the `DataFrameSchema` object.

```{code-cell} python
schema = pa.DataFrameSchema(
    parsers=pa.Parser(lambda df: df.transform("sqrt")),
    columns={
        "a": pa.Column(float),
        "b": pa.Column(float, parsers=pa.Parser(lambda s: s * -1)),
        "c": pa.Column(float, parsers=pa.Parser(lambda s: s + 1)),
    }
)

data = pd.DataFrame({
    "a": [2.0, 4.0, 9.0],
    "b": [2.0, 4.0, 9.0],
    "c": [2.0, 4.0, 9.0],
})

schema.validate(data)
```

## Parsers in `DataFrameModel`

We can write a `DataFrameModel` that's equivalent to the schema above with the
`@parse` and `@dataframe_parse`  decorators:

```{code-cell} python
class DFModel(pa.DataFrameModel):
    a: float
    b: float
    c: float

    @pa.dataframe_parse
    def sqrt(cls, df):
        return df.transform("sqrt")

    @pa.parse("b")
    def negate(cls, series):
        return series * -1

    @pa.parse("c")
    def plus_one(cls, series):
        return series + 1
```
