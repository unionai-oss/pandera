---
file_format: mystnb
---

% pandera documentation for Parsers

```{currentmodule} pandera
```

(parsers)=

# Preprocessing with Parsers

*new in 0.19.0*

Parsers allow you to do some custom preprocessing on dataframes, columns, and
series objects before running the validation checks. This is useful when you want
to normalize, clip, or otherwise clean data values before applying validation
checks.

:::{important}
This feature is only available in the pandas validation backend.
:::

## Parsing versus validation

Pandera distinguishes between data validation and parsing. Validation is the act
of verifying whether data follows some set of constraints, whereas parsing transforms
raw data into some desired set of constraints.

Pandera ships with a few core parsers that you may already be familiar with:

- `coerce=True` will convert the datatypes of the incoming data to validate.
  This option is available in both {class}`~pandera.api.pandas.container.DataFrameSchema`
  and {class}`~pandera.api.pandas.components.Column` objects. See {ref}`here <coerced>`
  for more details.
- `strict="filter"` will remove columns in the data that are not specified in
  the {class}`~pandera.api.pandas.container.DataFrameSchema`. See {ref}`here <strict>`
  for more details.
- `add_missing_columns=True` will add missing columns to the data if the
  {class}`~pandera.api.pandas.components.Column` is nullable or specifies a
   default value. See {ref}`here <adding-missing-columns>`.

The {class}`~pandera.api.parsers.Parser` abstraction allows you to specify any
arbitrary transform that occurs before validation so that you can codify
and standardize the preprocessing steps needed to get your raw data into a valid
state.

```{important}
This feature is currently only supported with the `pandas` validation backend.
```

With parsers, you can codify and reuse preprocessing logic as part of the schema.
Note that this feature is optional, meaning that you can always do preprocessing
before calling `schema.validate` with the native dataframe API:

```{code-cell} python
import pandas as pd
import pandera.pandas as pa

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

You can specify both dataframe- and column-level parsers, where
dataframe-level parsers are performed before column-level parsers. Assuming
that a schema contains parsers and checks, the validation process consists of
the following steps, in order:

1. dataframe-level parsing
2. for each column: column-level parsing, then column-level checks
3. index-level parsing, then index-level checks
4. dataframe-level checks

This ordering is intentional: dataframe-level checks are meant to be applied
only once all of the parsing steps have finished, so that they validate the
data in the state that represents the user's best effort at parsing the raw
dataframe into a valid one. Since checks are independent of one another by
design, there's no functional difference between running dataframe-level
checks before or after column- and index-level checks, but running
dataframe-level checks *before* the column-level parsers have run would break
the assumption that checks operate on already-parsed data.

```{note}
`coerce=True` is itself a built-in parser (see {ref}`coerced` for more details
on dtype coercion), so coercion happens during the parsing phase described
above, before any checks run. A column's own parsers run *before* that column
is coerced, so if a value needs to be transformed into a coercible form —
for example, normalizing `"1,0"` to `"1.0"` before it can be coerced to a
float — you can do that either in a dataframe-level parser or in a parser on
the column itself.
```

You can verify the order of operations for yourself by adding some print
statements to your parsers and checks:

```{code-cell} python
schema = pa.DataFrameSchema(
    parsers=pa.Parser(lambda df: print("=== dataframe parser ===") or df),
    columns={
        "a": pa.Column(
            int,
            parsers=pa.Parser(lambda s: print("=== column a parser ===") or s),
            checks=pa.Check(lambda s: print("=== column a check ===") or True),
        ),
        "b": pa.Column(
            int,
            parsers=pa.Parser(lambda s: print("=== column b parser ===") or s),
            checks=pa.Check(lambda s: print("=== column b check ===") or True),
        ),
    },
    checks=pa.Check(lambda df: print("=== dataframe check ===") or True),
)

data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
schema.validate(data)
```

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

```{note}
Similar to the column-level parsers, you can also provide a list of `Parser`s
at the dataframe level.
```

## Parsers in `DataFrameModel`

We can write a `DataFrameModel` that's equivalent to the schema above with the
{py:func}`~pandera.api.dataframe.model_components.parse` and
{py:func}`~pandera.api.dataframe.model_components.dataframe_parse`  decorators:

```{code-cell} python
class DFModel(pa.DataFrameModel):
    a: float
    b: float
    c: float

    @pa.dataframe_parser
    def sqrt(cls, df):
        return df.transform("sqrt")

    @pa.parser("b")
    def negate(cls, series):
        return series * -1

    @pa.parser("c")
    def plus_one(cls, series):
        return series + 1
```
