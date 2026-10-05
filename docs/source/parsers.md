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

(derived-columns)=

## Deriving columns

A parser can declare the columns it reads and the columns it produces, which
turns "this column is computed from that one" into something the schema states
rather than something a closure happens to do:

```{code-cell} python
import pandas as pd
import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {"body": pa.Column(str), "n_words": pa.Column(int)},
    parsers=[
        pa.Parser(
            lambda s: s.str.split().str.len(),
            source="body",
            target="n_words",
        )
    ],
)

schema.validate(pd.DataFrame({"body": ["a b c", "d e"]}))
```

The derived column is an ordinary column from there on: it is coerced to its
declared dtype, its checks run, and it is permitted under `strict=True`.

Declaring `source` and `target` buys three things beyond documentation.

**Errors name the parser.** A parser reading a column that is not there raises
{class}`~pandera.errors.ParserSourceError` identifying the parser, the missing
column, and the columns that *are* present — rather than a bare `KeyError`. A
parser that does not produce what it promised raises
{class}`~pandera.errors.ParserTargetError`.

**Ordering follows dependencies, not list position.** Parsers are sorted so
each runs after the parsers producing its sources, so the order they are
written in does not matter:

```{code-cell} python
schema = pa.DataFrameSchema(
    {"body": pa.Column(str), "a": pa.Column(int), "b": pa.Column(int)},
    parsers=[
        pa.Parser(lambda s: s * 2, source="a", target="b"),      # listed first
        pa.Parser(lambda s: s.str.len(), source="body", target="a"),
    ],
)

schema.validate(pd.DataFrame({"body": ["abc"]}))
```

A cycle among derived columns raises `SchemaInitError` naming the columns
involved. Parsers that declare neither a source nor a target keep their list
position and run first, so existing schemas are unaffected.

**Calling conventions.** With a single `source` and a single `target`, the
function is called with the source column as a `Series` and must return a
`Series`. Otherwise it is called with the selected source columns (or the whole
dataframe, when only `target` is declared) and must return a `DataFrame`
containing the targets.

:::{note}
`Parser` forwards unrecognized keyword arguments to the parser function, so
`source` and `target` previously arrived as function kwargs. Promoting them to
real parameters is a breaking change for a parser function that took a keyword
by either name.
:::

### Declaring derivation on the column

`source`/`target` on a schema-level `Parser` states the relationship, but it
still lives away from the column it describes. {class}`~pandera.api.pandas.components.ParsedColumn`
and {func}`~pandera.api.dataframe.model_components.ParsedField` put it on the
column itself:

```{code-cell} python
schema = pa.DataFrameSchema({
    "body": pa.Column(str),
    "n_words": pa.ParsedColumn(
        int,
        source="body",
        parser=lambda s: s.str.split().str.len(),
        checks=pa.Check.ge(1),
    ),
})

schema.validate(pd.DataFrame({"body": ["a b c", "d e"]}))
```

and the model equivalent:

```{code-cell} python
class Tickets(pa.DataFrameModel):
    body: str
    n_words: int = pa.ParsedField(
        parser=lambda s: s.str.split().str.len(),
        ge=1,
    )

    class Config:
        parser_source = "body"

Tickets.validate(pd.DataFrame({"body": ["a b c", "d e"]}))
```

`Config.parser_source` (or `DataFrameSchema(parser_source=...)`) is the default
for columns that do not name their own `source`. A column deriving from a
column the schema does not declare raises `SchemaInitError`.

The same thing can be written imperatively by giving `@pa.parser` a `source`,
in which case the named fields become what it produces:

```{code-cell} python
class Tickets(pa.DataFrameModel):
    body: str
    n_words: int

    @pa.parser("n_words", source="body")
    def count_words(cls, s):
        """Number of whitespace-separated tokens."""
        return s.str.split().str.len()
```

Without `source`, `@pa.parser` keeps its original meaning — a transform applied
to the named fields.

(column-parsers)=

### Parser objects

`parser=` also accepts an object implementing the
{class}`~pandera.api.parsers.ColumnParser` protocol, which is how a parser can
inspect the column it is filling and how independent columns can be filled in
one pass:

```python
class ColumnParser(Protocol):
    def bind(self, ctx: ParseContext) -> Callable: ...
    def batch_key(self, ctx: ParseContext) -> Hashable: ...
```

`bind` is called once when the schema is built and receives a `ParseContext`
describing the target column — its name, declared dtype, description,
nullability and checks, along with the resolved source columns and the schema
itself. Two consequences: a parser can derive its behavior from the column's
declared type instead of being told what it is producing, and raising
`SchemaInitError` from `bind` surfaces the problem before any data is touched.

`batch_key` groups parsers that can share work. Columns whose parsers return
equal non-`None` keys are handed to the class's `batch` classmethod and filled
by a single call; returning `None` opts out. This is what lets a schema declare
one derivation per column without paying for one pass per column.

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
