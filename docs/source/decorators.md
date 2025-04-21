---
file_format: mystnb
---

% pandera documentation for check_input and check_output decorators

```{currentmodule} pandera
```

(decorators)=

# Decorators for Pipeline Integration

If you have an existing data pipeline that uses pandas data structures,
you can use the {func}`~pandera.decorators.check_input` and {func}`~pandera.decorators.check_output` decorators
to easily check function arguments or returned variables from existing
functions.

## Check Input

Validates input pandas DataFrame/Series before entering the wrapped
function.

```{code-cell} python
import pandas as pd
import pandera.pandas as pa


df = pd.DataFrame({
    "column1": [1, 4, 0, 10, 9],
    "column2": [-1.3, -1.4, -2.9, -10.1, -20.4],
})

in_schema = pa.DataFrameSchema({
    "column1": pa.Column(
        int, pa.Check(lambda x: 0 <= x <= 10, element_wise=True)
    ),
    "column2": pa.Column(float, pa.Check(lambda x: x < -1.2)),
})

# by default, check_input assumes that the first argument is
# dataframe/series.
@pa.check_input(in_schema)
def preprocessor(dataframe):
    dataframe["column3"] = dataframe["column1"] + dataframe["column2"]
    return dataframe

preprocessed_df = preprocessor(df)
print(preprocessed_df)
```

You can also provide the argument name as a string

```{code-cell} python
@pa.check_input(in_schema, "dataframe")
def preprocessor(dataframe):
    ...
```

Or an integer representing the index in the positional arguments.

```{code-cell} python
@pa.check_input(in_schema, 1)
def preprocessor(foo, dataframe):
    ...
```

## Check Output

The same as `check_input`, but this decorator checks the output
DataFrame/Series of the decorated function.

```{code-cell} python
import pandas as pd
import pandera.pandas as pa


preprocessed_df = pd.DataFrame({
    "column1": [1, 4, 0, 10, 9],
})

# assert that all elements in "column1" are zero
out_schema = pa.DataFrameSchema({
    "column1": pa.Column(int, pa.Check(lambda x: x == 0))
})


# by default assumes that the pandas DataFrame/Schema is the only output
@pa.check_output(out_schema)
def zero_column_1(df):
    df["column1"] = 0
    return df


# you can also specify in the index of the argument if the output is list-like
@pa.check_output(out_schema, 1)
def zero_column_1_arg(df):
    df["column1"] = 0
    return "foobar", df


# or the key containing the data structure to verify if the output is dict-like
@pa.check_output(out_schema, "out_df")
def zero_column_1_dict(df):
    df["column1"] = 0
    return {"out_df": df, "out_str": "foobar"}


# for more complex outputs, you can specify a function
@pa.check_output(out_schema, lambda x: x[1]["out_df"])
def zero_column_1_custom(df):
    df["column1"] = 0
    return ("foobar", {"out_df": df})


zero_column_1(preprocessed_df)
zero_column_1_arg(preprocessed_df)
zero_column_1_dict(preprocessed_df)
zero_column_1_custom(preprocessed_df)
```

## Check IO

For convenience, you can also use the {func}`~pandera.decorators.check_io`
decorator where you can specify input and output schemas more concisely:

```{code-cell} python
import pandas as pd
import pandera.pandas as pa


df = pd.DataFrame({
    "column1": [1, 4, 0, 10, 9],
    "column2": [-1.3, -1.4, -2.9, -10.1, -20.4],
})

in_schema = pa.DataFrameSchema({
    "column1": pa.Column(int),
    "column2": pa.Column(float),
})

out_schema = in_schema.add_columns({"column3": pa.Column(float)})

@pa.check_io(df1=in_schema, df2=in_schema, out=out_schema)
def preprocessor(df1, df2):
    return (df1 + df2).assign(column3=lambda x: x.column1 + x.column2)

preprocessed_df = preprocessor(df, df)
print(preprocessed_df)
```

## Decorate Functions and Coroutines

*All* pandera decorators work on synchronous as well as asynchronous code, on both bound and unbound
functions/coroutines. For example, one can use the same decorators on:

- sync/async functions
- sync/async methods
- sync/async class methods
- sync/async static methods

All decorators work on sync/async regular/class/static methods of metaclasses as well.

```{code-cell} python
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

class Schema(pa.DataFrameModel):
    col1: Series[int]

    class Config:
        strict = True

@pa.check_types
async def coroutine(df: DataFrame[Schema]) -> DataFrame[Schema]:
    return df

@pa.check_types
async def function(df: DataFrame[Schema]) -> DataFrame[Schema]:
    return df

class SomeClass:
    @pa.check_output(Schema.to_schema())
    async def regular_coroutine(self, df) -> DataFrame[Schema]:
        return df

    @classmethod
    @pa.check_input(Schema.to_schema(), "df")
    async def class_coroutine(cls, df):
        return Schema.validate(df)

    @staticmethod
    @pa.check_io(df=Schema.to_schema(), out=Schema.to_schema())
    def static_method(df):
        return df
```
