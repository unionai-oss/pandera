# pandera

[![Build Status](https://travis-ci.org/cosmicBboy/pandera.svg?branch=master)](https://travis-ci.org/cosmicBboy/pandera)
[![PyPI version shields.io](https://img.shields.io/pypi/v/pandera.svg)](https://pypi.org/project/pandera/)
[![PyPI license](https://img.shields.io/pypi/l/pandera.svg)](https://pypi.python.org/pypi/pandera/)

**Supports:** python 2.7, 3.5, 3.6

A light-weight and flexible validation package for
[pandas](http://pandas.pydata.org) data structures.


## Why?

Because pandas data structures hide a lot of information, and explicitly
validating them in production-critical or reproducible research settings is
a good idea.

And it also makes it easier to review pandas code :)


## Install

```
pip install pandera
```

## Example Usage

### `DataFrameSchema`

```python
import pandas as pd

from pandera import Column, DataFrameSchema, Float, Int, String, Check


# validate columns
schema = DataFrameSchema({
    # the check function expects a series argument and should output a boolean
    # or a boolean Series.
    "column1": Column(Int, Check(lambda s: s <= 10)),
    "column2": Column(Float, Check(lambda s: s < -1.2)),
    # you can provide a list of validators
    "column3": Column(String, [
        Check(lambda s: s.str.startswith("value_")),
        Check(lambda s: s.str.split("_", expand=True).shape[1] == 2)
    ]),
})

# alternatively, you can pass strings representing the legal pandas datatypes:
# http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
schema = DataFrameSchema({
    "column1": Column("int64", Check(lambda s: s <= 10)),
    ...
})

df = pd.DataFrame({
    "column1": [1, 4, 0, 10, 9],
    "column2": [-1.3, -1.4, -2.9, -10.1, -20.4],
    "column3": ["value_1", "value_2", "value_3", "value_2", "value_1"]
})

validated_df = schema.validate(df)
print(validated_df)

#     column1  column2  column3
#  0        1     -1.3  value_1
#  1        4     -1.4  value_2
#  2        0     -2.9  value_3
#  3       10    -10.1  value_2
#  4        9    -20.4  value_1
```


### Vectorized vs. Element-wise Checks

By default, the functions passed into `Check`s are expected to have the
following signature: `pd.Series -> bool|pd.Series[bool]`. For the `Check` to
pass, all of the elements in the boolean series must evaluate to `True`.

If you want to make atomic checks for each element in the Column, then you
can provide the `element_wise=True` keyword argument:


```python
import pandas as pd

from pandera import Check, Column, DataFrameSchema, Int

schema = DataFrameSchema({
    "a": Column(Int, [
        # a vectorized check that returns a bool
        Check(lambda s: s.mean() > 5, element_wise=False),
        # a vectorized check that returns a boolean series
        Check(lambda s: s > 0, element_wise=False),
        # an element-wise check that returns a bool
        Check(lambda x: x > 0, element_wise=True),
    ]),
})

df = pd.DataFrame({"a": [4, 4, 5, 6, 6, 7, 8, 9]})
schema.validate(df)
```

By default `element_wise=False` so that you can take advantage of the speed
gains provided by the `pandas.Series` API by writing vectorized checks.


#### Validating DataFrame Index

You can also specify an `Index` in the `DataFrameSchema`.

```python
import pandas as pd

from pandera import Column, DataFrameSchema, Index, Int, String, Check

schema = DataFrameSchema(
    columns={"a": Column(Int)},
    index=Index(
        String,
        Check(lambda x: x.startswith("index_"))))

df = pd.DataFrame({"a": [1, 2, 3]}, index=["index_1", "index_2", "index_3"])

print(schema.validate(df))

#          a
# index_1  1
# index_2  2
# index_3  3


df.index = ["foo1", "foo2", "foo3"]
schema.validate(df)

# SchemaError: <Schema Index> failed element-wise validator 0:
# <lambda>
# failure cases:
#              index  count
# failure_case
# foo1           [0]      1
# foo2           [1]      1
# foo3           [2]      1
```


#### Informative Errors

If the dataframe does not pass validation checks, `pandera` provides useful
error messages. An `error` argument can also be supplied to `Check` for
custom error messages.

```python
import pandas as pd

from pandera import Column, DataFrameSchema, Int, Check

simple_schema = DataFrameSchema({
    "column1": Column(
        Int, Check(lambda x: 0 <= x <= 10, error="range checker [0, 10]"))
})

# validation rule violated
fail_check_df = pd.DataFrame({
    "column1": [-20, 5, 10, 30],
})

simple_schema.validate(fail_check_df)

# schema.SchemaError: series failed element-wise validator 0:
# <lambda>: range checker [0, 10]
# failure cases:
#              index  count
# failure_case
# -20            [0]      1
#  30            [3]      1


# column name mis-specified
wrong_column_df = pd.DataFrame({
    "foo": ["bar"] * 10,
    "baz": [1] * 10
})

simple_schema.validate(wrong_column_df)

#  SchemaError: column 'column1' not in dataframe
#     foo  baz
#  0  bar    1
#  1  bar    1
#  2  bar    1
#  3  bar    1
#  4  bar    1
```

### Nullable Columns

By default, SeriesSchema/Column objects assume that values are not nullable.
In order to accept null values, you need to explicitly specify `nullable=True`,
or else you'll get an error.

```python
import numpy as np
import pandas as pd

from pandera import Check, Column, DataFrameSchema, Int

df = pd.DataFrame({"column1": [5, 1, np.nan]})

non_null_schema = DataFrameSchema({
    "column1": Column(Int, Check(lambda x: x > 0))
})

non_null_schema.validate(df)

# SchemaError: non-nullable series contains null values: {2: nan}
```

**NOTE:** Due to a known limitation in
[pandas](http://pandas.pydata.org/pandas-docs/stable/gotchas.html#support-for-integer-na),
integer arrays cannot contain `NaN` values, so this schema will return a
dataframe where `column1` is of type `float`.

```python
from pandera import Check, Column, DataFrameSchema, Int

df = ...
null_schema = DataFrameSchema({
    "column1": Column(Int, Check(lambda x: x > 0), nullable=True)
})

null_schema.validate(df)

#    column1
# 0      5.0
# 1      1.0
# 2      NaN
```


### Coercing Types on Columns

If you specify `Column(dtype, ..., coerce=True)` as part of the DataFrameSchema
definition, calling `schema.validate` will first coerce the column into the
specified `dtype`.

```python
import pandas as pd

from pandera import Column, DataFrameSchema, String

df = pd.DataFrame({"column1": [1, 2, 3]})
schema = DataFrameSchema({"column1": Column(String, coerce=True)})

validated_df = schema.validate(df)
assert isinstance(validated_df.column1.iloc[0], str)
```

Note the special case of integers columns not supporting `nan` values. In this
case, `schema.validate` will complain if `coerce == True` and null
values are allowed in the column.

The best way to handle this case is to simply specify the column as a `Float`
or `Object`.

```python
import pandas as pd

from pandera import Column, DataFrameSchema, Float, Int, Object

df = pd.DataFrame({"column1": [1., 2., 3, pd.np.nan]})
schema = DataFrameSchema({"column1": Column(Int, coerce=True, nullable=True)})

validated_df = schema.validate(df)
# ValueError: cannot convert float NaN to integer


schema_object = DataFrameSchema({
    "column1": Column(Object, coerce=True, nullable=True)})
schema_float = DataFrameSchema({
    "column1": Column(Float, coerce=True, nullable=True)})

schema_object.validate(df).dtypes
# column1    object


schema_float.validate(df).dtypes
# column1    float64
```

If you want to coerce all of the columns specified in the `DataFrameSchema`,
you can specify the `coerce` argument with `DataFrameSchema(..., coerce=True)`.

### Required Columns

By default all columns specified in the schema are required, meaning that if a
column is missing in the input dataframe an exception will be thrown. If you
want to make a column optional specify `required=False` in the column
constructor:

```python
import pandas as pd

from pandera import Column, DataFrameSchema, Int, String

df = pd.DataFrame({"column2": ["hello", "pandera"]})
schema = DataFrameSchema({
    "column1": Column(Int, required=False),
    "column2": Column(String)
})

validated_df = schema.validate(df)
# list(validated_df.columns) == ["column2"]

```

### Handling of Dataframe Columns not in the Schema

By default, columns that aren't specified in the schema aren't checked. If you
want to check that the dataframe *only* contains columns in the schema, specify
`strict=True`:

```python
import pandas as pd
from pandera import Column, DataFrameSchema, Int

schema = DataFrameSchema({"column1": Column(Int, nullable=True)},
                         strict=True)
df = pd.DataFrame({"column2": [1, 2, 3]})

schema.validate(df)

# SchemaError: column 'column2' not in DataFrameSchema {'column1': <Schema Column: 'None' type=int64>}
```

### `SeriesSchema`

```python
import pandas as pd

from pandera import Check, SeriesSchema, String

# specify multiple validators
schema = SeriesSchema(String, [
    Check(lambda x: "foo" in x),
    Check(lambda x: x.endswith("bar")),
    Check(lambda x: len(x) > 3)])

schema.validate(pd.Series(["1_foobar", "2_foobar", "3_foobar"]))

#  0    1_foobar
#  1    2_foobar
#  2    3_foobar
#  dtype: object
```

## Plugging into Existing Workflows

If you have an existing data pipeline that uses pandas data structures, you can
use the `check_input` and `check_output` decorators to easily check
function arguments or returned variables from existing functions.


### `check_input`

Validates input pandas DataFrame/Series before entering the wrapped function.

```python
import pandas as pd

from pandera import DataFrameSchema, Column, Check, Int, Float, check_input


df = pd.DataFrame({
    "column1": [1, 4, 0, 10, 9],
    "column2": [-1.3, -1.4, -2.9, -10.1, -20.4],
})

in_schema = DataFrameSchema({
    "column1": Column(Int, Check(lambda x: 0 <= x <= 10)),
    "column2": Column(Float, Check(lambda x: x < -1.2)),
})


# by default, assumes that the first argument is dataframe/series.
@check_input(in_schema)
def preprocessor(dataframe):
    dataframe["column4"] = dataframe["column1"] + dataframe["column2"]
    return dataframe


# or you can provide the argument name as a string
@check_input(in_schema, "dataframe")
def preprocessor(dataframe):
    ...


# or integer representing index in the positional arguments.
@check_input(in_schema, 1)
def preprocessor(foo, dataframe):
    ...


preprocessed_df = preprocessor(df)
print(preprocessed_df)

#  Output:
#     column1  column2  column3  column4
#  0        1     -1.3  value_1     -0.3
#  1        4     -1.4  value_2      2.6
#  2        0     -2.9  value_3     -2.9
#  3       10    -10.1  value_2     -0.1
#  4        9    -20.4  value_1    -11.4
```


### `check_output`

The same as `check_input`, but this decorator checks the output
DataFrame/Series of the decorated function.

```python
from pandera import DataFrameSchema, Column, Check, Int, check_output


preprocessed_df = ...

# assert that all elements in "column1" are zero
out_schema = DataFrameSchema({
    "column1": Column(Int, Check(lambda x: x == 0))
})


# by default assumes that the pandas DataFrame/Schema is the only output
@check_output(out_schema)
def zero_column_1(df):
    df["column1"] = 0
    return df


# you can also specify in the index of the argument if the output is list-like
@check_output(out_schema, 1)
def zero_column_1_arg(df):
    df["column1"] = 0
    return "foobar", df


# or the key containing the data structure to verify if the output is dict-like
@check_output(out_schema, "out_df")
def zero_column_1_dict(df):
    df["column1"] = 0
    return {"out_df": df, "out_str": "foobar"}


# for more complex outputs, you can specify a function
@check_output(out_schema, lambda x: x[1]["out_df"])
def zero_column_1_custom(df):
    df["column1"] = 0
    return ("foobar", {"out_df": df})


zero_column_1(preprocessed_df)
zero_column_1_arg(preprocessed_df)
zero_column_1_dict(preprocessed_df)
zero_column_1_custom(preprocessed_df)
```

## Advanced Features

### Column Check Groups

`Column` `Check`s support grouping by a different column so that you can make
assertions about subsets of the `Column` of interest. This changes the function
signature of the `Check` function so that its input is a dict where keys
are the group names and keys are subsets of the `Column` series.

Specifying `groupby` as a column name, list of column names, or callable
changes the expected signature of the `Check` function argument to
`dict[Any|tuple[Any], Series] -> bool|Series[bool]` where the dict keys are
the discrete keys in the `groupby` columns.

```python
import pandas as pd

from pandera import DataFrameSchema, Column, Check, Bool, Float, Int, String


schema = DataFrameSchema({
    "height_in_feet": Column(Float, [
        # groupby as a single column
        Check(lambda g: g[False].mean() > 6, groupby="age_less_than_20"),
        # define multiple groupby columns
        Check(lambda g: g[(True, "F")].sum() == 9.1,
              groupby=["age_less_than_20", "sex"]),
        # groupby as a callable with signature (DataFrame) -> DataFrameGroupBy
        Check(lambda g: g[(False, "M")].median() == 6.75,
              groupby=lambda df: (
                df
                .assign(age_less_than_15=lambda d: d["age"] < 15)
                .groupby(["age_less_than_15", "sex"]))),
    ]),
    "age": Column(Int, Check(lambda s: s > 0)),
    "age_less_than_20": Column(Bool),
    "sex": Column(String, Check(lambda s: s.isin(["M", "F"])))
})

df = (
    pd.DataFrame({
        "height_in_feet": [6.5, 7, 6.1, 5.1, 4],
        "age": [25, 30, 21, 18, 13],
        "sex": ["M", "M", "F", "F", "F"]
    })
    .assign(age_less_than_20=lambda x: x["age"] < 20)
)

schema.validate(df)
```

In the above example we define a `DataFrameSchema` with column checks for
`height_in_feet` using a single column, multiple columns, and a more complex
groupby function that creates a new column `age_less_than_15` on the fly.



### Hypothesis Testing

`Column` `Hypothesis` tests support testing different column so that assertions
can be made about the relationships between `Column`s.

`Hypothesis` contains built in methods, which can be called as in this example:

```python
import pandas as pd

from pandera import Column, DataFrameSchema, Float, Check, String, Hypothesis

from scipy import stats

df = (
    pd.DataFrame({
        "height_in_feet": [6.5, 7, 6.1, 5.1, 4],
        "sex": ["M", "M", "F", "F", "F"]
    })
)

schema = DataFrameSchema({
    "height_in_feet": Column(Float, [
        Hypothesis.two_sample_ttest(groupby="sex",
                                    groups=["M", "F"],
                                    relationship="greater_than",
                                    alpha = 0.05
                                    ),
    ]),
    "sex": Column(String)
})

schema.validate(df)

# SchemaError: <Schema Column: 'height_in_feet' type=float64> failed series validator 0: _check_fn
```

`Hypothesis` also supports passing custom `test`s and `relationship`s. This
enables the user to use non-built in functions as follows: 

```python
schema = DataFrameSchema({
    "height_in_feet": Column(Float, [
        Hypothesis(test=stats.ttest_ind,
                   groupby="sex",
                   groups=["M", "F"],
                   relationship="greater_than",
                   relationship_kwargs={"alpha":0.5}
                  ),
    ]),
    "sex": Column(String)
})
```

## Tests


```
pip install pytest
pytest tests
```

## Contributing to pandera [![GitHub contributors](https://img.shields.io/github/contributors/cosmicBboy/pandera.svg)](https://github.com/cosmicBboy/pandera/graphs/contributors)
All contributions, bug reports, bug fixes, documentation improvements, enhancements and ideas are welcome.

A detailed overview on how to contribute can be found in the [contributing guide](https://github.com/cosmicBboy/pandera/blob/master/.github/CONTRIBUTING.md) on GitHub.

## Issues


Go [here](https://github.com/cosmicBboy/pandera/issues) to submit feature
requests or bugfixes.
