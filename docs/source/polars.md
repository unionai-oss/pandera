---
file_format: mystnb
---

```{currentmodule} pandera.polars
```

(polars)=

# Data Validation with Polars

*new in 0.19.0*

[Polars](https://docs.pola.rs/) is a blazingly fast DataFrame library for
manipulating structured data. Since the core is written in Rust, you get the
performance of C/C++ while providing SDKs in other languages like Python.

## Usage

With the polars integration, you can define pandera schemas to validate polars
dataframes in Python. First, install `pandera` with the `polars` extra:

```bash
pip install 'pandera[polars]'
```

:::{note}
As of `pandera >= 0.21.0`, only `polars >= 1.0.0` is supported.
:::

:::{important}
If you're on an Apple Silicon machine, you'll need to install polars via
`pip install polars-lts-cpu`.

You may have to delete `polars` if it's already installed:

```
pip uninstall polars
pip install polars-lts-cpu
```

:::

Then you can use pandera schemas to validate polars dataframes. In the example
below we'll use the {ref}`class-based API <dataframe-models>` to define a
{py:class}`~pandera.api.polars.model.DataFrameModel`, which we then use to
validate a {py:class}`polars.LazyFrame` object.

```{code-cell} python
import pandera.polars as pa
import polars as pl


class Schema(pa.DataFrameModel):
    state: str
    city: str
    price: int = pa.Field(in_range={"min_value": 5, "max_value": 20})


lf = pl.LazyFrame(
    {
        'state': ['FL','FL','FL','CA','CA','CA'],
        'city': [
            'Orlando',
            'Miami',
            'Tampa',
            'San Francisco',
            'Los Angeles',
            'San Diego',
        ],
        'price': [8, 12, 10, 16, 20, 18],
    }
)
Schema.validate(lf).collect()
```

You can also use the {py:func}`~pandera.decorators.check_types` decorator to
validate polars LazyFrame function annotations at runtime:

```{code-cell} python
from pandera.typing.polars import LazyFrame

@pa.check_types
def function(lf: LazyFrame[Schema]) -> LazyFrame[Schema]:
    return lf.filter(pl.col("state").eq("CA"))

function(lf).collect()
```

And of course, you can use the object-based API to define a
{py:class}`~pandera.api.polars.container.DataFrameSchema`:

```{code-cell} python
schema = pa.DataFrameSchema({
    "state": pa.Column(str),
    "city": pa.Column(str),
    "price": pa.Column(int, pa.Check.in_range(min_value=5, max_value=20))
})
schema.validate(lf).collect()
```

You can also validate {py:class}`polars.DataFrame` objects, which are objects that
execute computations eagerly. Under the hood, `pandera` will convert
the `polars.DataFrame` to a `polars.LazyFrame` before validating it. This is done
so that the internal validation routine that pandera implements can take
advantage of the optimizations that the polars lazy API provides.

```{code-cell} python
df: pl.DataFrame = lf.collect()
schema.validate(df)
```

## Synthesizing data for testing

:::{warning}
The {ref}`data-synthesis-strategies` functionality is not yet supported in
the polars integration. At this time you can use the polars-native
[parametric testing](https://docs.pola.rs/py-polars/html/reference/testing.html#parametric-testing)
functions to generate test data for polars.
:::

## How it works

Compared to the way `pandera` handles `pandas` dataframes, `pandera`
attempts to leverage the `polars` [lazy API](https://docs.pola.rs/user-guide/lazy/using/)
as much as possible to leverage its query optimization benefits.

At a high level, this is what happens during schema validation:

- **Apply parsers**: add missing columns if `add_missing_columns=True`,
  coerce the datatypes if `coerce=True`, filter columns if `strict="filter"`,
  and set defaults if `default=<value>`.
- **Apply checks**: run all core, built-in, and custom checks on the data. Checks
  on metadata are done without `.collect()` operations, but checks that inspect
  data values do.
- **Raise an error**: if data errors are found, a {py:class}`~pandera.errors.SchemaError`
  is raised. If `validate(..., lazy=True)`, a {py:class}`~pandera.errors.SchemaErrors`
  exception is raised with all of the validation errors present in the data.
- **Return validated output**: if no data errors are found, the validated object
  is returned

:::{note}
Datatype coercion on `pl.LazyFrame` objects are done without `.collect()`
operations, but coercion on `pl.DataFrame` will, resulting in more
informative error messages since all failure cases can be reported.
:::

`pandera`'s validation behavior aligns with the way `polars` handles lazy
vs. eager operations. When you call `schema.validate()` on a `polars.LazyFrame`,
`pandera` will apply all of the parsers and checks that can be done without
any `collect()` operations. This means that it only does validations
at the schema-level, e.g. column names and data types.

However, if you validate a `polars.DataFrame`, `pandera` performs
schema-level and data-level validations.

:::{note}
Under the hood, `pandera` will convert `polars.DataFrame`s to a
`polars.LazyFrame`s before validating them. This is done to leverage the
polars lazy API during the validation process. While this feature isn't
fully optimized in the `pandera` library, this design decision lays the
ground-work for future performance improvements.
:::

### `LazyFrame` Method Chain

::::{tab-set}

:::{tab-item} DataFrameSchema
```{testcode} polars
import pandera.polars as pa
import polars as pl

schema = pa.DataFrameSchema({"a": pa.Column(int)})

df = (
    pl.LazyFrame({"a": [1.0, 2.0, 3.0]})
    .cast({"a": pl.Int64})
    .pipe(schema.validate) # this only validates schema-level properties
    .with_columns(b=pl.lit("a"))
    # do more lazy operations
    .collect()
)
print(df)
```

```{testoutput} polars
shape: (3, 2)
┌─────┬─────┐
│ a   ┆ b   │
│ --- ┆ --- │
│ i64 ┆ str │
╞═════╪═════╡
│ 1   ┆ a   │
│ 2   ┆ a   │
│ 3   ┆ a   │
└─────┴─────┘
```
:::

:::{tab-item} DataFrameModel

```{testcode} polars
import pandera.polars as pa
import polars as pl

class SimpleModel(pa.DataFrameModel):
    a: int

df = (
    pl.LazyFrame({"a": [1.0, 2.0, 3.0]})
    .cast({"a": pl.Int64})
    .pipe(SimpleModel.validate) # this only validates schema-level properties
    .with_columns(b=pl.lit("a"))
    # do more lazy operations
    .collect()
)
print(df)
```

```{testoutput} polars
shape: (3, 2)
┌─────┬─────┐
│ a   ┆ b   │
│ --- ┆ --- │
│ i64 ┆ str │
╞═════╪═════╡
│ 1   ┆ a   │
│ 2   ┆ a   │
│ 3   ┆ a   │
└─────┴─────┘
```
:::

::::

### `DataFrame` Method Chain

::::{tab-set}

:::{tab-item} DataFrameSchema
```{testcode} polars
schema = pa.DataFrameSchema({"a": pa.Column(int)})

df = (
    pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    .cast({"a": pl.Int64})
    .pipe(schema.validate) # this validates schema- and data- level properties
    .with_columns(b=pl.lit("a"))
    # do more eager operations
)
print(df)
```

```{testoutput} polars
shape: (3, 2)
┌─────┬─────┐
│ a   ┆ b   │
│ --- ┆ --- │
│ i64 ┆ str │
╞═════╪═════╡
│ 1   ┆ a   │
│ 2   ┆ a   │
│ 3   ┆ a   │
└─────┴─────┘
```
:::

:::{tab-item} DataFrameModel
```{testcode} polars
class SimpleModel(pa.DataFrameModel):
    a: int

df = (
    pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    .cast({"a": pl.Int64})
    .pipe(SimpleModel.validate) # this validates schema- and data- level properties
    .with_columns(b=pl.lit("a"))
    # do more eager operations
)
print(df)
```

```{testoutput} polars
shape: (3, 2)
┌─────┬─────┐
│ a   ┆ b   │
│ --- ┆ --- │
│ i64 ┆ str │
╞═════╪═════╡
│ 1   ┆ a   │
│ 2   ┆ a   │
│ 3   ┆ a   │
└─────┴─────┘
```
:::

::::


## Error Reporting

In the event of a validation error, `pandera` will raise a {py:class}`~pandera.errors.SchemaError`
eagerly.

```{code-cell} python
class SimpleModel(pa.DataFrameModel):
    a: int

invalid_lf = pl.LazyFrame({"a": pl.Series(["1", "2", "3"], dtype=pl.Utf8)})
try:
    SimpleModel.validate(invalid_lf)
except pa.errors.SchemaError as exc:
    print(exc)
```

And if you use lazy validation, `pandera` will raise a {py:class}`~pandera.errors.SchemaErrors`
exception. This is particularly useful when you want to collect all of the validation errors
present in the data.

:::{note}
{ref}`Lazy validation <lazy-validation>` in pandera is different from the
lazy API in polars, which is an unfortunate name collision. Lazy validation
means that all parsers and checks are applied to the data before raising
a {py:class}`~pandera.errors.SchemaErrors` exception. The lazy API
in polars allows you to build a computation graph without actually
executing it in-line, where you call `.collect()` to actually execute
the computation.
:::

::::{tab-set}

:::{tab-item} LazyFrame validation

By default, ``pl.LazyFrame`` validation will only validate schema-level properties:

```{testcode} polars
class ModelWithChecks(pa.DataFrameModel):
    a: int
    b: str = pa.Field(isin=[*"abc"])
    c: float = pa.Field(ge=0.0, le=1.0)

invalid_lf = pl.LazyFrame({
    "a": pl.Series(["1", "2", "3"], dtype=pl.Utf8),
    "b": ["d", "e", "f"],
    "c": [0.0, 1.1, -0.1],
})
ModelWithChecks.validate(invalid_lf, lazy=True)
```

```{testoutput} polars
Traceback (most recent call last):
...
pandera.errors.SchemaErrors: {
    "SCHEMA": {
        "WRONG_DATATYPE": [
            {
                "schema": "ModelWithChecks",
                "column": "a",
                "check": "dtype('Int64')",
                "error": "expected column 'a' to have type Int64, got String"
            }
        ]
    }
}
```
:::

:::{tab-item} DataFrame validation

By default, ``pl.DataFrame`` validation will validate both schema-level
and data-level properties:

```{testcode} polars
class ModelWithChecks(pa.DataFrameModel):
    a: int
    b: str = pa.Field(isin=[*"abc"])
    c: float = pa.Field(ge=0.0, le=1.0)

invalid_lf = pl.DataFrame({
    "a": pl.Series(["1", "2", "3"], dtype=pl.Utf8),
    "b": ["d", "e", "f"],
    "c": [0.0, 1.1, -0.1],
})
ModelWithChecks.validate(invalid_lf, lazy=True)
```

```{testoutput} polars
Traceback (most recent call last):
...
pandera.errors.SchemaErrors: {
    "SCHEMA": {
        "WRONG_DATATYPE": [
            {
                "schema": "ModelWithChecks",
                "column": "a",
                "check": "dtype('Int64')",
                "error": "expected column 'a' to have type Int64, got String"
            }
        ]
    },
    "DATA": {
        "DATAFRAME_CHECK": [
            {
                "schema": "ModelWithChecks",
                "column": "b",
                "check": "isin(['a', 'b', 'c'])",
                "error": "Column 'b' failed validator number 0: <Check isin: isin(['a', 'b', 'c'])> failure case examples: [{'b': 'd'}, {'b': 'e'}, {'b': 'f'}]"
            },
            {
                "schema": "ModelWithChecks",
                "column": "c",
                "check": "greater_than_or_equal_to(0.0)",
                "error": "Column 'c' failed validator number 0: <Check greater_than_or_equal_to: greater_than_or_equal_to(0.0)> failure case examples: [{'c': -0.1}]"
            },
            {
                "schema": "ModelWithChecks",
                "column": "c",
                "check": "less_than_or_equal_to(1.0)",
                "error": "Column 'c' failed validator number 1: <Check less_than_or_equal_to: less_than_or_equal_to(1.0)> failure case examples: [{'c': 1.1}]"
            }
        ]
    }
}
```
:::

::::

(supported-polars-dtypes)=

## Supported Data Types

`pandera` currently supports all of the
[polars data types](https://docs.pola.rs/py-polars/html/reference/datatypes.html).
Built-in python types like `str`, `int`, `float`, and `bool` will be
handled in the same way that `polars` handles them:

```{code-cell} python
assert pl.Series([1,2,3], dtype=int).dtype == pl.Int64
assert pl.Series([*"abc"], dtype=str).dtype == pl.Utf8
assert pl.Series([1.0, 2.0, 3.0], dtype=float).dtype == pl.Float64
```

So the following schemas are equivalent:

```{code-cell} python
schema1 = pa.DataFrameSchema({
    "a": pa.Column(int),
    "b": pa.Column(str),
    "c": pa.Column(float),
})

schema2 = pa.DataFrameSchema({
    "a": pa.Column(pl.Int64),
    "b": pa.Column(pl.Utf8),
    "c": pa.Column(pl.Float64),
})

assert schema1 == schema2
```

### Nested Types

Polars nested datetypes are also supported via {ref}`parameterized data types <parameterized-dtypes>`.
See the examples below for the different ways to specify this through the
object-based and class-based APIs:

::::{tab-set}

:::{tab-item} DataFrameSchema

```{testcode} polars
schema = pa.DataFrameSchema(
    {
        "list_col": pa.Column(pl.List(pl.Int64())),
        "array_col": pa.Column(pl.Array(pl.Int64(), 3)),
        "struct_col": pa.Column(pl.Struct({"a": pl.Utf8(), "b": pl.Float64()})),
    },
)
```
:::

:::{tab-item} DataFrameModel (Annotated)

```{testcode} polars
try:
    from typing import Annotated  # python 3.9+
except ImportError:
    from typing_extensions import Annotated

class ModelWithAnnotated(pa.DataFrameModel):
    list_col: Annotated[pl.List, pl.Int64()]
    array_col: Annotated[pl.Array, pl.Int64(), 3]
    struct_col: Annotated[pl.Struct, {"a": pl.Utf8(), "b": pl.Float64()}]
```
:::

:::{tab-item} DataFrameModel (Field)

```{testcode} polars
class ModelWithDtypeKwargs(pa.DataFrameModel):
    list_col: pl.List = pa.Field(dtype_kwargs={"inner": pl.Int64()})
    array_col: pl.Array = pa.Field(dtype_kwargs={"inner": pl.Int64(), "width": 3})
    struct_col: pl.Struct = pa.Field(dtype_kwargs={"fields": {"a": pl.Utf8(), "b": pl.Float64()}})
```
:::

::::

### Time-agnostic DateTime

In some use cases, it may not matter whether a column containing `pl.DateTime`
data has a timezone or not. In that case, you can use the pandera-native
polars datatype:

::::{tab-set}

:::{tab-item} DataFrameSchema

```{testcode} polars
from pandera.engines.polars_engine import DateTime


schema = pa.DataFrameSchema({
    "created_at": pa.Column(DateTime(time_zone_agnostic=True)),
})
```

:::

:::{tab-item} DataFrameModel (Annotated)

```{testcode} polars
from pandera.engines.polars_engine import DateTime


class DateTimeModel(pa.DataFrameModel):
    created_at: Annotated[DateTime, True, "us", None]
```
.
```{note}
For `Annotated` types, you need to pass in all positional and keyword arguments.
```

:::

:::{tab-item} DataFrameModel (Field)

```{testcode} polars
from pandera.engines.polars_engine import DateTime


class DateTimeModel(pa.DataFrameModel):
    created_at: DateTime = pa.Field(dtype_kwargs={"time_zone_agnostic": True})
```

:::

::::


## Custom checks

All of the built-in {py:class}`~pandera.api.checks.Check` methods are supported
in the polars integration.

To create custom checks, you can create functions that take a {py:class}`~pandera.api.polars.types.PolarsData`
named tuple as input and produces a `polars.LazyFrame` as output. {py:class}`~pandera.api.polars.types.PolarsData`
contains two attributes:

- A `lazyframe` attribute, which contains the `polars.LazyFrame` object you want
  to validate.
- A `key` attribute, which contains the column name you want to validate. This
  will be `None` for dataframe-level checks.

Element-wise checks are also supported by setting `element_wise=True`. This
will require a function that takes in a single element of the column/dataframe
and returns a boolean scalar indicating whether the value passed.

:::{warning}
Under the hood, element-wise checks use the
[map_elements](https://docs.pola.rs/py-polars/html/reference/expressions/api/polars.Expr.map_elements.html)
function, which is slower than the native polars expressions API.
:::

### Column-level Checks

Here's an example of a column-level custom check:

::::{tab-set}

:::{tab-item} DataFrameSchema

```{testcode} polars
from pandera.polars import PolarsData


def is_positive_vector(data: PolarsData) -> pl.LazyFrame:
    """Return a LazyFrame with a single boolean column."""
    return data.lazyframe.select(pl.col(data.key).gt(0))

def is_positive_scalar(data: PolarsData) -> pl.LazyFrame:
    """Return a LazyFrame with a single boolean scalar."""
    return data.lazyframe.select(pl.col(data.key).gt(0).all())

def is_positive_element_wise(x: int) -> bool:
    """Take a single value and return a boolean scalar."""
    return x > 0

schema_with_custom_checks = pa.DataFrameSchema({
    "a": pa.Column(
        int,
        checks=[
            pa.Check(is_positive_vector),
            pa.Check(is_positive_scalar),
            pa.Check(is_positive_element_wise, element_wise=True),
        ]
    )
})

lf = pl.LazyFrame({"a": [1, 2, 3]})
validated_df = lf.collect().pipe(schema_with_custom_checks.validate)
print(validated_df)
```

```{testoutput} polars
shape: (3, 1)
┌─────┐
│ a   │
│ --- │
│ i64 │
╞═════╡
│ 1   │
│ 2   │
│ 3   │
└─────┘
```
:::

:::{tab-item} DataFrameModel

```{testcode} polars
from pandera.polars import PolarsData


class ModelWithCustomChecks(pa.DataFrameModel):
    a: int

    @pa.check("a")
    def is_positive_vector(cls, data: PolarsData) -> pl.LazyFrame:
        """Return a LazyFrame with a single boolean column."""
        return data.lazyframe.select(pl.col(data.key).gt(0))

    @pa.check("a")
    def is_positive_scalar(cls, data: PolarsData) -> pl.LazyFrame:
        """Return a LazyFrame with a single boolean scalar."""
        return data.lazyframe.select(pl.col(data.key).gt(0).all())

    @pa.check("a", element_wise=True)
    def is_positive_element_wise(cls, x: int) -> bool:
        """Take a single value and return a boolean scalar."""
        return x > 0

validated_df = lf.collect().pipe(ModelWithCustomChecks.validate)
print(validated_df)
```

```{testoutput} polars
shape: (3, 1)
┌─────┐
│ a   │
│ --- │
│ i64 │
╞═════╡
│ 1   │
│ 2   │
│ 3   │
└─────┘
```
:::

::::


For column-level checks, the custom check function should return a
`polars.LazyFrame` containing a single boolean column or a single boolean scalar.

### DataFrame-level Checks

If you need to validate values on an entire dataframe, you can specify a check
at the dataframe level. The expected output is a `polars.LazyFrame` containing
multiple boolean columns, a single boolean column, or a scalar boolean.

::::{tab-set}

:::{tab-item} DataFrameSchema

```{testcode} polars
def col1_gt_col2(data: PolarsData, col1: str, col2: str) -> pl.LazyFrame:
    """Return a LazyFrame with a single boolean column."""
    return data.lazyframe.select(pl.col(col1).gt(pl.col(col2)))

def is_positive_df(data: PolarsData) -> pl.LazyFrame:
    """Return a LazyFrame with multiple boolean columns."""
    return data.lazyframe.select(pl.col("*").gt(0))

def is_positive_element_wise(x: int) -> bool:
    """Take a single value and return a boolean scalar."""
    return x > 0

schema_with_df_checks = pa.DataFrameSchema(
    columns={
        "a": pa.Column(int),
        "b": pa.Column(int),
    },
    checks=[
        pa.Check(col1_gt_col2, col1="a", col2="b"),
        pa.Check(is_positive_df),
        pa.Check(is_positive_element_wise, element_wise=True),
    ]
)

lf = pl.LazyFrame({"a": [2, 3, 4], "b": [1, 2, 3]})
validated_df = lf.collect().pipe(schema_with_df_checks.validate)
print(validated_df)
```

```{testoutput} polars
shape: (3, 2)
┌─────┬─────┐
│ a   ┆ b   │
│ --- ┆ --- │
│ i64 ┆ i64 │
╞═════╪═════╡
│ 2   ┆ 1   │
│ 3   ┆ 2   │
│ 4   ┆ 3   │
└─────┴─────┘
```
:::

:::{tab-item} DataFrameModel

```{testcode} polars
class ModelWithDFChecks(pa.DataFrameModel):
    a: int
    b: int

    @pa.dataframe_check
    def cola_gt_colb(cls, data: PolarsData) -> pl.LazyFrame:
        """Return a LazyFrame with a single boolean column."""
        return data.lazyframe.select(pl.col("a").gt(pl.col("b")))

    @pa.dataframe_check
    def is_positive_df(cls, data: PolarsData) -> pl.LazyFrame:
        """Return a LazyFrame with multiple boolean columns."""
        return data.lazyframe.select(pl.col("*").gt(0))

    @pa.dataframe_check(element_wise=True)
    def is_positive_element_wise(cls, x: int) -> bool:
        """Take a single value and return a boolean scalar."""
        return x > 0

validated_df = lf.collect().pipe(ModelWithDFChecks.validate)
print(validated_df)
```

```{testoutput} polars
shape: (3, 2)
┌─────┬─────┐
│ a   ┆ b   │
│ --- ┆ --- │
│ i64 ┆ i64 │
╞═════╪═════╡
│ 2   ┆ 1   │
│ 3   ┆ 2   │
│ 4   ┆ 3   │
└─────┴─────┘
```
:::

::::


## Data-level Validation with LazyFrames

As mentioned earlier in this page, by default calling `schema.validate` on
a `pl.LazyFrame` will only perform schema-level validation checks. If you want
to validate data-level properties on a `pl.LazyFrame`, the recommended way
would be to first call `.collect()`:

```{code-cell} python
class SimpleModel(pa.DataFrameModel):
        a: int

lf: pl.LazyFrame = (
    pl.LazyFrame({"a": [1.0, 2.0, 3.0]})
    .cast({"a": pl.Int64})
    .collect()  # convert to pl.DataFrame
    .pipe(SimpleModel.validate)
    .lazy()     # convert back to pl.LazyFrame
    # do more lazy operations
)
```

This syntax is nice because it's clear what's happening just from reading the
code. Pandera schemas serve as a clear point in the method chain where the data
is materialized.

However, if you don't mind a little magic 🪄, you can set the
`PANDERA_VALIDATION_DEPTH` environment variable to `SCHEMA_AND_DATA` to
validate data-level properties on a `polars.LazyFrame`. This will be equivalent
to the explicit code above:

```bash
export PANDERA_VALIDATION_DEPTH=SCHEMA_AND_DATA
```

```{code-cell} python
lf: pl.LazyFrame = (
    pl.LazyFrame({"a": [1.0, 2.0, 3.0]})
    .cast({"a": pl.Int64})
    .pipe(SimpleModel.validate)  # this will validate schema- and data-level properties
    # do more lazy operations
)
```

Under the hood, the validation process will make `.collect()` calls on the
LazyFrame in order to run data-level validation checks, and it will still
return a `pl.LazyFrame` after validation is done.

## Supported and Unsupported Functionality

Since the pandera-polars integration is less mature than pandas support, some
of the functionality offered by the pandera with pandas DataFrames are
not yet supported with polars DataFrames.

Here is a list of supported and unsupported features. You can
refer to the {ref}`supported features matrix <supported-features>` to see
which features are implemented in the polars validation backend.
