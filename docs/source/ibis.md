---
file_format: mystnb
---

```{currentmodule} pandera.ibis
```

(ibis)=

# Data Validation with Ibis

*new in 0.25.0*

[Ibis](https://ibis-project.org/) is an open-source dataframe library that
works with any data system. You can use the same API for 20 backends, from
fast local engines like DuckDB, Polars, and DataFusion to distributed data
systems like BigQuery, Snowflake, and Databricks.

## Usage

With the Ibis integration, you can define Pandera schemas to validate Ibis
tables in Python. First, install `pandera` with the `ibis` extra alongside
the Ibis backend that you're using:

```bash
pip install 'pandera[ibis]' 'ibis-framework[duckdb]'
```

:::{note}
You can find the command to install the Ibis backend of your choice on the
[Installation page of the Ibis documentation](https://ibis-project.org/install).
:::

Then, you can start validating Ibis tables using Pandera schemas. In the example
below, we'll use the {ref}`class-based API <dataframe-models>` to define a
{py:class}`~pandera.api.ibis.model.DataFrameModel`, which we'll then use to
validate an {py:class}`ibis.Table` object.

```{code-cell} python
import ibis
import pandera.ibis as pa


class Schema(pa.DataFrameModel):
    state: str
    city: str
    price: int = pa.Field(in_range={"min_value": 5, "max_value": 20})


t = ibis.memtable(
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
Schema.validate(t).execute()
```

You can also use the {py:func}`~pandera.decorators.check_types` decorator to
validate Ibis table function annotations at runtime:

```{code-cell} python
from pandera.typing.ibis import Table


@pa.check_types
def function(t: Table[Schema]) -> Table[Schema]:
    return t.filter(t.state == "CA")


function(t).execute()
```

And of course, you can use the object-based API to define a
{py:class}`~pandera.api.ibis.container.DataFrameSchema`:

```{code-cell} python
schema = pa.DataFrameSchema({
    "state": pa.Column(str),
    "city": pa.Column(str),
    "price": pa.Column(int, pa.Check.in_range(min_value=5, max_value=20))
})
schema.validate(t).execute()
```

## Synthesizing data for testing

:::{warning}
The {ref}`data-synthesis-strategies` functionality is not yet supported in
the Ibis integration. At this time, you can use other frameworks to generate
test data for Ibis. For example, you can use the polars-native
[parametric testing](https://docs.pola.rs/py-polars/html/reference/testing.html#parametric-testing)
functions to producing Polars DataFrames or LazyFrames, from which you can
construct {py:class}`ibis.memtable`s.
:::

## How it works

Compared to the way `pandera` handles `pandas` dataframes, the Ibis backend for
`pandera` leverages the fact that [Ibis tables are lazy](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table).

At a high level, this is what happens during schema validation:

- **Apply parsers**: add missing columns if `add_missing_columns=True`,
  coerce the datatypes if `coerce=True`, filter columns if `strict="filter"`,
  and set defaults if `default=<value>`.
- **Apply checks**: run all core, built-in, and custom checks on the data. Checks
  on metadata are done without `.execute()` operations, but checks that inspect
  data values do.
- **Raise an error**: if data errors are found, a {py:class}`~pandera.errors.SchemaError`
  is raised. If `validate(..., lazy=True)`, a {py:class}`~pandera.errors.SchemaErrors`
  exception is raised with all of the validation errors present in the data.
- **Return validated output**: if no data errors are found, the validated object
  is returned.

`pandera`'s validation behavior aligns with the way `ibis` handles lazy
vs. eager operations. When you call `schema.validate()` on an Ibis table,
`pandera` will apply all of the parsers and checks that can be done without
any `execute()` operations. This means that it only does validations
at the schema-level, e.g. column names and data types.

### Method Chaining

#### Using `DataFrameSchema`

```{code-cell} python
import ibis
import pandera.ibis as pa

schema = pa.DataFrameSchema({"a": pa.Column(int)})

df = (
    ibis.memtable({"a": [1.0, 2.0, 3.0]})
    .cast({"a": "int64"})
    .pipe(schema.validate) # this validates schema- and data-level properties
    .mutate(b=ibis.literal("a"))
    # do more lazy operations
    .execute()
)
print(df)
```

#### Using `DataFrameModel`

```{code-cell} python
import ibis
import pandera.ibis as pa


class SimpleModel(pa.DataFrameModel):
    a: int


df = (
    ibis.memtable({"a": [1.0, 2.0, 3.0]})
    .cast({"a": "int64"})
    .pipe(SimpleModel.validate) # this validates schema- and data-level properties
    .mutate(b=ibis.literal("a"))
    # do more lazy operations
    .execute()
)
print(df)
```

## Error Reporting

In the event of a validation error, `pandera` will raise a {py:class}`~pandera.errors.SchemaError`
eagerly.

```{code-cell} python
class SimpleModel(pa.DataFrameModel):
    a: int

invalid_t = ibis.memtable({"a": ["1", "2", "3"]})
try:
    SimpleModel.validate(invalid_t)
except pa.errors.SchemaError as exc:
    print(exc)
```

And if you use lazy validation, `pandera` will raise a {py:class}`~pandera.errors.SchemaErrors`
exception. This is particularly useful when you want to collect all of the validation errors
present in the data.

By default, Pandera will validate both schema- and data-level properties:

```{code-cell} python
:tags: ["raises-exception"]
class ModelWithChecks(pa.DataFrameModel):
    a: int
    b: str = pa.Field(isin=[*"abc"])
    c: float = pa.Field(ge=0.0, le=1.0)

invalid_t = ibis.memtable({
    "a": ["1", "2", "3"],
    "b": ["d", "e", "f"],
    "c": [0.0, 1.1, -0.1],
})
ModelWithChecks.validate(invalid_t, lazy=True)
```

(supported-ibis-dtypes)=

## Supported Data Types

`pandera` currently supports most of the
[Ibis data types](https://ibis-project.org/reference/datatypes).
Built-in Python types like `str`, `int`, `float`, and `bool` will be
handled in the same way that Ibis handles them:

```{code-cell} python
schema1 = ibis.schema({"x": int, "y": str, "z": float})
schema2 = ibis.schema({"x": "int64", "y": "string", "z": "float64"})
assert schema1 == schema2
```

So the following schemas are equivalent:

```{code-cell} python
schema1 = pa.DataFrameSchema({
    "a": pa.Column(int),
    "b": pa.Column(str),
    "c": pa.Column(float),
})

schema2 = pa.DataFrameSchema({
    "a": pa.Column(ibis.dtype("int64")),
    "b": pa.Column(ibis.dtype("string")),
    "c": pa.Column(ibis.dtype("float64")),
})

assert schema1 == schema2
```

### Nested Types

:::{warning}
Using {ref}`parameterized data types <parameterized-dtypes>` for nested Ibis
data types is not yet supported in the Ibis integration.
:::

### Time-agnostic DateTime

In some use cases, it may not matter whether a column containing timestamp
data has a timezone or not.

:::{warning}
The ``time_zone_agnostic`` argument for the timestamp data type is not yet
supported in the Ibis integration.
:::


## Custom checks

All of the built-in {py:class}`~pandera.api.checks.Check` methods are supported
in the Ibis integration.

To create custom checks, you can create functions that take a {py:class}`~pandera.api.ibis.types.IbisData`
named tuple as input and produces a ``ibis.Table`` as output. {py:class}`~pandera.api.ibis.types.IbisData`
contains two attributes:

- A `table` attribute, which contains the `ibis.Table` object you want
  to validate.
- A `key` attribute, which contains the column name you want to validate. This
  will be `None` for table-level checks.

Element-wise checks are also supported by setting `element_wise=True`. This
will require a function that takes in a single element of the column/dataframe
and returns a boolean scalar indicating whether the value passed.

:::{warning}
Under the hood, element-wise checks use
[Python UDFs](https://ibis-project.org/reference/scalar-udfs#ibis.expr.operations.udf.scalar.python),
which are likely to be **much** slower than vectorized checks.
:::

### Column-level Checks

For column-level checks, the custom check function should return an Ibis
table containing a single boolean column or a single boolean scalar.

Here's an example of a column-level custom check:

#### Using `DataFrameSchema`

```{code-cell} python
from pandera.ibis import IbisData


def is_positive_vector(data: IbisData) -> ibis.Table:
    """Return a table with a single boolean column."""
    return data.table.select(data.table[data.key] > 0)

def is_positive_scalar(data: IbisData) -> ibis.Table:
    """Return a table with a single boolean scalar."""
    return data.table[data.key] > 0

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

t = ibis.memtable({"a": [1, 2, 3]})
validated_t = t.pipe(schema_with_custom_checks.validate)
print(validated_t)
```

#### Using `DataFrameModel`

```{code-cell} python
from pandera.ibis import IbisData


class ModelWithCustomChecks(pa.DataFrameModel):
    a: int

    @pa.check("a")
    def is_positive_vector(cls, data: IbisData) -> ibis.Table:
        """Return a table with a single boolean column."""
        return data.table.select(data.table[data.key] > 0)

    @pa.check("a")
    def is_positive_scalar(cls, data: IbisData) -> ibis.Table:
        """Return a table with a single boolean scalar."""
        return data.table[data.key] > 0


t = ibis.memtable({"a": [1, 2, 3]})
validated_t = t.pipe(ModelWithCustomChecks.validate)
print(validated_t)
```

:::{warning}
Element-wise checks using ``DataFrameModel`` are not yet supported in
the Ibis integration; use ``DataFrameSchema`` instead.
:::

### DataFrame-level Checks

If you need to validate values on an entire dataframe, you can specify a check
at the dataframe level. The expected output is an Ibis table containing
multiple boolean columns, a single boolean column, or a scalar boolean.

#### Using `DataFrameSchema`

```{code-cell} python
from ibis import _, selectors as s


def col1_gt_col2(data: IbisData, col1: str, col2: str) -> ibis.Table:
    """Return a table with a single boolean column."""
    return data.table.select(data.table[col1] > data.table[col2])

def is_positive_df(data: IbisData) -> ibis.Table:
    """Return a table with multiple boolean columns."""
    return data.table.select(s.across(s.all(), _ > 0))

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

t = ibis.memtable({"a": [2, 3, 4], "b": [1, 2, 3]})
validated_t = t.pipe(schema_with_df_checks.validate)
print(validated_t)
```

#### Using `DataFrameModel`

```{code-cell} python
class ModelWithDFChecks(pa.DataFrameModel):
    a: int
    b: int

    @pa.dataframe_check
    def cola_gt_colb(cls, data: IbisData) -> ibis.Table:
        """Return a table with a single boolean column."""
        return data.table.select(data.table["a"] > data.table["b"])

    @pa.dataframe_check
    def is_positive_df(cls, data: IbisData) -> ibis.Table:
        """Return a table with multiple boolean columns."""
        return data.table.select(s.across(s.all(), _ > 0))


t = ibis.memtable({"a": [2, 3, 4], "b": [1, 2, 3]})
validated_t = t.pipe(ModelWithDFChecks.validate)
print(validated_t)
```

:::{warning}
Element-wise checks using ``DataFrameModel`` are not yet supported in
the Ibis integration; use ``DataFrameSchema`` instead.
:::

## Supported and Unsupported Functionality

Since the Pandera-Ibis integration is less mature than pandas support, some
of the functionality offered by the pandera with pandas DataFrames are
not yet supported with Ibis tables.

Here is a list of supported and unsupported features. You can
refer to the {ref}`supported features matrix <supported-features>` to see
which features are implemented in the Ibis validation backend.
