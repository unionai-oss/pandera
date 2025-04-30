---
file_format: mystnb
---

% pandera documentation for DataFrameSchemas

```{currentmodule} pandera
```

(dataframeschemas)=

# DataFrame Schemas

The {class}`~pandera.api.pandas.container.DataFrameSchema` class enables the specification of a schema
that verifies the columns and index of a pandas `DataFrame` object.

The {class}`~pandera.api.pandas.container.DataFrameSchema` object consists of `Column`s and an `Index` (if applicable).

```{code-cell} python
import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {
        "column1": pa.Column(int),
        "column2": pa.Column(float, pa.Check(lambda s: s < -1.2)),
        # you can provide a list of validators
        "column3": pa.Column(str, [
            pa.Check(lambda s: s.str.startswith("value")),
            pa.Check(lambda s: s.str.split("_", expand=True).shape[1] == 2)
        ]),
    },
    index=pa.Index(int),
    strict=True,
    coerce=True,
)
```

You can refer to {ref}`dataframe-models` to see how to define dataframe schemas
using the alternative pydantic/dataclass-style syntax.

(column)=

## Column Validation

A {class}`~pandera.api.pandas.components.Column` must specify the properties of a
column in a dataframe object. It can be optionally verified for its data type,
[null values] or
duplicate values. The column can be `coerce`d into the specified type, and the
[required] parameter allows control over whether or not the column is allowed to
be missing.

Similarly to pandas, the data type can be specified as:

- a string alias, as long as it is recognized by pandas.
- a python type: `int`, `float`, `double`, `bool`, `str`
- a [numpy data type](https://numpy.org/doc/stable/user/basics.types.html)
- a [pandas extension type](https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#dtypes):
  it can be an instance (e.g `pd.CategoricalDtype(["a", "b"])`) or a
  class (e.g `pandas.CategoricalDtype`) if it can be initialized with default
  values.
- a pandera {class}`~pandera.dtypes.DataType`: it can also be an instance or a
  class.

:::{important}
You can learn more about how data type validation works
{ref}`dtype-validation`.
:::

{ref}`Column checks<checks>` allow for the DataFrame's values to be
checked against a user-provided function. `Check` objects also support
{ref}`grouping<grouping>` by a different column so that the user can make
assertions about subsets of the column of interest.

Column Hypotheses enable you to perform statistical hypothesis tests on a
DataFrame in either wide or tidy format. See
{ref}`Hypothesis Testing<hypothesis>` for more details.

(null-values)=

### Null Values in Columns

By default, SeriesSchema/Column objects assume that values are not
nullable. In order to accept null values, you need to explicitly specify
`nullable=True`, or else you’ll get an error.

```{code-cell} python
import numpy as np
import pandas as pd
import pandera.pandas as pa

df = pd.DataFrame({"column1": [5, 1, np.nan]})

non_null_schema = pa.DataFrameSchema({
    "column1": pa.Column(float, pa.Check(lambda x: x > 0))
})

try:
    non_null_schema.validate(df)
except pa.errors.SchemaError as exc:
    print(exc)
```

Setting `nullable=True` allows for null values in the corresponding column.

```{code-cell} python
null_schema = pa.DataFrameSchema({
    "column1": pa.Column(float, pa.Check(lambda x: x > 0), nullable=True)
})

null_schema.validate(df)
```

To learn more about how the nullable check interacts with data type checks,
see {ref}`here <how-nullable-works>`.

(coerced)=

### Coercing Types on Columns

If you specify `Column(dtype, ..., coerce=True)` as part of the
DataFrameSchema definition, calling `schema.validate` will first
coerce the column into the specified `dtype` before applying validation
checks.

```{code-cell} python
import pandas as pd
import pandera.pandas as pa

df = pd.DataFrame({"column1": [1, 2, 3]})
schema = pa.DataFrameSchema({"column1": pa.Column(str, coerce=True)})

validated_df = schema.validate(df)
assert isinstance(validated_df.column1.iloc[0], str)
```

:::{note}
Note the special case of integers columns not supporting `nan`
values. In this case, `schema.validate` will complain if `coerce == True`
and null values are allowed in the column.
:::

```{code-cell} python
df = pd.DataFrame({"column1": [1., 2., 3, np.nan]})
schema = pa.DataFrameSchema({
    "column1": pa.Column(int, coerce=True, nullable=True)
})

try:
    schema.validate(df)
except pa.errors.SchemaError as exc:
    print(exc)
```

The best way to handle this case is to simply specify the column as a
`Float` or `Object`.

```{code-cell} python
schema_object = pa.DataFrameSchema({
    "column1": pa.Column(object, coerce=True, nullable=True)
})
schema_float = pa.DataFrameSchema({
    "column1": pa.Column(float, coerce=True, nullable=True)
})

print(schema_object.validate(df).dtypes)
print(schema_float.validate(df).dtypes)
```

If you want to coerce all of the columns specified in the
`DataFrameSchema`, you can specify the `coerce` argument with
`DataFrameSchema(..., coerce=True)`. Note that this will have
the effect of overriding any `coerce=False` arguments specified at
the `Column` or `Index` level.

(required)=

### Required Columns

By default all columns specified in the schema are required, meaning
that if a column is missing in the input DataFrame an exception will be
thrown. If you want to make a column optional, specify `required=False`
in the column constructor:

```{code-cell} python
import pandas as pd
import pandera.pandas as pa


df = pd.DataFrame({"column2": ["hello", "pandera"]})
schema = pa.DataFrameSchema({
    "column1": pa.Column(int, required=False),
    "column2": pa.Column(str)
})

schema.validate(df)
```

Since `required=True` by default, missing columns would raise an error:

```{code-cell} python
schema = pa.DataFrameSchema({
    "column1": pa.Column(int),
    "column2": pa.Column(str),
})

try:
    schema.validate(df)
except pa.errors.SchemaError as exc:
    print(exc)
```

(column-validation-1)=

### Stand-alone Column Validation

In addition to being used in the context of a `DataFrameSchema`, `Column`
objects can also be used to validate columns in a dataframe on its own:

```{code-cell} python
import pandas as pd
import pandera.pandas as pa

df = pd.DataFrame({
    "column1": [1, 2, 3],
    "column2": ["a", "b", "c"],
})

column1_schema = pa.Column(int, name="column1")
column2_schema = pa.Column(str, name="column2")

# pass the dataframe as an argument to the Column object callable
df = column1_schema(df)
validated_df = column2_schema(df)

# or explicitly use the validate method
df = column1_schema.validate(df)
validated_df = column2_schema.validate(df)

# use the DataFrame.pipe method to validate two columns
df.pipe(column1_schema).pipe(column2_schema)
```

For multi-column use cases, the {class}`~pandera.api.pandas.container.DataFrameSchema`
is still recommended, but if you have one or a small number of columns to verify,
using `Column` objects by themselves is appropriate.

(column-name-regex)=

### Column Regex Pattern Matching

In the case that your dataframe has multiple columns that share common
statistical properties, you might want to specify a regex pattern that matches
a set of meaningfully grouped columns that have `str` names.

```{code-cell} python
import numpy as np
import pandas as pd
import pandera.pandas as pa

categories = ["A", "B", "C"]

np.random.seed(100)

dataframe = pd.DataFrame({
    "cat_var_1": np.random.choice(categories, size=100),
    "cat_var_2": np.random.choice(categories, size=100),
    "num_var_1": np.random.uniform(0, 10, size=100),
    "num_var_2": np.random.uniform(20, 30, size=100),
})

schema = pa.DataFrameSchema({
    "num_var_.+": pa.Column(
        float,
        checks=pa.Check.greater_than_or_equal_to(0),
        regex=True,
    ),
    "cat_var_.+": pa.Column(
        pa.Category,
        checks=pa.Check.isin(categories),
        coerce=True,
        regex=True,
    ),
})

schema.validate(dataframe).head()
```

You can also regex pattern match on `pd.MultiIndex` columns:

```{code-cell} python
np.random.seed(100)

dataframe = pd.DataFrame({
    ("cat_var_1", "y1"): np.random.choice(categories, size=100),
    ("cat_var_2", "y2"): np.random.choice(categories, size=100),
    ("num_var_1", "x1"): np.random.uniform(0, 10, size=100),
    ("num_var_2", "x2"): np.random.uniform(0, 10, size=100),
})

schema = pa.DataFrameSchema({
    ("num_var_.+", "x.+"): pa.Column(
        float,
        checks=pa.Check.greater_than_or_equal_to(0),
        regex=True,
    ),
    ("cat_var_.+", "y.+"): pa.Column(
        pa.Category,
        checks=pa.Check.isin(categories),
        coerce=True,
        regex=True,
    ),
})

schema.validate(dataframe).head()
```

(strict)=

### Handling Dataframe Columns not in the Schema

By default, columns that aren’t specified in the schema aren’t checked.
If you want to check that the DataFrame *only* contains columns in the
schema, specify `strict=True`:

```{code-cell} python
import pandas as pd
import pandera.pandas as pa


schema = pa.DataFrameSchema(
    {"column1": pa.Column(int)},
    strict=True)

df = pd.DataFrame({"column2": [1, 2, 3]})

try:
    schema.validate(df)
except pa.errors.SchemaError as exc:
    print(exc)
```

Alternatively, if your DataFrame contains columns that are not in the schema,
and you would like these to be dropped on validation,
you can specify `strict='filter'`.

```{code-cell} python
import pandas as pd
import pandera.pandas as pa


df = pd.DataFrame({"column1": ["drop", "me"],"column2": ["keep", "me"]})
schema = pa.DataFrameSchema({"column2": pa.Column(str)}, strict='filter')

schema.validate(df)
```

(ordered)=

### Validating the order of the columns

For some applications the order of the columns is important. For example:

- If you want to use
  [selection by position](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html#selection-by-position)
  instead of the more common
  [selection by label](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html#selection-by-label).
- Machine learning: Many ML libraries will cast a Dataframe to numpy arrays,
  for which order becomes crucial.

To validate the order of the Dataframe columns, specify `ordered=True`:

```{code-cell} python
import pandas as pd
import pandera.pandas as pa

schema = pa.DataFrameSchema(
    columns={"a": pa.Column(int), "b": pa.Column(int)}, ordered=True
)
df = pd.DataFrame({"b": [1], "a": [1]})

try:
    schema.validate(df)
except pa.errors.SchemaError as exc:
    print(exc)
```

(index)=

### Validating the joint uniqueness of columns

In some cases you might want to ensure that a group of columns are unique:

```{code-cell} python
import pandas as pd
import pandera.pandas as pa

schema = pa.DataFrameSchema(
    columns={col: pa.Column(int) for col in ["a", "b", "c"]},
    unique=["a", "c"],
)
df = pd.DataFrame.from_records([
    {"a": 1, "b": 2, "c": 3},
    {"a": 1, "b": 2, "c": 3},
])
try:
    schema.validate(df)
except pa.errors.SchemaError as exc:
    print(exc)
```

To control how unique errors are reported, the `report_duplicates` argument accepts:
: - `exclude_first`: (default) report all duplicates except first occurrence
  - `exclude_last`: report all duplicates except last occurrence
  - `all`: report all duplicates

```{code-cell} python
import pandas as pd
import pandera.pandas as pa

schema = pa.DataFrameSchema(
    columns={col: pa.Column(int) for col in ["a", "b", "c"]},
    unique=["a", "c"],
    report_duplicates = "exclude_first",
)
df = pd.DataFrame.from_records([
    {"a": 1, "b": 2, "c": 3},
    {"a": 1, "b": 2, "c": 3},
])

try:
    schema.validate(df)
except pa.errors.SchemaError as exc:
    print(exc)
```

(adding-missing-columns)=

### Adding missing columns

When loading raw data into a form that's ready for data processing, it's often
useful to have guarantees that the columns specified in the schema are present,
even if they're missing from the raw data. This is where it's useful to
specify `add_missing_columns=True` in your schema definition.

When you call `schema.validate(data)`, the schema will add any missing columns
to the dataframe, defaulting to the `default` value if supplied at the column-level,
or to `NaN` if the column is nullable.

```{code-cell} python
import pandas as pd
import pandera.pandas as pa

schema = pa.DataFrameSchema(
    columns={
        "a": pa.Column(int),
        "b": pa.Column(int, default=1),
        "c": pa.Column(float, nullable=True),
    },
    add_missing_columns=True,
    coerce=True,
)
df = pd.DataFrame({"a": [1, 2, 3]})
schema.validate(df)
```

(index-validation)=

## Index Validation

You can also specify an {class}`~pandera.api.pandas.components.Index` in the {class}`~pandera.api.pandas.container.DataFrameSchema`.

```{code-cell} python
import pandas as pd
import pandera.pandas as pa


schema = pa.DataFrameSchema(
    columns={"a": pa.Column(int)},
    index=pa.Index(
        str,
        pa.Check(lambda x: x.str.startswith("index_"))))

df = pd.DataFrame(
    data={"a": [1, 2, 3]},
    index=["index_1", "index_2", "index_3"])

schema.validate(df)
```

In the case that the DataFrame index doesn't pass the `Check`.

```{code-cell} python
df = pd.DataFrame(
    data={"a": [1, 2, 3]},
    index=["foo1", "foo2", "foo3"]
)

try:
    schema.validate(df)
except pa.errors.SchemaError as exc:
    print(exc)
```

(multiindex-validation)=

## MultiIndex Validation

`pandera` also supports multi-index column and index validation.

### MultiIndex Columns

Specifying multi-index columns follows the `pandas` syntax of specifying
tuples for each level in the index hierarchy:

```{code-cell} python
import pandas as pd
import pandera.pandas as pa


schema = pa.DataFrameSchema({
    ("foo", "bar"): pa.Column(int),
    ("foo", "baz"): pa.Column(str)
})

df = pd.DataFrame({
    ("foo", "bar"): [1, 2, 3],
    ("foo", "baz"): ["a", "b", "c"],
})

schema.validate(df)
```

(multiindex)=

### MultiIndex Indexes

The {class}`~pandera.api.pandas.components.MultiIndex` class allows you to define multi-index
indexes by composing a list of `pandera.Index` objects.

```{code-cell} python
import pandas as pd
import pandera.pandas as pa

schema = pa.DataFrameSchema(
    columns={"column1": pa.Column(int)},
    index=pa.MultiIndex([
        pa.Index(str,
            pa.Check(lambda s: s.isin(["foo", "bar"])),
            name="index0"),
        pa.Index(int, name="index1"),
    ])
)

df = pd.DataFrame(
    data={"column1": [1, 2, 3]},
    index=pd.MultiIndex.from_arrays(
        [["foo", "bar", "foo"], [0, 1,2 ]],
        names=["index0", "index1"]
    )
)

schema.validate(df)
```

## Get Pandas Data Types

Pandas provides a `dtype` parameter for casting a dataframe to a specific dtype
schema. {class}`~pandera.api.pandas.container.DataFrameSchema` provides
a {attr}`~pandera.api.pandas.container.DataFrameSchema.dtypes` property which returns a
dictionary whose keys are column names and values are {class}`~pandera.dtypes.DataType`.

Some examples of where this can be provided to pandas are:

- <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html>
- <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html>

```{code-cell} python
import pandas as pd
import pandera.pandas as pa

schema = pa.DataFrameSchema(
    columns={
    "column1": pa.Column(int),
    "column2": pa.Column(pa.Category),
    "column3": pa.Column(bool)
    },
)

df = (
    pd.DataFrame.from_dict(
        {
            "a": {"column1": 1, "column2": "valueA", "column3": True},
            "b": {"column1": 1, "column2": "valueB", "column3": True},
        },
        orient="index",
    )
    .astype({col: str(dtype) for col, dtype in schema.dtypes.items()})
    .sort_index(axis=1)
)

schema.validate(df)
```

(dataframe-schema-transformations)=

## DataFrameSchema Transformations

Once you've defined a schema, you can then make modifications to it, both on
the schema level -- such as adding or removing columns and setting or resetting
the index -- or on the column or index level -- such as changing the data type
or checks.

This is useful for reusing schema objects in a data pipeline when additional
computation has been done on a dataframe, where the column or index objects
may have changed or perhaps where additional checks may be required.

```{code-cell} python
import pandas as pd
import pandera.pandas as pa

data = pd.DataFrame({"col1": range(1, 6)})

schema = pa.DataFrameSchema(
    columns={"col1": pa.Column(int, pa.Check(lambda s: s >= 0))},
    strict=True)

transformed_schema = schema.add_columns({
    "col2": pa.Column(str, pa.Check(lambda s: s == "value")),
    "col3": pa.Column(float, pa.Check(lambda x: x == 0.0)),
})

# validate original data
data = schema.validate(data)

# transformation
transformed_data = data.assign(col2="value", col3=0.0)

# validate transformed data
transformed_schema.validate(transformed_data)
```

Similarly, if you want dropped columns to be explicitly validated in a
data pipeline:

```{code-cell} python
import pandera.pandas as pa

schema = pa.DataFrameSchema(
    columns={
        "col1": pa.Column(int, pa.Check(lambda s: s >= 0)),
        "col2": pa.Column(str, pa.Check(lambda x: x <= 0)),
        "col3": pa.Column(object, pa.Check(lambda x: x == 0)),
    },
    strict=True,
)

schema.remove_columns(["col2", "col3"])
```

If during the course of a data pipeline one of your columns is moved into the
index, you can simply update the initial input schema using the
{func}`~pandera.api.dataframe.container.DataFrameSchema.set_index` method to create a schema for
the pipeline output.

```{code-cell} python
import pandera.pandas as pa


schema = pa.DataFrameSchema(
    {
        "column1": pa.Column(int),
        "column2": pa.Column(float)
    },
    index=pa.Index(int, name = "column3"),
    strict=True,
    coerce=True,
)
schema.set_index(["column1"], append = True)
```

And if you want to update the checks on a column or an index, you can use the
{func}`~pandera.api.dataframe.container.DataFrameSchema.update_column` or
{func}`~pandera.api.dataframe.container.DataFrameSchema.update_index` method.

```{code-cell} python
import pandas as pd
import pandera.pandas as pa
schema = pa.DataFrameSchema(
    {
        "column1": pa.Column(int),
        "column2": pa.Column(float)
    },
    index=pa.Index(int, name = "column3"),
    strict=True,
    coerce=True,
)
df = pd.DataFrame(
    data={"column1": [1, 2, 3], "column2": [1.0, 2.0, 3.0]},
    index=pd.Index([0, 1, 2], name="column3")
)
schema.validate(df)

schema = (
  schema
  .update_index(
    "column3", checks=pa.Check(lambda s: s.isin([0, 1, 2])),
  ).update_column(
    "column1", checks=pa.Check(lambda s: s > 0)
  )
)
df = pd.DataFrame(
    data={"column1": [1, 2, 3], "column2": [1.0, 2.0, 3.0]},
    index=pd.Index([0, 1, 2], name="column3")
)

schema.validate(df)
```


The available methods for altering the schema are:

- {func}`~pandera.api.dataframe.container.DataFrameSchema.add_columns`
- {func}`~pandera.api.dataframe.container.DataFrameSchema.remove_columns`
- {func}`~pandera.api.dataframe.container.DataFrameSchema.update_column`
- {func}`~pandera.api.dataframe.container.DataFrameSchema.update_columns`
- {func}`~pandera.api.dataframe.container.DataFrameSchema.rename_columns`
- {func}`~pandera.api.dataframe.container.DataFrameSchema.update_index`
- {func}`~pandera.api.dataframe.container.DataFrameSchema.update_indexes`
- {func}`~pandera.api.dataframe.container.DataFrameSchema.rename_indexes`
- {func}`~pandera.api.dataframe.container.DataFrameSchema.set_index`
- {func}`~pandera.api.dataframe.container.DataFrameSchema.reset_index`
