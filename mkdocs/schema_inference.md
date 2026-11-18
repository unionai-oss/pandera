# Schema Inference and Persistence

*New in version 0.4.0*

With simple use cases, writing a schema definition manually is pretty
straight-forward with pandera. However, it can get tedious to do this with
dataframes that have many columns of various data types.

## Inferring a schema from data

To help you handle these cases, the [`infer_schema`](./reference/generated/pandera.schema_inference.pandas.infer_schema.html) function enables
you to quickly infer a draft schema from a pandas dataframe or series. Below
is a simple example:

```python
import pandas as pd
import pandera.pandas as pa

df = pd.DataFrame({
    "column1": [5, 10, 20],
    "column2": ["a", "b", "c"],
    "column3": pd.to_datetime(["2010", "2011", "2012"]),
})
schema = pa.infer_schema(df)
print(schema)
```

These inferred schemas are **rough drafts** that shouldn't be used for
validation without modification. You can modify the inferred schema to
obtain the schema definition that you're satisfied with.

For [`DataFrameSchema`](./reference/generated/pandera.api.pandas.container.DataFrameSchema.html) objects, the following methods create
modified copies of the schema:

- [`add_columns`](./reference/generated/pandera.api.pandas.container.DataFrameSchema.html)
- [`remove_columns`](./reference/generated/pandera.api.pandas.container.DataFrameSchema.html)
- [`update_column`](./reference/generated/pandera.api.pandas.container.DataFrameSchema.html)

For [`SeriesSchema`](./reference/generated/pandera.api.pandas.array.SeriesSchema.html) objects:

- [`set_checks`](./reference/generated/pandera.api.pandas.array.SeriesSchema.html)

The section below describes two workflows for persisting and modifying an
inferred schema.

## Persisting a schema

The schema persistence feature requires a pandera installation with the `io`
extension. See the [installation](./index.html#install) instructions for more
details.

There are two ways of persisting schemas, inferred or otherwise.

### Write to a Python script

You can also write your schema to a python script with [`to_script`](./reference/generated/pandera.io.pandas_io.to_script.html):

```python
# supply a file-like object, Path, or str to write to a file. If not
# specified, to_script will output the code as a string.
schema_script = schema.to_script()
print(schema_script)
```

As a python script, you can iterate on an inferred schema and use it to
validate data once you are satisfied with your schema definition.

### Write to YAML

You can also write the schema object to a yaml file with [`to_yaml`](./reference/generated/pandera.io.pandas_io.to_yaml.html),
and you can then read it into memory with [`from_yaml`](./reference/generated/pandera.io.pandas_io.from_yaml.html). The
[`DataFrameSchema.to_yaml`](./reference/generated/pandera.api.pandas.container.DataFrameSchema.html) and [`DataFrameSchema.from_yaml`](./reference/generated/pandera.api.pandas.container.DataFrameSchema.html)
is a convenience method for this functionality.

```python
# supply a file-like object, Path, or str to write to a file. If not
# specified, to_yaml will output a yaml string.
yaml_schema = schema.to_yaml()
print(yaml_schema)
```

You can edit this yaml file to modify the schema. For example, you can specify
new column names under the `column` key, and the respective values map onto
key-word arguments in the [`Column`](./reference/generated/pandera.api.pandas.components.Column.html) class.

!!! note

    Currently, only built-in [`Check`](./reference/generated/pandera.api.checks.Check.html) methods are supported under the
    `checks` key.

### Write to JSON

Finally, you can also write the schema object to a json file with [`to_json`](./reference/generated/pandera.io.pandas_io.to_json.html),
and you can then read it into memory with [`from_json`](./reference/generated/pandera.io.pandas_io.from_json.html). The
[`DataFrameSchema.to_json`](./reference/generated/pandera.api.pandas.container.DataFrameSchema.html) and [`DataFrameSchema.from_json`](./reference/generated/pandera.api.pandas.container.DataFrameSchema.html)
is a convenience method for this functionality.

```python
# supply a file-like object, Path, or str to write to a file. If not
# specified, to_yaml will output a yaml string.
json_schema = schema.to_json(indent=4)
print(json_schema)
```

You can edit this json file to update the schema as needed, and then load
it back into a pandera schema object with [`from_json`](./reference/generated/pandera.io.pandas_io.from_json.html) or
[`DataFrameSchema.from_json`](./reference/generated/pandera.api.pandas.container.DataFrameSchema.html).
