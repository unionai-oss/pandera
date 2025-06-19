---
file_format: mystnb
---

```{currentmodule} pandera
```

(schema-inference)=

# Schema Inference and Persistence

*New in version 0.4.0*

With simple use cases, writing a schema definition manually is pretty
straight-forward with pandera. However, it can get tedious to do this with
dataframes that have many columns of various data types.

## Inferring a schema from data

To help you handle these cases, the {func}`~pandera.schema_inference.pandas.infer_schema` function enables
you to quickly infer a draft schema from a pandas dataframe or series. Below
is a simple example:

```{code-cell} python
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

For {class}`~pandera.api.pandas.container.DataFrameSchema` objects, the following methods create
modified copies of the schema:

- {func}`~pandera.api.pandas.container.DataFrameSchema.add_columns`
- {func}`~pandera.api.pandas.container.DataFrameSchema.remove_columns`
- {func}`~pandera.api.pandas.container.DataFrameSchema.update_column`

For {class}`~pandera.api.pandas.array.SeriesSchema` objects:

- {func}`~pandera.api.pandas.array.SeriesSchema.set_checks`

The section below describes two workflows for persisting and modifying an
inferred schema.

(schema-persistence)=

## Persisting a schema

The schema persistence feature requires a pandera installation with the `io`
extension. See the {ref}`installation<installation>` instructions for more
details.

There are two ways of persisting schemas, inferred or otherwise.

### Write to a Python script

You can also write your schema to a python script with {func}`~pandera.io.to_script`:

```{code-cell} python
# supply a file-like object, Path, or str to write to a file. If not
# specified, to_script will output the code as a string.
schema_script = schema.to_script()
print(schema_script)
```

As a python script, you can iterate on an inferred schema and use it to
validate data once you are satisfied with your schema definition.

### Write to YAML

You can also write the schema object to a yaml file with {func}`~pandera.io.to_yaml`,
and you can then read it into memory with {func}`~pandera.io.from_yaml`. The
{func}`~pandera.api.pandas.container.DataFrameSchema.to_yaml` and {func}`~pandera.api.pandas.container.DataFrameSchema.from_yaml`
is a convenience method for this functionality.

```{code-cell} python
# supply a file-like object, Path, or str to write to a file. If not
# specified, to_yaml will output a yaml string.
yaml_schema = schema.to_yaml()
print(yaml_schema)
```

You can edit this yaml file to modify the schema. For example, you can specify
new column names under the `column` key, and the respective values map onto
key-word arguments in the {class}`~pandera.api.pandas.components.Column` class.

```{note}
Currently, only built-in {class}`~pandera.api.checks.Check` methods are supported under the
`checks` key.
```

### Write to JSON

Finally, you can also write the schema object to a json file with {func}`~pandera.io.to_json`,
and you can then read it into memory with {func}`~pandera.io.from_json`. The
{func}`~pandera.api.pandas.container.DataFrameSchema.to_json` and {func}`~pandera.api.pandas.container.DataFrameSchema.from_json`
is a convenience method for this functionality.

```{code-cell} python
# supply a file-like object, Path, or str to write to a file. If not
# specified, to_yaml will output a yaml string.
json_schema = schema.to_json(indent=4)
print(json_schema)
```

You can edit this json file to update the schema as needed, and then load
it back into a pandera schema object with {func}`~pandera.io.from_json` or
{func}`~pandera.api.pandas.container.DataFrameSchema.from_json`.
