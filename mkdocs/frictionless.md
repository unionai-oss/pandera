# Reading Third-Party Schema

*new in 0.7.0*

Pandera now accepts schema from other data validation frameworks. This requires
a pandera installation with the `io` extension; please see the
[installation](./index.html#install) instructions for more details.

## Frictionless Data Schema

!!! note

    Please see the
    [Frictionless schema](https://specs.frictionlessdata.io/table-schema/)
    documentation for more information on this standard.

### `pandera.io.pandas_io.from_frictionless_schema`

```python
from_frictionless_schema(schema)
```

Create a [`DataFrameSchema`](./reference/generated/pandera.api.pandas.container.DataFrameSchema.html) from either a
frictionless json/yaml schema file saved on disk, or from a frictionless
schema already loaded into memory.

Each field from the frictionless schema will be converted to a pandera
column specification using `FrictionlessFieldParser`
to map field characteristics to pandera column specifications.

**Parameters**

- `schema` (`str | Path | dict | frictionless.Schema`): the frictionless
  schema object (or a string/Path to the location on disk of a schema
  specification) to parse.

**Returns**

- [`DataFrameSchema`](./reference/generated/pandera.api.pandas.container.DataFrameSchema.html): dataframe schema with
  frictionless field specs converted to pandera column checks and
  constraints for use as normal.

**Example**

Here, we're defining a very basic frictionless schema in memory before
parsing it and then querying the resulting
[`DataFrameSchema`](./reference/generated/pandera.api.pandas.container.DataFrameSchema.html) object as per any other Pandera
schema:

```python
from pandera.io.pandas_io import from_frictionless_schema

FRICTIONLESS_SCHEMA = {
    "fields": [
        {
            "name": "column_1",
            "type": "integer",
            "constraints": {"minimum": 10, "maximum": 99}
        },
        {
            "name": "column_2",
            "type": "string",
            "constraints": {"maxLength": 10, "pattern": "\\S+"}
        },
    ],
    "primaryKey": "column_1"
}
schema = from_frictionless_schema(FRICTIONLESS_SCHEMA)
schema.columns["column_1"].checks
# [<Check in_range: in_range(10, 99)>]
schema.columns["column_1"].required
# True
schema.columns["column_1"].unique
# True
schema.columns["column_2"].checks
# [<Check str_length: str_length(None, 10)>, <Check str_matches: str_matches('^\S+$')>]
```

under the hood, this uses the `pandera.io.pandas_io.FrictionlessFieldParser` class
to parse each frictionless field (column):

### `pandera.io.pandas_io.FrictionlessFieldParser`

```python
FrictionlessFieldParser(field, primary_keys)
```

Parses frictionless data schema field specifications so we can convert
them to an equivalent Pandera [`Column`](./reference/generated/pandera.api.pandas.components.Column.html)
schema.

For this implementation, we are using field names, constraints and types
but leaving other frictionless parameters out (e.g. foreign keys, type
formats).

**Parameters**

- `field`: a field object from a frictionless schema.
- `primary_keys`: the primary keys from a frictionless schema. These
  are used to ensure primary key fields are treated properly - no
  duplicates, no missing values etc.

**Members**

- `dtype` (property): determine what type of field this is, so we can feed
  that into `pandera.dtypes.DataType`. If no type is specified in the
  frictionless schema, we default to string values.
- `checks` (property): convert the frictionless field constraints into a list
  of pandera check specifications.
- `nullable` (property): whether the field is nullable.
- `unique` (property): whether the field values must be unique.
- `coerce` (property): whether the field values should be coerced.
- `required` (property): whether the field is required.
- `regex` (property): whether the field name should be treated as a regex.
- `to_pandera_column()`: convert the parsed field into a dictionary
  specification of a pandera `Column`.
