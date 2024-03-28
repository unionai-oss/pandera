---
file_format: mystnb
---

```{currentmodule} pandera
```

(lazy-validation)=

# Lazy Validation

*New in version 0.4.0*

By default, when you call the `validate` method on schema or schema component
objects, a {class}`~pandera.errors.SchemaError` is raised as soon as one of the
assumptions specified in the schema is falsified. For example, for a
{class}`~pandera.api.pandas.container.DataFrameSchema` object, the following situations will raise an
exception:

- a column specified in the schema is not present in the dataframe.
- if `strict=True`, a column in the dataframe is not specified in the schema.
- the `data type` does not match.
- if `coerce=True`, the dataframe column cannot be coerced into the specified
  `data type`.
- the {class}`~pandera.api.checks.Check` specified in one of the columns returns `False` or
  a boolean series containing at least one `False` value.

For example:

```{code-cell} python
import pandas as pd
import pandera as pa

from pandera import Check, Column, DataFrameSchema

df = pd.DataFrame({"column": ["a", "b", "c"]})

schema = pa.DataFrameSchema({"column": Column(int)})

try:
    schema.validate(df)
except pa.errors.SchemaError as exc:
    print(exc)
```

For more complex cases, it is useful to see all of the errors raised during
the `validate` call so that you can debug the causes of errors on different
columns and checks. The `lazy` keyword argument in the `validate` method
of all schemas and schema components gives you the option of doing just this:

```{code-cell} python
import json

import pandas as pd
import pandera as pa

from pandera import Check, Column, DataFrameSchema

schema = pa.DataFrameSchema(
    columns={
        "int_column": Column(int),
        "float_column": Column(float, Check.greater_than(0)),
        "str_column": Column(str, Check.equal_to("a")),
        "date_column": Column(pa.DateTime),
    },
    strict=True
)

df = pd.DataFrame({
    "int_column": ["a", "b", "c"],
    "float_column": [0, 1, 2],
    "str_column": ["a", "b", "d"],
    "unknown_column": None,
})

try:
    schema.validate(df, lazy=True)
except pa.errors.SchemaErrors as exc:
    print(json.dumps(exc.message, indent=2))
```

As you can see from the output above, a {class}`~pandera.errors.SchemaErrors`
exception is raised with a summary of the error counts and failure cases
caught by the schema. This summary is called an {ref}`error-report`.

You can also inspect the failure cases in a more granular form:

```{code-cell} python
try:
    schema.validate(df, lazy=True)
except pa.errors.SchemaErrors as exc:
    print("Schema errors and failure cases:")
    print(exc.failure_cases)
    print("\nDataFrame object that failed validation:")
    print(exc.data)
```
