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
import pandera.pandas as pa


df = pd.DataFrame({"column": ["a", "b", "c"]})

schema = pa.DataFrameSchema({"column": pa.Column(int)})

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
import pandera.pandas as pa


schema = pa.DataFrameSchema(
    columns={
        "int_column": pa.Column(int),
        "float_column": pa.Column(float, pa.Check.greater_than(0)),
        "str_column": pa.Column(str, pa.Check.equal_to("a")),
        "date_column": pa.Column(pa.DateTime),
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

(n-failure-cases)=

## Limiting reported failure cases with `n_failure_cases`

When a {class}`~pandera.api.checks.Check` fails, pandera collects the failing
values so you can inspect them via `SchemaError.failure_cases` /
`SchemaErrors.failure_cases`. By default, **all** failure cases are reported
(`n_failure_cases=None`).

For large datasets this can produce very large failure reports. Pass an integer
`n_failure_cases` to truncate the reported cases to the first *n* unique
failures for that check:

```{code-cell} python
import pandas as pd
import pandera.pandas as pa

df = pd.DataFrame({"n": range(20)})

# Default: report every failure case (n_failure_cases=None)
schema_all = pa.DataFrameSchema({
    "n": pa.Column(int, pa.Check.greater_than(30)),
})

# Limit: report only the first 5 failure cases for this check
schema_limited = pa.DataFrameSchema({
    "n": pa.Column(
        int,
        pa.Check.greater_than(30, n_failure_cases=5),
    ),
})

try:
    schema_all.validate(df, lazy=True)
except pa.errors.SchemaErrors as exc:
    print(f"all failures: {len(exc.failure_cases)}")

try:
    schema_limited.validate(df, lazy=True)
except pa.errors.SchemaErrors as exc:
    print(f"limited failures: {len(exc.failure_cases)}")
    print(exc.failure_cases)
```

You can set the same option on {func}`~pandera.api.dataframe.model_components.Field`
when using the class-based {class}`~pandera.api.pandas.model.DataFrameModel`
API:

```{code-cell} python
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series


class Schema(pa.DataFrameModel):
    n: Series[int] = pa.Field(gt=30, n_failure_cases=5)


try:
    Schema.validate(pd.DataFrame({"n": range(20)}), lazy=True)
except pa.errors.SchemaErrors as exc:
    print(exc.failure_cases)
```

```{note}
`n_failure_cases` only controls how many failure cases are **reported** for a
check. It does not change whether the check passes or fails, and it does not
drop invalid rows from the data. Use {ref}`drop-invalid-rows` if you need to
filter invalid data out of the validated object.
```
