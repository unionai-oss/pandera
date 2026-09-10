---
file_format: mystnb
---

```{currentmodule} pandera
```

(data-format-conversion)=

# Data Format Conversion

*new in 0.9.0*

The class-based API provides configuration options for converting data to/from
supported serialization formats in the context of
{py:func}`~pandera.decorators.check_types` -decorated functions.

:::{note}
Currently, {py:class}`pandera.typing.pandas.DataFrame` is the only data
type that supports this feature.
:::

Consider this simple example:

```{code-cell} python
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

class InSchema(pa.DataFrameModel):
    str_col: Series[str] = pa.Field(unique=True, isin=[*"abcd"])
    int_col: Series[int]

class OutSchema(InSchema):
    float_col: pa.typing.Series[float]

@pa.check_types
def transform(df: DataFrame[InSchema]) -> DataFrame[OutSchema]:
    return df.assign(float_col=1.1)
```

With the schema type annotations and
{py:func}`~pandera.decorators.check_types` decorator, the `transform`
function validates DataFrame inputs and outputs according to the `InSchema`
and `OutSchema` definitions.

But what if your input data is serialized in parquet format, and you want to
read it into memory, validate the DataFrame, and then pass it to a downstream
function for further analysis? Similarly, what if you want the output of
`transform` to be a list of dictionary records instead of a pandas DataFrame?

## The `to/from_format` Configuration Options

To easily fulfill the use cases described above, you can implement the
read/write logic by hand, or you can configure schemas to do so. We can first
define a subclass of `InSchema` with additional configuration so that our
`transform` function can read data directly from parquet files or buffers:

```{code-cell} python
class InSchemaParquet(InSchema):
    class Config:
        from_format = "parquet"
```

Then, we define subclass of `OutSchema` to specify that `transform`
should output a list of dictionaries representing the rows of the output
dataframe.

```{code-cell} python
class OutSchemaDict(OutSchema):
    class Config:
        to_format = "dict"
        to_format_kwargs = {"orient": "records"}
```

Note that the `{to/from}_format_kwargs` configuration option should be
supplied with a dictionary of key-word arguments to be passed into the
respective pandas `{to/from}_format` method.

Finally, we redefine our `transform` function:

```{code-cell} python
@pa.check_types
def transform(df: DataFrame[InSchemaParquet]) -> DataFrame[OutSchemaDict]:
    return df.assign(float_col=1.1)
```

We can test this out using a buffer to store the parquet file.

:::{note}
A string or path-like object representing the filepath to a parquet file
would also be a valid input to `transform`.
:::

```{code-cell} python
import io
import json

import pandas as pd

buffer = io.BytesIO()
data = pd.DataFrame({"str_col": [*"abc"], "int_col": range(3)})
data.to_parquet(buffer)
buffer.seek(0)

dict_output = transform(buffer)
print(json.dumps(dict_output, indent=2))
```

## Custom Converters with Callables

In addition to specifying a literal string argument for `from_format` a
generic callable that returns a pandas dataframe can be passed. For example,
`pd.read_excel`, `pd.read_sql`, or `pd.read_gbq`. Depending on the function
passed, some of the kwargs arguments may be required rather than optional in
`from_format_kwargs` (`pd.read_sql` requires a connection object).

A callable can also be an argument for the `to_format` parameter, with the
additional, optional, `to_format_buffer` parameter. Some pandas dataframe writing
methods, such as `pd.to_pickle`, have a required path argument, that must be
either a string file path or a bytes object. An example for writing data to a
pickle file would be:

```{code-cell} python
import tempfile

def custom_to_pickle(data, *args, **kwargs):
    return data.to_pickle(*args, **kwargs)

def custom_to_pickle_buffer():
    """Create a named temporary file handle to write the pickle file."""
    return tempfile.NamedTemporaryFile()

class OutSchemaPickleCallable(OutSchema):
    class Config:
        to_format = custom_to_pickle

        # If provided, the output of this function will be supplied as
        # the first positional argument to the ``to_format`` function.
        to_format_buffer = custom_to_pickle_buffer
```

In this example, we use a `custom_to_pickle_buffer` function as the
`to_format_buffer` property, which returns a {func}`tempfile.NamedTemporaryFile`.
This will be supplied as a positional argument to the `custom_to_pickle`
function.

The full set of configuration options are:

```{eval-rst}
.. list-table:: Title
   :widths: 50 60
   :header-rows: 1

   * - Format
     - Argument
   * - dict
     - "dict"
   * - csv
     - "csv"
   * - json
     - "json"
   * - feather
     - "feather"
   * - parquet
     - "parquet"
   * - pickle
     - "pickle"
   * - Callable
     - Callable
```

## Takeaway

Data Format Conversion using the `{to/from}_format` configuration option
can modify the behavior of {py:func}`~pandera.decorators.check_types` -decorated
functions to convert input data from a particular serialization format into
a dataframe. Additionally, you can convert the output data from a dataframe to
potentially another format.

This dovetails well with the {ref}`FastAPI Integration <fastapi-integration>`
for validating the inputs and outputs of app endpoints.
