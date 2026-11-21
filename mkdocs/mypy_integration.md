# Mypy

*new in 0.8.0*

Pandera integrates with mypy to provide static type-linting of dataframes,
relying on [pandas-stubs](https://github.com/pandas-dev/pandas-stubs)
for typing information.

```bash
pip install pandera[mypy]
```

Then enable the plugin in your `mypy.ini` or `setug.cfg` file:

```
[mypy]
plugins = pandera.mypy
```

!!! note

    Mypy static type-linting is supported for only pandas dataframes.

## Static DataFrameModel field names with ty

`DataFrameModel` replaces field attributes with column-name descriptors at
runtime. Plain dtype annotations are treated as the dtype by a generic static
checker, so they do not by themselves prevent a `ty` false positive when a
field is passed to an API expecting a `str`.

Use the backend-neutral, typing-only
[`FieldType`](./reference/generated/pandera.typing.FieldType.html) descriptor when that static contract is
needed:

```python
import polars as pl
import pandera.polars as pa
from pandera.typing import FieldType


class Schema(pa.DataFrameModel):
    values: FieldType[pl.List] = pa.Field()
    nullable: FieldType[int | None] = pa.Field()
    optional: FieldType[int] | None


def accepts_name(name: str) -> str:
    return name


accepts_name(Schema.values)  # accepted by ty without a Pandera plugin
```

`FieldType[T]` models class-level access as `str` and leaves `T` available to
Pandera's runtime parser. It is not instantiated. The backend-specific
`pa.Field(...)` remains the runtime metadata object, either as the assigned
value shown above or as an additional `FieldType` metadata argument:
`FieldType[T, pa.Field(...)]`. Because generic checkers may reject calls inside
type arguments, use the assignment form shown above when running `ty`. The
descriptor does not change the required, nullable, optional-presence,
explicit-override, or legacy
`Optional[Series[T]]` semantics documented in the
[DataFrameModel guide](./dataframe_models.html).

!!! warning

    This functionality is experimental 🧪. Since the
    [pandas-stubs](https://github.com/pandas-dev/pandas-stubs) type stub
    annotations don't always match the official
    [pandas effort to support type annotations](https://github.com/pandas-dev/pandas/issues/28142#issuecomment-991967009),
    installing the `pandera[mypy]` extra may yield false positives in your
    pandas code, many of which are are documented in `tests/mypy/pandas_modules`
    (see [here](https://github.com/unionai-oss/pandera/tree/main/tests/mypy/pandas_modules) ).

    We encourage you to [file an issue](https://github.com/pandera-dev/pandera/issues/new?assignees=&labels=bug,mypy&template=bug_report.md&title=)
    if you find any false positives or negatives being reported by `mypy`.
    A list of such issues can be found [here](https://github.com/pandera-dev/pandera/labels/mypy).
    We'll most likely have to escalate this to the official `pandas-stubs`
    [issues](https://github.com/pandas-dev/pandas-stubs/issues).

In the example below, we define a few schemas to see how type-linting with
pandera works.

```python
from typing import Optional, cast

import pandas as pd

import pandera.pandas as pa
from pandera.typing import DataFrame, Series


class Schema(pa.DataFrameModel):
    id: Series[int]
    name: Series[str]


class SchemaOut(pa.DataFrameModel):
    age: Series[int]


class AnotherSchema(pa.DataFrameModel):
    id: Series[int]
    first_name: Series[str] | None
```

The mypy linter will complain if the output type of the function body doesn't
match the function's return signature.

```python
def fn(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    return df.assign(age=30).pipe(DataFrame[SchemaOut])  # mypy okay


def fn_pipe_incorrect_type(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    return df.assign(age=30).pipe(DataFrame[AnotherSchema])  # mypy error
    # error: Argument 1 to "pipe" of "NDFrame" has incompatible type "Type[DataFrame[Any]]";  # noqa
    # expected "Union[Callable[..., DataFrame[SchemaOut]], Tuple[Callable[..., DataFrame[SchemaOut]], str]]"  [arg-type]  # noqa


def fn_assign_copy(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    return df.assign(age=30)  # mypy error
    # error: Incompatible return value type (got "pandas.core.frame.DataFrame",
    # expected "pandera.typing.pandas.DataFrame[SchemaOut]")  [return-value]
```

It'll also complain if the input type doesn't match the expected input type.
Note that we're using the `pandera.typing.pandas.DataFrame` generic
type to define dataframes that are validated against the
[`DataFrameModel`](./reference/generated/pandera.api.pandas.model.DataFrameModel.html) type variable on initialization.

```python
# Define a few dataframe objects
schema_df = DataFrame[Schema]({"id": [1], "name": ["foo"]})
pandas_df = pd.DataFrame({"id": [1], "name": ["foo"]})
another_df = DataFrame[AnotherSchema]({"id": [1], "first_name": ["foo"]})


fn(schema_df)  # mypy okay

fn(pandas_df)  # mypy error
# error: Argument 1 to "fn" has incompatible type "pandas.core.frame.DataFrame";  # noqa
# expected "pandera.typing.pandas.DataFrame[Schema]"  [arg-type]

fn(another_df)  # mypy error
# error: Argument 1 to "fn" has incompatible type "DataFrame[AnotherSchema]";
# expected "DataFrame[Schema]"  [arg-type]
```

To make mypy happy with respect to the return type, you can either initialize
a dataframe of the expected type:

```python
def fn_pipe_dataframe(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    return df.assign(age=30).pipe(DataFrame[SchemaOut])  # mypy okay
```

!!! note

    If you construct a dataframe with `DataFrame[Schema](**data)` or
    `.pipe(DataFrame[Schema])`, pandera validates it at construction time.
    The [`check_types`](./reference/generated/pandera.decorators.check_types.html) decorator still runs full schema
    validation on inputs and outputs when the function is called, so the same
    object may be validated more than once. That is intentional: skipping the
    second pass would miss in-place mutations between construction and return.

Or use `typing.cast` to indicate to mypy that the return value of
the function is of the correct type.

```python
def fn_cast_dataframe(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    return cast(DataFrame[SchemaOut], df.assign(age=30))  # mypy okay
```

## Limitations

An important caveat to static type-linting with pandera dataframe types is that,
since pandas dataframes are mutable objects, there's no way for `mypy` to
know whether a mutated instance of a
[`DataFrameModel`](./reference/generated/pandera.api.pandas.model.DataFrameModel.html)-typed dataframe has the correct
contents. Fortunately, we can simply rely on the [`check_types`](./reference/generated/pandera.decorators.check_types.html)
decorator to verify that the output dataframe is valid (including after any
in-place changes). Expect full validation on each decorated call even when
data was already validated at `DataFrame[Schema]` construction; see the note
above.

Consider the examples below:

```python
def fn_pipe_dataframe(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    return df.assign(age=30).pipe(DataFrame[SchemaOut])  # mypy okay


def fn_cast_dataframe(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    return cast(DataFrame[SchemaOut], df.assign(age=30))  # mypy okay


@pa.check_types
def fn_mutate_inplace(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    out = df.assign(age=30).pipe(DataFrame[SchemaOut])
    out.drop(columns="age", inplace=True)
    return out  # okay for mypy, pandera raises error


@pa.check_types
def fn_assign_and_get_index(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    return df.assign(foo=30).iloc[:3]  # mypy error
```

Even though the outputs of these functions are incorrect, mypy doesn't catch
the error during static type-linting but pandera will raise a
[`SchemaError`](./reference/generated/pandera.errors.SchemaError.html) or [`SchemaErrors`](./reference/generated/pandera.errors.SchemaErrors.html)
exception at runtime, depending on whether you're doing
[lazy validation](./lazy_validation.html) or not.

```python
@pa.check_types
def fn_cast_dataframe_invalid(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    return cast(
        DataFrame[SchemaOut], df
    )  # okay for mypy, pandera raises error
```
