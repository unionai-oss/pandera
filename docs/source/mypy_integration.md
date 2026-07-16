```{eval-rst}
.. currentmodule:: pandera
```

(mypy-integration)=

# Mypy

*new in 0.8.0*

Pandera integrates with mypy to provide static type-linting of dataframes,
relying on [pandas-stubs](https://github.com/pandas-dev/pandas-stubs)
for typing information.

```bash
pip install pandera[mypy]
```

Then enable the plugin in your `mypy.ini` or `setup.cfg` file:

```
[mypy]
plugins = pandera.mypy
follow_imports = silent
```

:::{important}
Do **not** use ``follow_imports = skip`` in your mypy config when using the
pandera plugin. Skipping import analysis causes ``DataFrame[Schema]`` to
degrade to ``Any``, which disables schema-level and column-level typing.
Use ``follow_imports = silent`` or ``normal`` instead.
:::

:::{note}
Mypy static type-linting is supported for only pandas dataframes.
:::

## Static DataFrameModel field names with ty

`DataFrameModel` replaces field attributes with column-name descriptors at
runtime. Plain dtype annotations are treated as the dtype by a generic static
checker, so they do not by themselves prevent a `ty` false positive when a
field is passed to an API expecting a `str`.

Use the backend-neutral, typing-only
{py:class}`pandera.typing.FieldType` descriptor when that static contract is
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
{ref}`DataFrameModel guide <dataframe-models>`.

:::{warning}
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
:::

In the example below, we define a few schemas to see how type-linting with
pandera works.

```{literalinclude} ../../tests/mypy/pandas_modules/pandas_dataframe.py
:lines: 8-27
```

The mypy linter will complain if the output type of the function body doesn't
match the function's return signature.

```{literalinclude} ../../tests/mypy/pandas_modules/pandas_dataframe.py
:lines: 30-43
```

It'll also complain if the input type doesn't match the expected input type.
Note that we're using the {py:class}`pandera.typing.pandas.DataFrame` generic
type to define dataframes that are validated against the
{py:class}`~pandera.api.pandas.model.DataFrameModel` type variable on initialization.

```{literalinclude} ../../tests/mypy/pandas_modules/pandas_dataframe.py
:lines: 47-60
```

To make mypy happy with respect to the return type, you can either initialize
a dataframe of the expected type:

```{literalinclude} ../../tests/mypy/pandas_modules/pandas_dataframe.py
:lines: 63-64
```

:::{note}
If you construct a dataframe with `DataFrame[Schema](**data)` or
`.pipe(DataFrame[Schema])`, pandera validates it at construction time.
The {py:func}`~pandera.check_types` decorator still runs full schema
validation on inputs and outputs when the function is called, so the same
object may be validated more than once. That is intentional: skipping the
second pass would miss in-place mutations between construction and return.
:::

Or use {py:func}`typing.cast` to indicate to mypy that the return value of
the function is of the correct type.

```{literalinclude} ../../tests/mypy/pandas_modules/pandas_dataframe.py
:lines: 67-68
```

## Limitations

An important caveat to static type-linting with pandera dataframe types is that,
since pandas dataframes are mutable objects, there's no way for `mypy` to
know whether a mutated instance of a
{py:class}`~pandera.api.pandas.model.DataFrameModel`-typed dataframe has the correct
contents. Fortunately, we can simply rely on the {py:func}`~pandera.check_types`
decorator to verify that the output dataframe is valid (including after any
in-place changes). Expect full validation on each decorated call even when
data was already validated at `DataFrame[Schema]` construction; see the note
above.

Consider the examples below:

```{literalinclude} ../../tests/mypy/pandas_modules/pandas_dataframe.py
:lines: 63-80
```

Even though the outputs of these functions are incorrect, mypy doesn't catch
the error during static type-linting but pandera will raise a
{py:class}`~pandera.errors.SchemaError` or {py:class}`~pandera.errors.SchemaErrors`
exception at runtime, depending on whether you're doing
{ref}`lazy validation <lazy-validation>` or not.

```{literalinclude} ../../tests/mypy/pandas_modules/pandas_dataframe.py
:lines: 83-87
```

## Column Access Typing

The mypy plugin can infer column dtypes for bracket access on
{py:class}`~pandera.typing.pandas.DataFrame` instances when the key is a
string literal:

```python
import pandera.pandas as pa
from pandera.typing import DataFrame, Series


class Schema(pa.DataFrameModel):
    id: Series[int]
    name: Series[str]


def fn(df: DataFrame[Schema]) -> Series[int]:
    return df["id"]  # okay with pandera.mypy plugin enabled
```

With the plugin enabled, attribute access is also typed when the column name
matches a model field:

```python
def fn_attr(df: DataFrame[Schema]) -> Series[int]:
    return df.id  # okay with pandera.mypy plugin enabled
```

Bare Python types in model fields (for example ``label: str``) are supported
in addition to ``Series[T]`` annotations.

## Pylance / Pyright

Pylance uses Pyright, which does **not** load the mypy plugin. Schema-level
types such as ``DataFrame[Schema]`` are preserved after
{meth}`~pandera.api.pandas.model.DataFrameModel.validate` and
{func}`~pandera.decorators.check_types`, but column-level inference
(``df["year"]`` as ``Series[int]``) is a mypy-plugin feature and is not
available under Pyright/Pylance. If you need typed column access there,
annotate the result explicitly or use {py:func}`typing.cast`.
