---
file_format: mystnb
---

% pandera documentation for class-based API.

```{currentmodule} pandera
```

(dataframe-models)=

# DataFrame Models

*new in 0.5.0*


`pandera` provides a class-based API that's heavily inspired by
[pydantic](https://pydantic-docs.helpmanual.io/). In contrast to the
{ref}`object-based API<DataFrameSchemas>`, you can define dataframe models in
much the same way you'd define `pydantic` models.

`DataFrameModel` s are annotated with the {mod}`pandera.typing` module using the standard
[typing](https://docs.python.org/3/library/typing.html) syntax. Models can be
explicitly converted to a {class}`~pandera.api.pandas.container.DataFrameSchema` or used to validate a
{class}`~pandas.DataFrame` directly.

:::{note}
Due to current limitations in the pandas library (see discussion
[here](https://github.com/pandera-dev/pandera/issues/253#issuecomment-665338337)),
`pandera` annotations are only used for **run-time** validation and has
limited support for static-type checkers like [mypy](http://mypy-lang.org/).
See the {ref}`Mypy Integration <mypy-integration>` for more details.
:::

## Basic Usage

```{code-cell} python
import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import Index, DataFrame, Series


class InputSchema(pa.DataFrameModel):
    year: Series[int] = pa.Field(gt=2000, coerce=True)
    month: Series[int] = pa.Field(ge=1, le=12, coerce=True)
    day: Series[int] = pa.Field(ge=0, le=365, coerce=True)

class OutputSchema(InputSchema):
    revenue: Series[float]

@pa.check_types
def transform(df: DataFrame[InputSchema]) -> DataFrame[OutputSchema]:
    return df.assign(revenue=100.0)


df = pd.DataFrame({
    "year": ["2001", "2002", "2003"],
    "month": ["3", "6", "12"],
    "day": ["200", "156", "365"],
})

transform(df)

invalid_df = pd.DataFrame({
    "year": ["2001", "2002", "1999"],
    "month": ["3", "6", "12"],
    "day": ["200", "156", "365"],
})

try:
    transform(invalid_df)
except pa.errors.SchemaError as exc:
    print(exc)
```

As you can see in the examples above, you can define a schema by sub-classing
{class}`~pandera.api.pandas.model.DataFrameModel` and defining column/index fields as class attributes.
The {func}`~pandera.decorators.check_types` decorator is required to perform validation of the dataframe at
run-time.

Note that {class}`~pandera.api.dataframe.model_components.Field` s apply to both
{class}`~pandera.api.pandas.components.Column` and {class}`~pandera.api.pandas.components.Index`
objects, exposing the built-in {class}`Check` s via key-word arguments.

*(New in 0.6.2)* When you access a class attribute defined on the schema,
it will return the name of the column used in the validated `pd.DataFrame`.
In the example above, this will simply be the string `"year"`.

```{code-cell} python
print(f"Column name for 'year' is {InputSchema.year}\n")
print(df.loc[:, [InputSchema.year, "day"]])
```

## Using Data Types directly for Column Type Annotations

*new in 0.15.0*

For conciseness, you can also use type annotations for columns without using
the {py:class}`~pandera.typing.Series` generic. This class attributes will be
interpreted as {py:class}`~pandera.api.pandas.components.Column` objects
under the hood.

```{code-cell} python
class InputSchema(pa.DataFrameModel):
    year: int = pa.Field(gt=2000, coerce=True)
    month: int = pa.Field(ge=1, le=12, coerce=True)
    day: int = pa.Field(ge=0, le=365, coerce=True)
```

### Reusing Field objects

To define reusable `Field` definitions, you need to use `functools.partial`.
This makes sure that each field attribute is bound to a unique `Field` instance.

```{code-cell} python
from functools import partial
from pandera.pandas import DataFrameModel, Field

NormalizedField = partial(Field, ge=0, le=1)

class SchemaWithReusedFields(DataFrameModel):
    xnorm: float = NormalizedField()
    ynorm: float = NormalizedField()
```

## Validate on Initialization

*new in 0.8.0*

Pandera provides an interface for validating dataframes on initialization.
This API uses the {py:class}`pandera.typing.pandas.DataFrame` generic type
to validated against the {py:class}`~pandera.api.pandas.model.DataFrameModel` type variable
on initialization:

```{code-cell} python
import pandas as pd
import pandera.pandas as pa

from pandera.typing import DataFrame, Series


class Schema(pa.DataFrameModel):
    state: Series[str]
    city: Series[str]
    price: Series[int] = pa.Field(in_range={"min_value": 5, "max_value": 20})

DataFrame[Schema](
    {
        'state': ['NY','FL','GA','CA'],
        'city': ['New York', 'Miami', 'Atlanta', 'San Francisco'],
        'price': [8, 12, 10, 16],
    }
)
```

Refer to {ref}`supported-dataframe-libraries` to see how this syntax applies
to other supported dataframe types.

## Converting to DataFrameSchema

You can easily convert a {class}`~pandera.api.pandas.model.DataFrameModel` class into a
{class}`~pandera.api.pandas.container.DataFrameSchema`:

```{code-cell} python
print(InputSchema.to_schema())
```

You can also use the {meth}`~pandera.api.pandas.model.DataFrameModel.validate` method to
validate dataframes:

```{code-cell} python
print(InputSchema.validate(df))
```

Or you can use the {meth}`~pandera.api.pandas.model.DataFrameModel` class directly to
validate dataframes, which is syntactic sugar that simply delegates to the
{meth}`~pandera.api.pandas.model.DataFrameModel.validate` method.

```{code-cell} python
print(InputSchema(df))
```

## Validate Against Multiple Schemas

*new in 0.14.0*

The built-in {class}`typing.Union` type is supported for multiple `DataFrame` schemas.

```{code-cell} python
from typing import Union
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

class OnlyZeroesSchema(pa.DataFrameModel):
    a: Series[int] = pa.Field(eq=0)

class OnlyOnesSchema(pa.DataFrameModel):
    a: Series[int] = pa.Field(eq=1)

@pa.check_types
def return_zeros_or_ones(
    df: Union[DataFrame[OnlyZeroesSchema], DataFrame[OnlyOnesSchema]]
) -> Union[DataFrame[OnlyZeroesSchema], DataFrame[OnlyOnesSchema]]:
    return df

# passes
return_zeros_or_ones(pd.DataFrame({"a": [0, 0]}))
return_zeros_or_ones(pd.DataFrame({"a": [1, 1]}))

# fails
try:
    return_zeros_or_ones(pd.DataFrame({"a": [0, 2]}))
except pa.errors.SchemaErrors as exc:
    print(exc)
```

Note that mixtures of `DataFrame` schemas and built-in types will ignore checking built-in types
with pandera. Pydantic should be used to check and/or coerce any built-in types.

```{code-cell} python
import pandas as pd
from typing import Union
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

class OnlyZeroesSchema(pa.DataFrameModel):
    a: Series[int] = pa.Field(eq=0)


@pa.check_types
def df_and_int_types(

    val: Union[DataFrame[OnlyZeroesSchema], int]
) -> Union[DataFrame[OnlyZeroesSchema], int]:
    return val


df_and_int_types(pd.DataFrame({"a": [0, 0]}))
int_val = df_and_int_types(5)
str_val = df_and_int_types("5")

no_pydantic_report = f"No Pydantic: {isinstance(int_val, int)}, {isinstance(str_val, int)}"


@pa.check_types(with_pydantic=True)
def df_and_int_types_with_pydantic(
    val: Union[DataFrame[OnlyZeroesSchema], int]
) -> Union[DataFrame[OnlyZeroesSchema], int]:
    return val


df_and_int_types_with_pydantic(pd.DataFrame({"a": [0, 0]}))
int_val_w_pyd = df_and_int_types_with_pydantic(5)
str_val_w_pyd = df_and_int_types_with_pydantic("5")

pydantic_report = f"With Pydantic: {isinstance(int_val_w_pyd, int)}, {isinstance(str_val_w_pyd, int)}"

print(no_pydantic_report)
print(pydantic_report)
```

## Excluded attributes

Class variables which begin with an underscore will be automatically excluded from
the model. {ref}`Config <schema-model-config>` is also a reserved name.
However, {ref}`aliases <schema-model-alias>` can be used to circumvent these limitations.

## Supported dtypes

Any dtypes supported by `pandera` can be used as type parameters for
{class}`~pandera.typing.Series` and {class}`~pandera.typing.Index`. There are,
however, a couple of gotchas.

:::{important}
You can learn more about how data type validation works
{doc}`dtype_validation`.
:::

### Dtype aliases

```
import pandera.pandas as pa
from pandera.typing import Series, String

class Schema(pa.DataFrameModel):
    a: Series[String]
```

### Type Vs instance

You must give a **type**, not an **instance**.

✅ Good:

```{code-cell} python
import pandas as pd

class Schema(pa.DataFrameModel):
    a: Series[pd.StringDtype]
```

❌ Bad:

:::{note}
This is only applicable for pandas versions \< 2.0.0. In pandas > 2.0.0,
pd.StringDtype() will produce a type.
:::

```{code-cell} python
:tags: [raises-exception]

class Schema(pa.DataFrameModel):
    a: Series[pd.StringDtype()]
```


(parameterized-dtypes)=

### Parametrized dtypes

Pandas supports a couple of parametrized dtypes. As of pandas 1.2.0:

| Kind of Data      | Data Type                 | Parameters              |
| ----------------- | ------------------------- | ----------------------- |
| tz-aware datetime | {class}`DatetimeTZDtype`  | `unit`, `tz`            |
| Categorical       | {class}`CategoricalDtype` | `categories`, `ordered` |
| period            | {class}`PeriodDtype`      | `freq`                  |
| sparse            | {class}`SparseDtype`      | `dtype`, `fill_value`   |
| intervals         | {class}`IntervalDtype`    | `subtype`               |

#### Annotated

Parameters can be given via {data}`typing.Annotated`. It requires python >= 3.9 or
[typing_extensions](https://pypi.org/project/typing-extensions/), which is already a
requirement of Pandera. Unfortunately {data}`typing.Annotated` has not been backported
to python 3.6.

✅ Good:

```{code-cell} python
try:
    from typing import Annotated  # python 3.9+
except ImportError:
    from typing_extensions import Annotated

class Schema(pa.DataFrameModel):
    col: Series[Annotated[pd.DatetimeTZDtype, "ns", "est"]]
```

Furthermore, you must pass all parameters in the order defined in the dtype's
constructor (see {ref}`table <parameterized-dtypes>`).

❌ Bad:

```{code-cell} python
:tags: [raises-exception]

class Schema(pa.DataFrameModel):
    col: Series[Annotated[pd.DatetimeTZDtype, "utc"]]

Schema.to_schema()
```

#### Field

✅ Good:

```{code-cell} python
class SchemaFieldDatetimeTZDtype(pa.DataFrameModel):
    col: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"unit": "ns", "tz": "EST"}
    )
```

You cannot use both {data}`typing.Annotated` and `dtype_kwargs`.

❌ Bad:

```{code-cell} python
:tags: [raises-exception]

class SchemaFieldDatetimeTZDtype(pa.DataFrameModel):
    col: Series[Annotated[pd.DatetimeTZDtype, "ns", "est"]] = pa.Field(
        dtype_kwargs={"unit": "ns", "tz": "EST"}
    )

Schema.to_schema()
```

## Required Columns

By default all columns specified in the schema are {ref}`required<required>`, meaning
that if a column is missing in the input DataFrame an exception will be
thrown. If you want to make a column optional, annotate it with {data}`typing.Optional`.

```{code-cell} python
from typing import Optional

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series


class Schema(pa.DataFrameModel):
    a: Series[str]
    b: Optional[Series[int]]

df = pd.DataFrame({"a": ["2001", "2002", "2003"]})
Schema.validate(df)
```

## Schema Inheritance

You can also use inheritance to build schemas on top of a base schema.

```{code-cell} python
class BaseSchema(pa.DataFrameModel):
    year: Series[str]

class FinalSchema(BaseSchema):
    year: Series[int] = pa.Field(ge=2000, coerce=True)  # overwrite the base type
    passengers: Series[int]
    idx: Index[int] = pa.Field(ge=0)

df = pd.DataFrame({
    "year": ["2000", "2001", "2002"],
})

@pa.check_types
def transform(df: DataFrame[BaseSchema]) -> DataFrame[FinalSchema]:
    return (
        df.assign(passengers=[61000, 50000, 45000])
        .set_index(pd.Index([1, 2, 3]))
        .astype({"year": int})
    )

transform(df)
```

(schema-model-config)=

## Config

Schema-wide options can be controlled via the `Config` class on the `DataFrameModel`
subclass. The full set of options can be found in the {class}`~pandera.api.pandas.model_config.BaseConfig`
class.

```{code-cell} python
class Schema(pa.DataFrameModel):

    year: Series[int] = pa.Field(gt=2000, coerce=True)
    month: Series[int] = pa.Field(ge=1, le=12, coerce=True)
    day: Series[int] = pa.Field(ge=0, le=365, coerce=True)

    class Config:
        name = "BaseSchema"
        strict = True
        coerce = True
        foo = "bar"  # Interpreted as dataframe check
        baz = ...    # Interpreted as a dataframe check with no additional arguments
```

It is not required for the `Config` to subclass
{class}`~pandera.api.pandas.model_config.BaseConfig` but
it **must** be named '**Config**'.

See {ref}`class-based-api-dataframe-checks` for details on using registered dataframe checks.

## MultiIndex

The {class}`~pandera.api.pandas.components.MultiIndex` capabilities are also supported with
the class-based API:

```{code-cell} python
import pandera.pandas as pa
from pandera.typing import Index, Series

class MultiIndexSchema(pa.DataFrameModel):

    year: Index[int] = pa.Field(gt=2000, coerce=True)
    month: Index[int] = pa.Field(ge=1, le=12, coerce=True)
    passengers: Series[int]

    class Config:
        # provide multi index options in the config
        multiindex_name = "time"
        multiindex_strict = True
        multiindex_coerce = True

index = MultiIndexSchema.to_schema().index
print(index)
```

```{code-cell} python
from pprint import pprint

pprint({name: col.checks for name, col in index.columns.items()})
```

Multiple {class}`~pandera.typing.Index` annotations are automatically converted into a
{class}`~pandera.api.pandas.components.MultiIndex`. MultiIndex options are given in the
{ref}`schema-model-config`.

## Index Name

Use `check_name` to validate the index name of a single-index dataframe:

```{code-cell} python
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Index, Series

class Schema(pa.DataFrameModel):
    year: Series[int] = pa.Field(gt=2000, coerce=True)
    passengers: Series[int]
    idx: Index[int] = pa.Field(ge=0, check_name=True)

df = pd.DataFrame({
    "year": [2001, 2002, 2003],
    "passengers": [61000, 50000, 45000],
})

try:
    Schema.validate(df)
except pa.errors.SchemaError as exc:
    print(exc)
```

`check_name` default value of `None` translates to `True` for columns and multi-index.

(schema-model-custom-check)=

## Custom Checks

Unlike the object-based API, custom checks can be specified as class methods.

### Column/Index checks

```{code-cell} python
import pandera.pandas as pa
from pandera.typing import Index, Series

class CustomCheckSchema(pa.DataFrameModel):

    a: Series[int] = pa.Field(gt=0, coerce=True)
    abc: Series[int]
    idx: Index[str]

    @pa.check("a", name="foobar")
    def custom_check(cls, a: Series[int]) -> Series[bool]:
        return a < 100

    @pa.check("^a", regex=True, name="foobar")
    def custom_check_regex(cls, a: Series[int]) -> Series[bool]:
        return a > 0

    @pa.check("idx")
    def check_idx(cls, idx: Index[int]) -> Series[bool]:
        return idx.str.contains("dog")
```

:::{note}
- You can supply the key-word arguments of the {class}`~pandera.api.checks.Check` class
  initializer to get the flexibility of {ref}`groupby checks <column-check-groups>`
- Similarly to `pydantic`, {func}`classmethod` decorator is added behind the scenes
  if omitted.
- You still may need to add the `@classmethod` decorator *after* the
  {func}`~pandera.api.dataframe.model_components.check` decorator if your static-type checker or
  linter complains.
- Since `checks` are class methods, the first argument value they receive is a
  DataFrameModel subclass, not an instance of a model.
:::

```{code-cell} python
from typing import Dict

class GroupbyCheckSchema(pa.DataFrameModel):

    value: Series[int] = pa.Field(gt=0, coerce=True)
    group: Series[str] = pa.Field(isin=["A", "B"])

    @pa.check("value", groupby="group", regex=True, name="check_means")
    def check_groupby(cls, grouped_value: Dict[str, Series[int]]) -> bool:
        return grouped_value["A"].mean() < grouped_value["B"].mean()

df = pd.DataFrame({
    "value": [100, 110, 120, 10, 11, 12],
    "group": list("AAABBB"),
})

try:
    print(GroupbyCheckSchema.validate(df))
except pa.errors.SchemaError as exc:
    print(exc)
```

(schema-model-dataframe-check)=

### DataFrame Checks

You can also define dataframe-level checks, similar to the
{ref}`object-based API <wide-checks>`, using the
{func}`~pandera.api.pandas.components.dataframe_check` decorator:

```{code-cell} python
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Index, Series

class DataFrameCheckSchema(pa.DataFrameModel):

    col1: Series[int] = pa.Field(gt=0, coerce=True)
    col2: Series[float] = pa.Field(gt=0, coerce=True)
    col3: Series[float] = pa.Field(lt=0, coerce=True)

    @pa.dataframe_check
    def product_is_negative(cls, df: pd.DataFrame) -> Series[bool]:
        return df["col1"] * df["col2"] * df["col3"] < 0

df = pd.DataFrame({
    "col1": [1, 2, 3],
    "col2": [5, 6, 7],
    "col3": [-1, -2, -3],
})

DataFrameCheckSchema.validate(df)
```

### Inheritance

The custom checks are inherited and therefore can be overwritten by the subclass.

```{code-cell} python
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Index, Series

class Parent(pa.DataFrameModel):

    a: Series[int] = pa.Field(coerce=True)

    @pa.check("a", name="foobar")
    def check_a(cls, a: Series[int]) -> Series[bool]:
        return a < 100


class Child(Parent):

    a: Series[int] = pa.Field(coerce=False)

    @pa.check("a", name="foobar")
    def check_a(cls, a: Series[int]) -> Series[bool]:
        return a > 100

is_a_coerce = Child.to_schema().columns["a"].coerce
print(f"coerce: {is_a_coerce}")
```

```{code-cell} python
df = pd.DataFrame({"a": [1, 2, 3]})

try:
    Child.validate(df)
except pa.errors.SchemaError as exc:
    print(exc)
```

(schema-model-alias)=

## Aliases

{class}`~pandera.api.pandas.model.DataFrameModel` supports columns which are not valid python variable names via the argument
`alias` of {class}`~pandera.api.dataframe.model_components.Field`.

Checks must reference the aliased names.

```{code-cell} python
import pandera.pandas as pa
import pandas as pd

class Schema(pa.DataFrameModel):
    col_2020: pa.typing.Series[int] = pa.Field(alias=2020)
    idx: pa.typing.Index[int] = pa.Field(alias="_idx", check_name=True)

    @pa.check(2020)
    def int_column_lt_100(cls, series):
        return series < 100


df = pd.DataFrame({2020: [99]}, index=[0])
df.index.name = "_idx"

print(Schema.validate(df))
```

*(New in 0.6.2)* The `alias` is respected when using the class attribute to get the underlying
`pd.DataFrame` column name or index level name.

```{code-cell} python
print(Schema.col_2020)
```

Very similar to the example above, you can also use the variable name directly within
the class scope, and it will respect the alias.

:::{note}
To access a variable from the class scope, you need to make it a class attribute,
and therefore assign it a default {class}`~pandera.api.dataframe.model_components.Field`.
:::

```{code-cell} python
import pandera.pandas as pa
import pandas as pd

class Schema(pa.DataFrameModel):
    a: pa.typing.Series[int] = pa.Field()
    col_2020: pa.typing.Series[int] = pa.Field(alias=2020)

    @pa.check(col_2020)
    def int_column_lt_100(cls, series):
        return series < 100

    @pa.check(a)
    def int_column_gt_100(cls, series):
        return series > 100


df = pd.DataFrame({2020: [99], "a": [101]})
print(Schema.validate(df))
```

## Manipulating DataFrame Models post-definition

One caveat of using inheritance to build schemas on top of each other is that there
is no clear way of how a child class can e.g. remove fields or update them without
completely overriding previous settings. This is because inheritance is strictly additive.

{class}`~pandera.api.pandas.container.DataFrameSchema` objects do have these options though, as described in
{ref}`dataframe-schema-transformations`, which you can leverage by overriding your
DataFrame Model's {func}`~pandera.api.pandas.model.DataFrameModel.to_schema` method.

DataFrame Models are for the most part just a proxy for the `DataFrameSchema` API; calling
{func}`~pandera.api.pandas.model.DataFrameModel.validate` will just redirect to the validate method of
the Data Frame Schema's {class}`~pandera.api.pandas.container.DataFrameSchema.validate` returned by
`to_schema`. As such, any updates to the schema that took place in there will propagate
cleanly.

As an example, the following class hierarchy can not remove the fields `b` and `c` from
`Baz` into a base-class without completely convoluting the inheritance tree. So, we can
get rid of them like this:

```{code-cell} python
import pandera.pandas as pa
import pandas as pd

class Foo(pa.DataFrameModel):
    a: pa.typing.Series[int]
    b: pa.typing.Series[int]

class Bar(pa.DataFrameModel):
    c: pa.typing.Series[int]
    d: pa.typing.Series[int]

class Baz(Foo, Bar):

    @classmethod
    def to_schema(cls) -> pa.DataFrameSchema:
        schema = super().to_schema()
        return schema.remove_columns(["b", "c"])

df = pd.DataFrame({"a": [99], "d": [101]})
print(Baz.validate(df))
```

:::{note}
There are drawbacks to manipulating schema shape in this way:

- Static code analysis has no way to figure out what fields have been removed/updated from
  the class definitions and inheritance hierarchy.
- Any children of classes which have overridden `to_schema` might experience
  surprising behavior -- if a child of `Baz` tries to define a field `b` or `c` again,
  it will lose it in its `to_schema` call because `Baz`'s `to_schema` will always
  be executed after any child's class body has already been fully assembled.
:::
