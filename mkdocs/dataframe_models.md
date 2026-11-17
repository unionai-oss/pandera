# DataFrame Models

*new in 0.5.0*


`pandera` provides a class-based API that's heavily inspired by
[pydantic](https://pydantic-docs.helpmanual.io/). In contrast to the
[object-based API](./dataframe_schemas.html), you can define dataframe models in
much the same way you'd define `pydantic` models.

`DataFrameModel` s are annotated with the `pandera.typing` module using the standard
[typing](https://docs.python.org/3/library/typing.html) syntax. Models can be
explicitly converted to a [DataFrameSchema](./reference/generated/pandera.api.pandas.container.DataFrameSchema.html) or used to validate a
`pandas.DataFrame` directly.

!!! note

    Due to current limitations in the pandas library (see discussion
    [here](https://github.com/pandera-dev/pandera/issues/253#issuecomment-665338337)),
    `pandera` annotations are primarily used for **run-time** validation and have
    limited support for static-type checkers like [mypy](http://mypy-lang.org/).
    The typing-only [FieldType](./reference/generated/pandera.typing.FieldType.html) descriptor also models
    class-level column-name access for generic checkers such as `ty`.
    See the [Mypy Integration](./mypy_integration.html) for more details.

## Basic Usage

```python
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
[DataFrameModel](./reference/generated/pandera.api.pandas.model.DataFrameModel.html) and defining column/index fields as class attributes.
The [check_types](./reference/generated/pandera.decorators.check_types.html) decorator is required to perform validation of the dataframe at
run-time.

The [Field](./reference/generated/pandera.api.dataframe.model_components.Field.html) class is used to define the schema
specification for a column or index.

Note that [Field](./reference/generated/pandera.api.dataframe.model_components.Field.html) s apply to both
[Column](./reference/generated/pandera.api.pandas.components.Column.html) and [Index](./reference/generated/pandera.api.pandas.components.Index.html)
objects, exposing the built-in [Check](./reference/generated/pandera.api.checks.Check.html) s via key-word arguments.

*(New in 0.6.2)* When you access a class attribute defined on the schema,
it will return the name of the column used in the validated `pd.DataFrame`.
In the example above, this will simply be the string `"year"`.

```python
print(f"Column name for 'year' is {InputSchema.year}\n")
print(df.loc[:, [InputSchema.year, "day"]])
```

## Using Data Types directly for Column Type Annotations

*new in 0.15.0*

For conciseness, you can also use type annotations for columns without using
the [Series](./reference/generated/pandera.typing.Series.html) generic. This class attributes will be
interpreted as [Column](./reference/generated/pandera.api.pandas.components.Column.html) objects
under the hood.

```python
class InputSchema(pa.DataFrameModel):
    year: int = pa.Field(gt=2000, coerce=True)
    month: int = pa.Field(ge=1, le=12, coerce=True)
    day: int = pa.Field(ge=0, le=365, coerce=True)
```

## Reusing Field objects

To define reusable `Field` definitions, you need to use `functools.partial`.
This makes sure that each field attribute is bound to a unique `Field` instance.

```python
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
This API uses the [DataFrame](./reference/generated/pandera.typing.DataFrame.html) generic type
to validated against the [DataFrameModel](./reference/generated/pandera.api.pandas.model.DataFrameModel.html) type variable
on initialization:

```python
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

Refer to [Supported DataFrame Libraries](./supported_libraries.html) to see how this syntax applies
to other supported dataframe types.

## GeoPandas `GeoDataFrameModel`

For `geopandas.GeoDataFrame` workflows, use ``import pandera.geopandas as pg``
(the module includes the full `pandera.pandas` API) and subclass
[GeoDataFrameModel](./reference/generated/pandera.geopandas.GeoDataFrameModel.html) instead of
[DataFrameModel](./reference/generated/pandera.api.pandas.model.DataFrameModel.html) when you need
[validate](./reference/generated/pandera.geopandas.GeoDataFrameModel.html) (and
[example](./reference/generated/pandera.geopandas.GeoDataFrameModel.html),
[empty](./reference/generated/pandera.geopandas.GeoDataFrameModel.html)) to return a
`GeoDataFrame`, preserving active geometry and CRS metadata after validation.
For the object-based API, use [GeoDataFrameSchema](./reference/generated/pandera.geopandas.GeoDataFrameSchema.html).
Field definitions, checks, parsers, and `Config` work the same as for
`DataFrameModel`; see [GeoPandas](./geopandas.html) and the [GeoPandas API reference](./reference/geopandas.html).

## Converting to DataFrameSchema

You can easily convert a [DataFrameModel](./reference/generated/pandera.api.pandas.model.DataFrameModel.html) class into a
[DataFrameSchema](./reference/generated/pandera.api.pandas.container.DataFrameSchema.html):

```python
print(InputSchema.to_schema())
```

You can also use the [validate](./reference/generated/pandera.api.pandas.model.DataFrameModel.html) method to
validate dataframes:

```python
print(InputSchema.validate(df))
```

Or you can use the [DataFrameModel](./reference/generated/pandera.api.pandas.model.DataFrameModel.html) class directly to
validate dataframes, which is syntactic sugar that simply delegates to the
[validate](./reference/generated/pandera.api.pandas.model.DataFrameModel.html) method.

```python
print(InputSchema(df))
```

## Validate Against Multiple Schemas

*new in 0.14.0*

The built-in `typing.Union` type is supported for multiple `DataFrame` schemas.

```python
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

```python
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
the model. [Config](#config) is also a reserved name.
However, [aliases](#aliases) can be used to circumvent these limitations.

## Supported dtypes

Any dtypes supported by `pandera` can be used as type parameters for
[Series](./reference/generated/pandera.typing.Series.html) and [Index](./reference/generated/pandera.typing.Index.html). There are,
however, a couple of gotchas.

!!! important

    You can learn more about how data type validation works
    in [Data Type Validation](./dtype_validation.html).

### Dtype aliases

```python
import pandera.pandas as pa
from pandera.typing import Series, String

class Schema(pa.DataFrameModel):
    a: Series[String]
```

### Type Vs instance

You must give a **type**, not an **instance**.

✅ Good:

```python
import pandas as pd

class Schema(pa.DataFrameModel):
    a: Series[pd.StringDtype]
```

❌ Bad:

!!! note

    This is only applicable for pandas versions \< 2.0.0. In pandas > 2.0.0,
    pd.StringDtype() will produce a type.

```python
class Schema(pa.DataFrameModel):
    a: Series[pd.StringDtype()]
```


### Parametrized dtypes

Pandas supports a couple of parametrized dtypes. As of pandas 1.2.0:

| Kind of Data      | Data Type                 | Parameters              |
| ----------------- | ------------------------- | ----------------------- |
| tz-aware datetime | `DatetimeTZDtype`         | `unit`, `tz`            |
| Categorical       | `CategoricalDtype`        | `categories`, `ordered` |
| period            | `PeriodDtype`             | `freq`                  |
| sparse            | `SparseDtype`             | `dtype`, `fill_value`   |
| intervals         | `IntervalDtype`           | `subtype`               |

#### Annotated

Parameters can be given via `typing.Annotated`. It requires python >= 3.9 or
[typing_extensions](https://pypi.org/project/typing-extensions/), which is already a
requirement of Pandera. Unfortunately `typing.Annotated` has not been backported
to python 3.6.

✅ Good:

```python
try:
    from typing import Annotated  # python 3.9+
except ImportError:
    from typing_extensions import Annotated

class Schema(pa.DataFrameModel):
    col: Series[Annotated[pd.DatetimeTZDtype, "ns", "est"]]
```

Furthermore, you must pass all parameters in the order defined in the dtype's
constructor (see [table](#parametrized-dtypes)).

❌ Bad:

```python
class Schema(pa.DataFrameModel):
    col: Series[Annotated[pd.DatetimeTZDtype, "utc"]]

Schema.to_schema()
```

#### Field

✅ Good:

```python
class SchemaFieldDatetimeTZDtype(pa.DataFrameModel):
    col: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"unit": "ns", "tz": "EST"}
    )
```

You cannot use both `typing.Annotated` and `dtype_kwargs`.

❌ Bad:

```python
class SchemaFieldDatetimeTZDtype(pa.DataFrameModel):
    col: Series[Annotated[pd.DatetimeTZDtype, "ns", "est"]] = pa.Field(
        dtype_kwargs={"unit": "ns", "tz": "EST"}
    )

Schema.to_schema()
```

### Field metadata and static field names

Use the typing-only [FieldType](./reference/generated/pandera.typing.FieldType.html) descriptor when a
generic checker such as `ty` needs to accept a model field as a column name.
The backend-specific `pa.Field(...)` object can be supplied as additional
`FieldType` metadata:

```python
import pandas as pd
import pandera.pandas as pa
from pandera.typing import FieldType


class Schema(pa.DataFrameModel):
    name: FieldType[str, pa.Field(description="Name of the person")]
    age: FieldType[int, pa.Field(ge=0, description="Age of the person")]
    month: FieldType[int, pa.Field(ge=1, le=12)]
    identifier: FieldType[int, pa.Field(unique=True, title="Identifier")]


schema = Schema.to_schema()
schema.columns["name"].description
```

For parameterized dtypes, use the checker-friendly assignment form to pass
`dtype_kwargs`:

```python
class SchemaWithDtypeParamsAndField(pa.DataFrameModel):
    ts: FieldType[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"unit": "ns", "tz": "EST"},
        description="Timestamp",
    )
```

The assignment form is the checker-friendly spelling: it keeps the value
expression out of the type argument while preserving metadata precedence:

The `FieldType[T, pa.Field(...)]` form is also accepted by Pandera's runtime
parser. Generic checkers may reject the function call inside the type argument,
so use assignment when running `ty`.

```python
class SchemaExplicitWins(pa.DataFrameModel):
    value: FieldType[int] = pa.Field(description="from assignment")


SchemaExplicitWins.to_schema().columns["value"].description
```

You can also embed field metadata with `typing.Annotated`:

```python
from typing import Annotated


class AnnotatedSchema(pa.DataFrameModel):
    value: Annotated[int, pa.Field(gt=0, description="A positive value")]


AnnotatedSchema.to_schema().columns["value"].description
```

This syntax is valid at runtime and may be convenient when you prefer standard
typing constructs. Its trade-off is that `Annotated` preserves the underlying
`int` annotation for static analysis; it does not provide the class-level `str`
descriptor contract. If model fields are not passed to string-taking APIs, this
limitation may not matter. If `ty` is part of your tooling, use `FieldType`
instead.

`FieldType[T]` supplies the static descriptor contract: class-level access is
a `str`, while `T` remains the dtype Pandera parses at runtime. It is not the
runtime metadata function and must not be instantiated. Existing bare dtypes,
`Series[T]`, and `Annotated` forms used for dtype parameters remain supported.

## Required columns

By default all columns specified in the schema are [required](./dataframe_schemas.html#required-columns)
and non-nullable, meaning that a missing column or a null value raises an
exception. An outer `| None` on the annotation makes the column itself
optional, and `None` *inside* `FieldType[...]` allows null values.

```python
import pandas as pd
import pandera.pandas as pa
from pandera.typing import FieldType


class Schema(pa.DataFrameModel):
    required: str
    nullable: FieldType[int | None]
    optional: float | None
    optional_checked: FieldType[float] | None

df = pd.DataFrame(
    {
        "required": ["2001", "2002", "2003"],
        "nullable": pd.Series([1, None, 3], dtype="Int64"),
    }
)
Schema.validate(df)
```

The meanings are:

| Annotation | Column required? | Values nullable? |
| ---------- | ---------------- | ---------------- |
| `T` | Yes | No |
| `FieldType[T \| None]` | Yes | Yes |
| `T \| None` | No | No |
| `FieldType[T] \| None` | No | No |

The same meanings apply to `FieldType[T, pa.Field(...)]`. An explicit
`required` or `nullable` argument in `pa.Field` overrides the inferred value
without changing the other dimension. Explicit assignment also takes
precedence over metadata supplied in `FieldType`.

An explicit `pa.Field(nullable=...)` setting takes precedence over the
nullability inferred from the annotation. Existing
`Optional[Series[T]]` and bare `Optional[T]` annotations remain supported for
optional column presence. An optional field does not get added during
validation. To add missing fields, use
[add_missing_columns=True](./dataframe_schemas.html#adding-missing-columns) in the model
`Config` with required fields that specify a `default` value or
`nullable=True`.

## Schema Inheritance

You can also use inheritance to build schemas on top of a base schema.

```python
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

### Multiple inheritance

Multiple inheritance is also supported, making it easy to compose schemas from
reusable "mixin" models without repeating yourself. For this to work, each of
the base classes **must** inherit from
[DataFrameModel](./reference/generated/pandera.api.pandas.model.DataFrameModel.html), otherwise their fields are not
collected:

```python
class A(pa.DataFrameModel):
    a: Series[int]

class B(pa.DataFrameModel):
    b: Series[int]

class C(A, B):
    c: Series[int]

C.to_schema()
```

If a base class does not inherit from `DataFrameModel`, its fields are not
recognized and a `KeyError` is raised when calling
[to_schema](./reference/generated/pandera.api.pandas.model.DataFrameModel.html):

```python
class A:  # not a DataFrameModel!
    a: Series[int]

class C(pa.DataFrameModel, A):
    c: Series[int]

C.to_schema()  # raises KeyError: 'a'
```

## Config

Schema-wide options can be controlled via the `Config` class on the `DataFrameModel`
subclass. The full set of options can be found in the [BaseConfig](./reference/generated/pandera.api.pandas.model_config.BaseConfig.html)
class.

```python
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
[BaseConfig](./reference/generated/pandera.api.pandas.model_config.BaseConfig.html) but
it **must** be named '**Config**'.

See [Registered Custom Checks with the Class-based API](./extensions.html#class-based-api-dataframe-checks) for details on using registered dataframe checks.

## MultiIndex

The [MultiIndex](./reference/generated/pandera.api.pandas.components.MultiIndex.html) capabilities are also supported with
the class-based API:

```python
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

```python
from pprint import pprint

pprint({name: col.checks for name, col in index.columns.items()})
```

Multiple [Index](./reference/generated/pandera.typing.Index.html) annotations are automatically converted into a
[MultiIndex](./reference/generated/pandera.api.pandas.components.MultiIndex.html). MultiIndex options are given in the
[Config](#config).

## Index Name

Use `check_name` to validate the index name of a single-index dataframe:

```python
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

## Custom Checks

Unlike the object-based API, custom checks can be specified as class methods.

### Column/Index checks

```python
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

!!! note

    - You can supply the key-word arguments of the [Check](./reference/generated/pandera.api.checks.Check.html) class
      initializer to get the flexibility of [groupby checks](./checks.html#column-check-groups)
    - Similarly to `pydantic`, `classmethod` decorator is added behind the scenes
      if omitted.
    - You still may need to add the `@classmethod` decorator *after* the
      [check](./reference/generated/pandera.api.dataframe.model_components.check.html) decorator if your static-type checker or
      linter complains.
    - Since `checks` are class methods, the first argument value they receive is a
      DataFrameModel subclass, not an instance of a model.

```python
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

### DataFrame Checks

You can also define dataframe-level checks, similar to the
[object-based API](./checks.html#wide-checks), using the
[dataframe_check](./reference/generated/pandera.api.dataframe.model_components.dataframe_check.html) decorator:

```python
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

```python
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

```python
df = pd.DataFrame({"a": [1, 2, 3]})

try:
    Child.validate(df)
except pa.errors.SchemaError as exc:
    print(exc)
```

## Aliases

[DataFrameModel](./reference/generated/pandera.api.pandas.model.DataFrameModel.html) supports columns which are not valid python variable names via the argument
`alias` of [Field](./reference/generated/pandera.api.dataframe.model_components.Field.html).

Checks must reference the aliased names.

```python
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

```python
print(Schema.col_2020)
```

Very similar to the example above, you can also use the variable name directly within
the class scope, and it will respect the alias.

!!! note

    To access a variable from the class scope, you need to make it a class attribute,
    and therefore assign it a default [Field](./reference/generated/pandera.api.dataframe.model_components.Field.html).

```python
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

[DataFrameSchema](./reference/generated/pandera.api.pandas.container.DataFrameSchema.html) objects do have these options though, as described in
[DataFrameSchema Transformations](./dataframe_schemas.html#dataframeschema-transformations), which you can leverage by overriding your
DataFrame Model's [to_schema](./reference/generated/pandera.api.pandas.model.DataFrameModel.html) method.

DataFrame Models are for the most part just a proxy for the `DataFrameSchema` API; calling
[validate](./reference/generated/pandera.api.pandas.model.DataFrameModel.html) will just redirect to the validate method of
the Data Frame Schema's [validate](./reference/generated/pandera.api.pandas.container.DataFrameSchema.html) returned by
`to_schema`. As such, any updates to the schema that took place in there will propagate
cleanly.

As an example, the following class hierarchy can not remove the fields `b` and `c` from
`Baz` into a base-class without completely convoluting the inheritance tree. So, we can
get rid of them like this:

```python
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

!!! note

    There are drawbacks to manipulating schema shape in this way:

    - Static code analysis has no way to figure out what fields have been removed/updated from
      the class definitions and inheritance hierarchy.
    - Any children of classes which have overridden `to_schema` might experience
      surprising behavior -- if a child of `Baz` tries to define a field `b` or `c` again,
      it will lose it in its `to_schema` call because `Baz`'s `to_schema` will always
      be executed after any child's class body has already been fully assembled.
