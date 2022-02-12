.. pandera documentation for class-based API.

.. currentmodule:: pandera

.. _schema_models:

Schema Models
=============

*new in 0.5.0*

``pandera`` provides a class-based API that's heavily inspired by
`pydantic <https://pydantic-docs.helpmanual.io/>`_. In contrast to the
:ref:`object-based API<DataFrameSchemas>`, you can define schema models in
much the same way you'd define ``pydantic`` models.

`Schema Models` are annotated with the :mod:`pandera.typing` module using the standard
`typing <https://docs.python.org/3/library/typing.html>`_ syntax. Models can be
explictly converted to a :class:`~pandera.schemas.DataFrameSchema` or used to validate a
:class:`~pandas.DataFrame` directly.

.. note::

   Due to current limitations in the pandas library (see discussion
   `here <https://github.com/pandera-dev/pandera/issues/253#issuecomment-665338337>`_),
   ``pandera`` annotations are only used for **run-time** validation and **cannot** be
   leveraged by static-type checkers like `mypy <http://mypy-lang.org/>`_. See the
   discussion `here <https://github.com/pandera-dev/pandera/issues/253#issuecomment-665338337>`_
   for more details.


Basic Usage
-----------

.. testcode:: dataframe_schema_model

    import pandas as pd
    import pandera as pa
    from pandera.typing import Index, DataFrame, Series


    class InputSchema(pa.SchemaModel):
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
    transform(invalid_df)


.. testoutput:: dataframe_schema_model

    Traceback (most recent call last):
    ...
    pandera.errors.SchemaError: <Schema Column: 'year' type=DataType(int64)> failed element-wise validator 0:
    <Check greater_than: greater_than(2000)>
    failure cases:
       index  failure_case
    0      2          1999


As you can see in the example above, you can define a schema by sub-classing
:class:`~pandera.model.SchemaModel` and defining column/index fields as class attributes.
The :func:`~pandera.decorators.check_types` decorator is required to perform validation of the dataframe at
run-time.

Note that :class:`~pandera.model_components.Field` s apply to both
:class:`~pandera.schema_components.Column` and :class:`~pandera.schema_components.Index`
objects, exposing the built-in :class:`Check` s via key-word arguments.

*(New in 0.6.2)* When you access a class attribute defined on the schema,
it will return the name of the column used in the validated `pd.DataFrame`.
In the example above, this will simply be the string `"year"`.

.. testcode:: dataframe_schema_model

    print(f"Column name for 'year' is {InputSchema.year}\n")
    print(df.loc[:, [InputSchema.year, "day"]])

.. testoutput:: dataframe_schema_model

    Column name for 'year' is year

       year  day
    0  2001  200
    1  2002  156
    2  2003  365


Validate on Initialization
--------------------------

*new in 0.8.0*

Pandera provides an interface for validating dataframes on initialization.
This API uses the :py:class:`pandera.typing.pandas.DataFrame` generic type
to validated against the :py:class:`~pandera.model.SchemaModel` type variable
on initialization:

.. testcode:: validate_on_init

    import pandas as pd
    import pandera as pa

    from pandera.typing import DataFrame, Series


    class Schema(pa.SchemaModel):
        state: Series[str]
        city: Series[str]
        price: Series[int] = pa.Field(in_range={"min_value": 5, "max_value": 20})

    df = DataFrame[Schema](
        {
            'state': ['NY','FL','GA','CA'],
            'city': ['New York', 'Miami', 'Atlanta', 'San Francisco'],
            'price': [8, 12, 10, 16],
        }
    )
    print(df)


.. testoutput:: validate_on_init

      state           city  price
    0    NY       New York      8
    1    FL          Miami     12
    2    GA        Atlanta     10
    3    CA  San Francisco     16


Refer to :ref:`supported-dataframe-libraries` to see how this syntax applies
to other supported dataframe types.


Converting to DataFrameSchema
-----------------------------

You can easily convert a :class:`~pandera.model.SchemaModel` class into a
:class:`~pandera.schemas.DataFrameSchema`:

.. testcode:: dataframe_schema_model

    print(InputSchema.to_schema())

.. testoutput:: dataframe_schema_model

    <Schema DataFrameSchema(
        columns={
            'year': <Schema Column(name=year, type=DataType(int64))>
            'month': <Schema Column(name=month, type=DataType(int64))>
            'day': <Schema Column(name=day, type=DataType(int64))>
        },
        checks=[],
        coerce=False,
        dtype=None,
        index=None,
        strict=False
        name=None,
        ordered=False,
        unique_column_names=False
    )>

You can also use the :meth:`~pandera.model.SchemaModel.validate` method to
validate dataframes:

.. testcode:: dataframe_schema_model

    print(InputSchema.validate(df))

.. testoutput:: dataframe_schema_model

       year  month  day
    0  2001      3  200
    1  2002      6  156
    2  2003     12  365

Or you can use the :meth:`~pandera.model.SchemaModel` class directly to
validate dataframes, which is syntactic sugar that simply delegates to the
:meth:`~pandera.model.SchemaModel.validate` method.

.. testcode:: dataframe_schema_model

    print(InputSchema(df))

.. testoutput:: dataframe_schema_model

       year  month  day
    0  2001      3  200
    1  2002      6  156
    2  2003     12  365

Excluded attributes
-------------------

Class variables which begin with an underscore will be automatically excluded from
the model. :ref:`Config<schema_model_config>` is also a reserved name.
However, :ref:`aliases<schema_model_alias>` can be used to circumvent these limitations.


Supported dtypes
----------------

Any dtypes supported by ``pandera`` can be used as type parameters for
:class:`~pandera.typing.Series` and :class:`~pandera.typing.Index`. There are,
however, a couple of gotchas.

Dtype aliases
^^^^^^^^^^^^^

:mod:`pandera.typing` aliases will be deprecated in a future version,
please use :class:`~pandera.dtypes.DataType` subclasses instead.

.. code-block::

    import pandera as pa
    from pandera.typing import Series, String

    class Schema(pa.SchemaModel):
        a: Series[String]

Type Vs instance
^^^^^^^^^^^^^^^^

You must give a **type**, not an **instance**.

:green:`✔` Good:

.. testcode:: dataframe_schema_model
    :skipif: SKIP_PANDAS_LT_V1

    import pandas as pd

    class Schema(pa.SchemaModel):
        a: Series[pd.StringDtype]

:red:`✘` Bad:

.. testcode:: dataframe_schema_model
    :skipif: SKIP_SCHEMA_MODEL

    class Schema(pa.SchemaModel):
        a: Series[pd.StringDtype()]

.. testoutput:: dataframe_schema_model
    :skipif: SKIP_SCHEMA_MODEL

    Traceback (most recent call last):
    ...
    TypeError: Parameters to generic types must be types. Got StringDtype.

.. _parameterized dtypes:

Parametrized dtypes
^^^^^^^^^^^^^^^^^^^
Pandas supports a couple of parametrized dtypes. As of pandas 1.2.0:


+-------------------+---------------------------+-----------------------------+
| Kind of Data      | Data Type                 | Parameters                  |
+===================+===========================+=============================+
| tz-aware datetime | :class:`DatetimeTZDtype`  | ``unit``, ``tz``            |
+-------------------+---------------------------+-----------------------------+
| Categorical       | :class:`CategoricalDtype` | ``categories``, ``ordered`` |
+-------------------+---------------------------+-----------------------------+
| period            | :class:`PeriodDtype`      | ``freq``                    |
+-------------------+---------------------------+-----------------------------+
| sparse            | :class:`SparseDtype`      | ``dtype``, ``fill_value``   |
+-------------------+---------------------------+-----------------------------+
| intervals         | :class:`IntervalDtype`    | ``subtype``                 |
+-------------------+---------------------------+-----------------------------+

Annotated
"""""""""

Parameters can be given via :data:`typing.Annotated`. It requires python > 3.9 or
`typing_extensions <https://pypi.org/project/typing-extensions/>`_, which is already a
requirement of Pandera. Unfortunately :data:`typing.Annotated` has not been backported
to python 3.6.

:green:`✔` Good:

.. testcode:: dataframe_schema_model
    :skipif: PY36

    try:
        from typing import Annotated  # python 3.9+
    except ImportError:
        from typing_extensions import Annotated

    class Schema(pa.SchemaModel):
        col: Series[Annotated[pd.DatetimeTZDtype, "ns", "est"]]

Furthermore, you must pass all parameters in the order defined in the dtype's
constructor (see :ref:`table <parameterized dtypes>`).

:red:`✘` Bad:

.. testcode:: dataframe_schema_model
    :skipif: PY36

    class Schema(pa.SchemaModel):
        col: Series[Annotated[pd.DatetimeTZDtype, "utc"]]

    Schema.to_schema()

.. testoutput:: dataframe_schema_model
    :skipif: PY36

    Traceback (most recent call last):
    ...
    TypeError: Annotation 'DatetimeTZDtype' requires all positional arguments ['unit', 'tz'].

Field
"""""

:green:`✔` Good:

.. testcode:: dataframe_schema_model

    class SchemaFieldDatetimeTZDtype(pa.SchemaModel):
        col: Series[pd.DatetimeTZDtype] = pa.Field(dtype_kwargs={"unit": "ns", "tz": "EST"})

You cannot use both :data:`typing.Annotated` and ``dtype_kwargs``.

:red:`✘` Bad:

.. testcode:: dataframe_schema_model
    :skipif: PY36

    class SchemaFieldDatetimeTZDtype(pa.SchemaModel):
        col: Series[Annotated[pd.DatetimeTZDtype, "ns", "est"]] = pa.Field(dtype_kwargs={"unit": "ns", "tz": "EST"})

    Schema.to_schema()

.. testoutput:: dataframe_schema_model
    :skipif: PY36

    Traceback (most recent call last):
    ...
    TypeError: Cannot specify redundant 'dtype_kwargs' for pandera.typing.Series[typing_extensions.Annotated[pandas.core.dtypes.dtypes.DatetimeTZDtype, 'ns', 'est']].
    Usage Tip: Drop 'typing.Annotated'.

Required Columns
----------------

By default all columns specified in the schema are :ref:`required<required>`, meaning
that if a column is missing in the input DataFrame an exception will be
thrown. If you want to make a column optional, annotate it with :data:`typing.Optional`.

.. testcode:: dataframe_schema_model
    :skipif: PY36

    from typing import Optional

    import pandas as pd
    import pandera as pa
    from pandera.typing import Series


    class Schema(pa.SchemaModel):
        a: Series[str]
        b: Optional[Series[int]]


    df = pd.DataFrame({"a": ["2001", "2002", "2003"]})
    Schema.validate(df)


Schema Inheritance
------------------

You can also use inheritance to build schemas on top of a base schema.

.. testcode:: dataframe_schema_model

    class BaseSchema(pa.SchemaModel):
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

    print(transform(df))

.. testoutput:: dataframe_schema_model

       year  passengers
    1  2000       61000
    2  2001       50000
    3  2002       45000

.. _schema_model_config:

Config
------

Schema-wide options can be controlled via the ``Config`` class on the ``SchemaModel``
subclass. The full set of options can be found in the :class:`~pandera.model.BaseConfig`
class.

.. testcode:: dataframe_schema_model

    class Schema(pa.SchemaModel):

        year: Series[int] = pa.Field(gt=2000, coerce=True)
        month: Series[int] = pa.Field(ge=1, le=12, coerce=True)
        day: Series[int] = pa.Field(ge=0, le=365, coerce=True)

        class Config:
            name = "BaseSchema"
            strict = True
            coerce = True
            foo = "bar"  # Interpreted as dataframe check

It is not required for the ``Config`` to subclass :class:`~pandera.model.BaseConfig` but
it **must** be named '**Config**'.

See :ref:`class_based_api_dataframe_checks` for details on using registered dataframe checks.

MultiIndex
----------

The :class:`~pandera.schema_components.MultiIndex` capabilities are also supported with
the class-based API:

.. testcode:: dataframe_schema_model

    import pandera as pa
    from pandera.typing import Index, Series

    class MultiIndexSchema(pa.SchemaModel):

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

.. testoutput:: dataframe_schema_model

    <Schema MultiIndex(
        indexes=[
            <Schema Index(name=year, type=DataType(int64))>
            <Schema Index(name=month, type=DataType(int64))>
        ]
        coerce=True,
        strict=True,
        name=time,
        ordered=True
    )>

.. testcode:: dataframe_schema_model

    from pprint import pprint

    pprint({name: col.checks for name, col in index.columns.items()})

.. testoutput:: dataframe_schema_model

    {'month': [<Check greater_than_or_equal_to: greater_than_or_equal_to(1)>,
            <Check less_than_or_equal_to: less_than_or_equal_to(12)>],
    'year': [<Check greater_than: greater_than(2000)>]}

Multiple :class:`~pandera.typing.Index` annotations are automatically converted into a
:class:`~pandera.schema_components.MultiIndex`. MultiIndex options are given in the
:ref:`schema_model_config`.

.. _schema_model_custom_check:

Custom Checks
-------------

Unlike the object-based API, custom checks can be specified as class methods.

Column/Index checks
^^^^^^^^^^^^^^^^^^^

.. testcode:: dataframe_schema_model

    import pandera as pa
    from pandera.typing import Index, Series

    class CustomCheckSchema(pa.SchemaModel):

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

.. note::

    * You can supply the key-word arguments of the :class:`~pandera.checks.Check` class
      initializer to get the flexibility of :ref:`groupby checks <column_check_groups>`
    * Similarly to ``pydantic``, :func:`classmethod` decorator is added behind the scenes
      if omitted.
    * You still may need to add the ``@classmethod`` decorator *after* the
      :func:`~pandera.model_components.check` decorator if your static-type checker or
      linter complains.
    * Since ``checks`` are class methods, the first argument value they receive is a
      SchemaModel subclass, not an instance of a model.

.. testcode:: dataframe_schema_model

    from typing import Dict

    class GroupbyCheckSchema(pa.SchemaModel):

        value: Series[int] = pa.Field(gt=0, coerce=True)
        group: Series[str] = pa.Field(isin=["A", "B"])

        @pa.check("value", groupby="group", regex=True, name="check_means")
        def check_groupby(cls, grouped_value: Dict[str, Series[int]]) -> bool:
            return grouped_value["A"].mean() < grouped_value["B"].mean()

    df = pd.DataFrame({
        "value": [100, 110, 120, 10, 11, 12],
        "group": list("AAABBB"),
    })

    print(GroupbyCheckSchema.validate(df))

.. testoutput:: dataframe_schema_model

    Traceback (most recent call last):
    ...
    pandera.errors.SchemaError: <Schema Column: 'value' type=DataType(int64)> failed series validator 1:
    <Check check_means>

.. _schema_model_dataframe_check:

DataFrame Checks
^^^^^^^^^^^^^^^^

You can also define dataframe-level checks, similar to the
:ref:`object-based API <wide_checks>`, using the
:func:`~pandera.schema_components.dataframe_check` decorator:

.. testcode:: dataframe_schema_model

    import pandas as pd
    import pandera as pa
    from pandera.typing import Index, Series

    class DataFrameCheckSchema(pa.SchemaModel):

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

Inheritance
^^^^^^^^^^^

The custom checks are inherited and therefore can be overwritten by the subclass.

.. testcode:: dataframe_schema_model

    import pandas as pd
    import pandera as pa
    from pandera.typing import Index, Series

    class Parent(pa.SchemaModel):

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

.. testoutput:: dataframe_schema_model

    coerce: False

.. testcode:: dataframe_schema_model

    df = pd.DataFrame({"a": [1, 2, 3]})
    print(Child.validate(df))

.. testoutput:: dataframe_schema_model

    Traceback (most recent call last):
    ...
    pandera.errors.SchemaError: <Schema Column: 'a' type=DataType(int64)> failed element-wise validator 0:
    <Check foobar>
    failure cases:
        index  failure_case
    0      0             1
    1      1             2
    2      2             3

.. _schema_model_alias:

Aliases
-------

:class:`~pandera.model.SchemaModel` supports columns which are not valid python variable names via the argument
`alias` of :class:`~pandera.model_components.Field`.

Checks must reference the aliased names.

.. testcode:: dataframe_schema_model

    import pandera as pa
    import pandas as pd

    class Schema(pa.SchemaModel):
        col_2020: pa.typing.Series[int] = pa.Field(alias=2020)
        idx: pa.typing.Index[int] = pa.Field(alias="_idx", check_name=True)

        @pa.check(2020)
        def int_column_lt_100(cls, series):
            return series < 100


    df = pd.DataFrame({2020: [99]}, index=[0])
    df.index.name = "_idx"

    print(Schema.validate(df))

.. testoutput:: dataframe_schema_model

          2020
    _idx
    0       99


*(New in 0.6.2)* The `alias` is respected when using the class attribute to get the underlying
`pd.DataFrame` column name or index level name.

.. testcode:: dataframe_schema_model

    print(Schema.col_2020)

.. testoutput:: dataframe_schema_model

    2020


Very similar to the example above, you can also use the variable name directly within
the class scope, and it will respect the alias.

.. note::

    To access a variable from the class scope, you need to make it a class attribute,
    and therefore assign it a default :class:`~pandera.model_components.Field`.

.. testcode:: dataframe_schema_model

    import pandera as pa
    import pandas as pd

    class Schema(pa.SchemaModel):
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

.. testoutput:: dataframe_schema_model

          2020    a
    0       99  101
