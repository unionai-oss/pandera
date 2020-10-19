.. pandera documentation for class-based API.

.. currentmodule:: pandera

.. _schema_models:

Schema Models (new)
===================

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
    pandera.errors.SchemaError: <Schema Column: 'year' type=<class 'int'>> failed element-wise validator 0:
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

Converting to DataFrameSchema
-----------------------------

You can easily convert a :class:`~pandera.model.SchemaModel` class into a 
:class:`~pandera.schemas.DataFrameSchema`:

.. testcode:: dataframe_schema_model

    print(InputSchema.to_schema())

.. testoutput:: dataframe_schema_model

    DataFrameSchema(
        columns={
            "year": "<Schema Column: 'year' type=<class 'int'>>",
            "month": "<Schema Column: 'month' type=<class 'int'>>",
            "day": "<Schema Column: 'day' type=<class 'int'>>"
        },
        checks=[],
        index=None,
        transformer=None,
        coerce=False,
        strict=False
    )

Or use the :meth:`~pandera.model.SchemaModel.validate` method to validate dataframes:

.. testcode:: dataframe_schema_model

    print(InputSchema.validate(df))

.. testoutput:: dataframe_schema_model

       year  month  day
    0  2001      3  200
    1  2002      6  156
    2  2003     12  365

Supported dtypes
----------------

Any dtypes supported by ``pandera`` can be used as type parameters for 
:class:`pandera.typing.Series` and :class:`pandera.typing.Index`.

There are, however, a couple of gotchas:

* The enumeration :class:`pandera.dtypes.PandasDtype` is not directly supported because 
  the type parameter of a :class:`~typing.Generic` cannot be an enumeration [#dtypes]_. 
  Instead, you can use the :mod:`pandera.typing` counterparts:
  :data:`pandera.typing.Category`, :data:`pandera.typing.Float32`, ...

:green:`✔` Good: 

.. code-block::
    
    import pandera as pa
    from pandera.typing import Series, String

    class Schema(pa.SchemaModel):
        a: Series[String]

:red:`✘` Bad:

.. testcode:: dataframe_schema_model
    :skipif: PY36  

    class Schema(pa.SchemaModel):
        a: Series[pa.PandasDtype.String]

.. testoutput:: dataframe_schema_model
    :skipif: PY36  
    
    Traceback (most recent call last):
    ...
    AttributeError: type object 'Generic' has no attribute 'value'

* You must give a **type**, not an **instance**.

:green:`✔` Good:

.. code-block::
    
    import pandas as pd

    class Schema(pa.SchemaModel):
        a: Series[pd.StringDtype]

:red:`✘` Bad:

.. testcode:: dataframe_schema_model
    
    class Schema(pa.SchemaModel):
        a: Series[pd.StringDtype()]

.. testoutput:: dataframe_schema_model

    Traceback (most recent call last):
    ...
    TypeError: Parameters to generic types must be types. Got StringDtype.

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
            foo = "bar"  # not a valid option, ignored

It is not required for the ``Config`` to subclass :class:`~pandera.model.BaseConfig` but
it **must** be named '**Config**'.

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

    MultiIndex(
        columns={
            "year": "<Schema Column: 'year' type=<class 'int'>>",
            "month": "<Schema Column: 'month' type=<class 'int'>>"
        },
        checks=[],
        index=None,
        transformer=None,
        coerce=True,
        strict=True
    )

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

Note that:

* You can supply the key-word arguments of the :class:`~pandera.checks.Check` class
  initializer to get the flexibility of :ref:`groupby checks <column_check_groups>`
* Similarly to ``pydantic``, :func:`classmethod` decorator is added behind the scenes 
  if omitted.
* You still may need to add the `@classmethod` decorator **after** the 
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
    pandera.errors.SchemaError: <Schema Column: 'value' type=<class 'int'>> failed series validator 1:
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

    class Base(pa.SchemaModel):

        a: Series[int] = pa.Field(coerce=True)

        @pa.check("a", name="foobar") 
        def check_a(cls, a: Series[int]) -> Series[bool]:
            return a < 100


    class Child(pa.SchemaModel):

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
    pandera.errors.SchemaError: <Schema Column: 'a' type=<class 'int'>> failed element-wise validator 0:
    <Check foobar>
    failure cases:
        index  failure_case
    0      0             1
    1      1             2
    2      2             3


Footnotes
---------

.. [#dtypes] It is actually possible to use a PandasDtype by encasing it in a 
    :class:`typing.Literal` like ``Series[Literal[PandasDtype.Category]]``. 
    :mod:`pandera.typing` defines aliases to reduce boilerplate.