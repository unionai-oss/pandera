.. pandera documentation for class-based API.

.. currentmodule:: pandera

.. _schema_models:

Schema Models (new)
===================

*new in 0.5.0*

``pandera`` provides a class-based API that's heavily inspired by
`pydantic <https://pydantic-docs.helpmanual.io/>`_. In contrast to the
:ref:`object-based API<DataFrameSchemas>`, you can define schema models in
much the same way you define ``pydantic`` models. You can define schema
models that can be used with the ``pandera.typing`` module to annotate
functions via the `typing <https://docs.python.org/3/library/typing.html>`_
syntax.

.. note::

   Due to current limitations in the pandas library (see discussion
   `here <https://github.com/pandera-dev/pandera/issues/253#issuecomment-665338337>`_),
   type annotations that use ``pandera`` schema models are only used for
   **run-time** validation and **cannot** be statically analyzed with packages
   like `mypy <http://mypy-lang.org/>`_. See the discussion
   `here <https://github.com/pandera-dev/pandera/issues/253#issuecomment-665338337>`_
   for more details.


Basic Usage
-----------

.. testcode:: dataframe_schema_model

    import pandas as pd
    import pandera as pa
    from pandera.typing import Index, DataFrame, Series


    class Schema(pa.SchemaModel):

        year: Series[int] = pa.Field(gt=2000, coerce=True)
        month: Series[int] = pa.Field(ge=1, le=12, coerce=True)
        day: Series[int] = pa.Field(ge=0, le=365, coerce=True)


    @pa.check_types
    def transform(df: DataFrame[Schema]):
        ...  # transform dataframe
        return df


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
:py:class:`SchemaModel` and defining column/index fields as class attributes.
The :py:func:`check_types` decorator is required to perform validation
of the dataframe at run-time.

Note that :py:class:`Field` s apply to both :py:class:`Column` and
:py:class:`Index` objects, exposing the built-in :py:class:`Check` s via
key-word arguments.

Converting to DataFrameSchema
-----------------------------

You can easily convert a :py:class:`SchemaModel` class into a
:py:class:`DataFrameSchema`:

.. testcode:: dataframe_schema_model

    print(Schema.to_schema())

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

Or use the :py:func:`SchemaModel.validate` method to validate dataframes:

.. testcode:: dataframe_schema_model

    print(Schema.validate(df))

.. testoutput:: dataframe_schema_model

       year  month  day
    0  2001      3  200
    1  2002      6  156
    2  2003     12  365


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


Config
------

The ``Config`` class can be specified within a ``SchemaModel`` subclass
definition, where you can set schema-wide options. The full set of options
can be found in the :class:`~pandera.model.BaseConfig` class.

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


MultiIndex
----------

The MultiIndex capabilities are also supported with the class-based API:

.. testcode:: dataframe_schema_model

    import pandera as pa
    from pandera.typing import Index, Series
    
    class MultiIndexSchema(pa.SchemaModel):
        
        year: Index[int]
        month: Index[int]
        passengers: Series[int]
    
        class Config:
            # provide multi index options in the config
            multiindex_name = "time"
            multiindex_strict = True
            multiindex_coerce = True
            
    print(MultiIndexSchema.to_schema().index)

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

.. _schema_model_custom_check:

Custom Checks
-------------

Unlike the object-based API, custom checks can be specified as class methods,
where the :py:func:`check` decorator automatically converts the method into
a ``classmethod``.

.. testcode:: dataframe_schema_model

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

Note the you can supply the key-word arguments of the :py:class:`Check` class
initializer to get the the flexibility of :ref:`groupby checks <column_check_groups>`

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
:ref:`object-based API <wide_checks>`, using the :py:func:`dataframe_check`
decorator:

.. testcode:: dataframe_schema_model

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
