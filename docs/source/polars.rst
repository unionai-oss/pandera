.. currentmodule:: pandera.polars

.. _polars:

Data Validation with Polars
================================

*new in 0.19.0*

`Polars <https://docs.pola.rs/>`__ is a blazingly fast DataFrame library for
manipulating structured data. Since the core is written in Rust, you get the
performance of C/C++ while providing SDKs in other languages like Python.

Usage
-----

With the polars integration, you can define pandera schemas to validate polars
dataframes in Python. First, install ``pandera`` with the ``polars`` extra:

.. code:: bash

   pip install pandera[polars]

Then you can use pandera schemas to validate polars dataframes. In the example
below we'll use the :ref:`class-based API <dataframe_models>` to define a
:py:class:`~pandera.api.polars.model.DataFrameModel`, which we then use to
validate a :py:class:`polars.LazyFrame` object.

.. testcode:: polars

    import pandera.polars as pa
    import polars as pl


    class Schema(pa.DataFrameModel):
        state: str
        city: str
        price: int = pa.Field(in_range={"min_value": 5, "max_value": 20})


    lf = pl.LazyFrame(
        {
            'state': ['FL','FL','FL','CA','CA','CA'],
            'city': [
                'Orlando',
                'Miami',
                'Tampa',
                'San Francisco',
                'Los Angeles',
                'San Diego',
            ],
            'price': [8, 12, 10, 16, 20, 18],
        }
    )
    print(Schema.validate(lf).collect())


.. testoutput:: polars

    shape: (6, 3)
    ┌───────┬───────────────┬───────┐
    │ state ┆ city          ┆ price │
    │ ---   ┆ ---           ┆ ---   │
    │ str   ┆ str           ┆ i64   │
    ╞═══════╪═══════════════╪═══════╡
    │ FL    ┆ Orlando       ┆ 8     │
    │ FL    ┆ Miami         ┆ 12    │
    │ FL    ┆ Tampa         ┆ 10    │
    │ CA    ┆ San Francisco ┆ 16    │
    │ CA    ┆ Los Angeles   ┆ 20    │
    │ CA    ┆ San Diego     ┆ 18    │
    └───────┴───────────────┴───────┘


You can also use the :py:func:`~pandera.decorators.check_types` decorator to
validate polars LazyFrame function annotations at runtime:


.. testcode:: polars

    from pandera.typing.polars import LazyFrame

    @pa.check_types
    def function(lf: LazyFrame[Schema]) -> LazyFrame[Schema]:
        return lf.filter(pl.col("state").eq("CA"))

    print(function(lf).collect())


.. testoutput:: polars

    shape: (3, 3)
    ┌───────┬───────────────┬───────┐
    │ state ┆ city          ┆ price │
    │ ---   ┆ ---           ┆ ---   │
    │ str   ┆ str           ┆ i64   │
    ╞═══════╪═══════════════╪═══════╡
    │ CA    ┆ San Francisco ┆ 16    │
    │ CA    ┆ Los Angeles   ┆ 20    │
    │ CA    ┆ San Diego     ┆ 18    │
    └───────┴───────────────┴───────┘


And of course, you can use the object-based API to define a
:py:class:`~pandera.api.polars.container.DataFrameSchema`:


.. testcode:: polars

    schema = pa.DataFrameSchema({
        "state": pa.Column(str),
        "city": pa.Column(str),
        "price": pa.Column(int, pa.Check.in_range(min_value=5, max_value=20))
    })
    print(schema(lf).collect())


.. testoutput:: polars

    shape: (6, 3)
    ┌───────┬───────────────┬───────┐
    │ state ┆ city          ┆ price │
    │ ---   ┆ ---           ┆ ---   │
    │ str   ┆ str           ┆ i64   │
    ╞═══════╪═══════════════╪═══════╡
    │ FL    ┆ Orlando       ┆ 8     │
    │ FL    ┆ Miami         ┆ 12    │
    │ FL    ┆ Tampa         ┆ 10    │
    │ CA    ┆ San Francisco ┆ 16    │
    │ CA    ┆ Los Angeles   ┆ 20    │
    │ CA    ┆ San Diego     ┆ 18    │
    └───────┴───────────────┴───────┘

You can also validate :py:class:`polars.DataFrame` objects, which are objects that
execute computations eagerly. Under the hood, ``pandera`` will convert
the ``polars.DataFrame`` to a ``polars.LazyFrame`` before validating it:

.. testcode:: polars

    df = lf.collect()
    print(schema(df))

.. testoutput:: polars

    shape: (6, 3)
    ┌───────┬───────────────┬───────┐
    │ state ┆ city          ┆ price │
    │ ---   ┆ ---           ┆ ---   │
    │ str   ┆ str           ┆ i64   │
    ╞═══════╪═══════════════╪═══════╡
    │ FL    ┆ Orlando       ┆ 8     │
    │ FL    ┆ Miami         ┆ 12    │
    │ FL    ┆ Tampa         ┆ 10    │
    │ CA    ┆ San Francisco ┆ 16    │
    │ CA    ┆ Los Angeles   ┆ 20    │
    │ CA    ┆ San Diego     ┆ 18    │
    └───────┴───────────────┴───────┘

.. note::

    The :ref:`data synthesis strategies` functionality is not yet supported in
    the polars integration. At this time you can use the polars-native
    `parametric testing <https://docs.pola.rs/py-polars/html/reference/testing.html#parametric-testing>`__
    functions to generate test data for polars.

How it works
------------

Compared to the way ``pandera`` handles ``pandas`` dataframes, ``pandera``
attempts to leverage the ``polars`` `lazy API <https://docs.pola.rs/user-guide/lazy/using/>`__
as much as possible to leverage its performance optimization benefits. However,
because ``pandera`` is a run-time validator, it still needs to ``.collect()`` the
data values at certain points of the validation process that require operating
on the data values contained in the ``LazyFrame``. Therefore, calling the
``.validate()`` method on a ``LazyFrame`` will trigger multiple ``.collect()``
operations depending on the schema specification.

The ``schema.validate()`` method is effectively an eager operation that converts
the validated data back into a ``polars.LazyFrame`` before returning the output.
At a high level, this is what happens:

- **Apply parsers**: add missing columns if ``add_missing_columns=True``,
  coerce the datatypes if ``coerce=True``, filter columns if ``strict="filter"``,
  and set defaults if ``default=<value>``. This results in multiple ``.collect()``.
  operations.
- **Apply checks**: run all core, built-in, and custom checks on the data. Checks
  on metadata are done without ``.collect()`` operations, but checks that inspect
  data values do.
- **Convert to LazyFrame**: this allows for continuing a chain of lazy operations.

In the context of a lazy computation pipeline, this means that you can use schemas
as eager checkpoints that validate the data. Pandera is designed such that you
can continue to use the polars lazy API after the schema validation step.



.. tabbed:: DataFrameSchema

   .. testcode:: polars

       schema = pa.DataFrameSchema({"a": pa.Column(int)})

       df = (
           pl.LazyFrame({"a": [1.0, 2.0, 3.0]})
           .cast({"a": pl.Int64})
           .pipe(schema.validate) # this calls .collect() on the LazyFrame
                                  # and calls .lazy() before returning
                                  # the output
           .with_columns(b=pl.lit("a"))
           # do more lazy operations
           .collect()
       )
       print(df)

   .. testoutput:: polars

       shape: (3, 2)
       ┌─────┬─────┐
       │ a   ┆ b   │
       │ --- ┆ --- │
       │ i64 ┆ str │
       ╞═════╪═════╡
       │ 1   ┆ a   │
       │ 2   ┆ a   │
       │ 3   ┆ a   │
       └─────┴─────┘

.. tabbed:: DataFrameModel

   .. testcode:: polars

       class SimpleModel(pa.DataFrameModel):
           a: int

       df = (
           pl.LazyFrame({"a": [1.0, 2.0, 3.0]})
           .cast({"a": pl.Int64})
           .pipe(SimpleModel.validate) # this calls .collect() on the LazyFrame
                                       # and calls .lazy() before returning
                                       # the output
           .with_columns(b=pl.lit("a"))
           # do more lazy operations
           .collect()
       )
       print(df)

   .. testoutput:: polars

       shape: (3, 2)
       ┌─────┬─────┐
       │ a   ┆ b   │
       │ --- ┆ --- │
       │ i64 ┆ str │
       ╞═════╪═════╡
       │ 1   ┆ a   │
       │ 2   ┆ a   │
       │ 3   ┆ a   │
       └─────┴─────┘

In the event of a validation error, ``pandera`` will raise a :py:class:`~pandera.errors.SchemaError`
eagerly.

.. testcode:: polars

    invalid_lf = pl.LazyFrame({"a": pl.Series(["1", "2", "3"], dtype=pl.Utf8)})
    SimpleModel.validate(invalid_lf)

.. testoutput:: polars

    Traceback (most recent call last):
    ...
    SchemaError: expected column 'a' to have type Int64, got String

And if you use lazy validation, ``pandera`` will raise a :py:class:`~pandera.errors.SchemaErrors`
exception. This is particularly useful when you want to collect all of the validation errors
present in the data.

.. note::

    :ref:`Lazy validation <lazy_validation>` in pandera is different from the
    lazy API in polars, which is an unfortunate name collision. Lazy validation
    means that all parsers and checks are applied to the data before raising
    a :py:class:`~pandera.errors.SchemaErrors` exception. The lazy API
    in polars allows you to build a computation graph without actually
    executing it in-line, where you call ``.collect()`` to actually execute
    the computation.

.. testcode:: polars

    class ModelWithChecks(pa.DataFrameModel):
        a: int
        b: str = pa.Field(isin=[*"abc"])
        c: float = pa.Field(ge=0.0, le=1.0)

    invalid_lf = pl.LazyFrame({
        "a": pl.Series(["1", "2", "3"], dtype=pl.Utf8),
        "b": ["d", "e", "f"],
        "c": [0.0, 1.1, -0.1],
    })
    ModelWithChecks.validate(invalid_lf, lazy=True)

.. testoutput:: polars

    Traceback (most recent call last):
    ...
    pandera.errors.SchemaErrors: Schema 'ModelWithChecks': 4 errors types were found with a total of 6 failures.
    shape: (6, 6)
    ┌──────────────┬────────────────┬────────┬───────────────────────────────┬──────────────┬───────┐
    │ failure_case ┆ schema_context ┆ column ┆ check                         ┆ check_number ┆ index │
    │ ---          ┆ ---            ┆ ---    ┆ ---                           ┆ ---          ┆ ---   │
    │ str          ┆ str            ┆ str    ┆ str                           ┆ i32          ┆ i32   │
    ╞══════════════╪════════════════╪════════╪═══════════════════════════════╪══════════════╪═══════╡
    │ String       ┆ Column         ┆ a      ┆ dtype('Int64')                ┆ null         ┆ null  │
    │ d            ┆ Column         ┆ b      ┆ isin(['a', 'b', 'c'])         ┆ 0            ┆ 0     │
    │ e            ┆ Column         ┆ b      ┆ isin(['a', 'b', 'c'])         ┆ 0            ┆ 1     │
    │ f            ┆ Column         ┆ b      ┆ isin(['a', 'b', 'c'])         ┆ 0            ┆ 2     │
    │ -0.1         ┆ Column         ┆ c      ┆ greater_than_or_equal_to(0.0) ┆ 0            ┆ 2     │
    │ 1.1          ┆ Column         ┆ c      ┆ less_than_or_equal_to(1.0)    ┆ 1            ┆ 1     │
    └──────────────┴────────────────┴────────┴───────────────────────────────┴──────────────┴───────┘


Supported Data Types
--------------------

``pandera`` currently supports all of the
`polars data types <https://docs.pola.rs/py-polars/html/reference/datatypes.html>`__.
Built-in python types like ``str``, ``int``, ``float``, and ``bool`` will be
handled in the same way that ``polars`` handles them:

.. testcode:: polars

    assert pl.Series([1,2,3], dtype=int).dtype == pl.Int64
    assert pl.Series([*"abc"], dtype=str).dtype == pl.Utf8
    assert pl.Series([1.0, 2.0, 3.0], dtype=float).dtype == pl.Float64

So the following schemas are equivalent:

.. testcode:: polars

    schema1 = pa.DataFrameSchema({
        "a": pa.Column(int),
        "b": pa.Column(str),
        "c": pa.Column(float),
    })

    schema2 = pa.DataFrameSchema({
        "a": pa.Column(pl.Int64),
        "b": pa.Column(pl.Utf8),
        "c": pa.Column(pl.Float64),
    })

    assert schema1 == schema2

Nested Types
^^^^^^^^^^^^

Polars nested datetypes are also supported via :ref:`parameterized data types <parameterized dtypes>`.
See the examples below for the different ways to specify this through the
object-based and class-based APIs:

.. tabbed:: DataFrameSchema

   .. testcode:: polars

        schema = DataFrameSchema(
            {
                "list_col": Column(pl.List(pl.Int64())),
                "array_col": Column(pl.Array(pl.Int64(), 3)),
                "struct_col": Column(pl.Struct({"a": pl.Utf8(), "b": pl.Float64()})),
            },
        )

.. tabbed:: DataFrameModel (Annotated)

   .. testcode:: polars

        class ModelWithAnnotated(DataFrameModel):
            list_col: Annotated[pl.List, pl.Int64()]
            array_col: Annotated[pl.Array, pl.Int64(), 3]
            struct_col: Annotated[pl.Struct, {"a": pl.Utf8(), "b": pl.Float64()}]

.. tabbed:: DataFrameModel (Field)

   .. testcode:: polars

        class ModelWithDtypeKwargs(DataFrameModel):
            list_col: pl.List = pa.Field(dtype_kwargs={"inner": pl.Int64()})
            array_col: pl.Array = pa.Field(dtype_kwargs={"inner": pl.Int64(), "width": 3})
            struct_col: pl.Struct = pa.Field(dtype_kwargs={"fields": {"a": pl.Utf8(), "b": pl.Float64()}})


Custom checks
-------------

All of the built-in :py:class:`~pandera.api.checks.Check` methods are supported
in the polars integration.

To create custom checks, you can create functions that take a :py:class:`~pandera.api.polars.types.PolarsData`
named tuple as input and produces a ``polars.LazyFrame`` as output. :py:class:`~pandera.api.polars.types.PolarsData`
contains two attributes:

- A ``lazyframe`` attribute, which contains the ``polars.LazyFrame`` object you want
  to validate.
- A ``key`` attribute, which contains the column name you want to validate. This
  will be ``None`` for dataframe-level checks.

Element-wise checks are also supported by setting ``element_wise=True``. This
will require a function that takes in a single element of the column/dataframe
and returns a boolean scalar indicating whether the value passed.

.. warning::

    Under the hood, element-wise checks use the
    `map_elements <https://docs.pola.rs/py-polars/html/reference/expressions/api/polars.Expr.map_elements.html>`__
    function, which is slower than the native polars expressions API.

Column-level Checks
^^^^^^^^^^^^^^^^^^^

Here's an example of a column-level custom check:

.. tabbed:: DataFrameSchema

   .. testcode:: polars

       from pandera.polars import PolarsData


       def is_positive_vector(data: PolarsData) -> pl.LazyFrame:
           """Return a LazyFrame with a single boolean column."""
           return data.lazyframe.select(pl.col(data.key).gt(0))

       def is_positive_scalar(data: PolarsData) -> pl.LazyFrame:
           """Return a LazyFrame with a single boolean scalar."""
           return data.lazyframe.select(pl.col(data.key).gt(0).all())

       def is_positive_element_wise(x: int) -> bool:
           """Take a single value and return a boolean scalar."""
           return x > 0

       schema_with_custom_checks = pa.DataFrameSchema({
           "a": pa.Column(
               int,
               checks=[
                   pa.Check(is_positive_vector),
                   pa.Check(is_positive_scalar),
                   pa.Check(is_positive_element_wise, element_wise=True),
               ]
           )
       })

       lf = pl.LazyFrame({"a": [1, 2, 3]})
       validated_df = schema_with_custom_checks.validate(lf).collect()
       print(validated_df)

   .. testoutput:: polars

       shape: (3, 1)
       ┌─────┐
       │ a   │
       │ --- │
       │ i64 │
       ╞═════╡
       │ 1   │
       │ 2   │
       │ 3   │
       └─────┘

.. tabbed:: DataFrameModel

   .. testcode:: polars

       from pandera.polars import PolarsData


       class ModelWithCustomChecks(pa.DataFrameModel):
           a: int

           @pa.check("a")
           def is_positive_vector(cls, data: PolarsData) -> pl.LazyFrame:
               """Return a LazyFrame with a single boolean column."""
               return data.lazyframe.select(pl.col(data.key).gt(0))

           @pa.check("a")
           def is_positive_scalar(cls, data: PolarsData) -> pl.LazyFrame:
               """Return a LazyFrame with a single boolean scalar."""
               return data.lazyframe.select(pl.col(data.key).gt(0).all())

           @pa.check("a", element_wise=True)
           def is_positive_element_wise(cls, x: int) -> bool:
               """Take a single value and return a boolean scalar."""
               return x > 0

       validated_df = ModelWithCustomChecks.validate(lf).collect()
       print(validated_df)

   .. testoutput:: polars

       shape: (3, 1)
       ┌─────┐
       │ a   │
       │ --- │
       │ i64 │
       ╞═════╡
       │ 1   │
       │ 2   │
       │ 3   │
       └─────┘

For column-level checks, the custom check function should return a
``polars.LazyFrame`` containing a single boolean column or a single boolean scalar.


DataFrame-level Checks
^^^^^^^^^^^^^^^^^^^^^^

If you need to validate values on an entire dataframe, you can specify at check
at the dataframe level. The expected output is a ``polars.LazyFrame`` containing
multiple boolean columns, a single boolean column, or a scalar boolean.

.. tabbed:: DataFrameSchema

   .. testcode:: polars

      def col1_gt_col2(data: PolarsData, col1: str, col2: str) -> pl.LazyFrame:
          """Return a LazyFrame with a single boolean column."""
          return data.lazyframe.select(pl.col(col1).gt(pl.col(col2)))

      def is_positive_df(data: PolarsData) -> pl.LazyFrame:
          """Return a LazyFrame with multiple boolean columns."""
          return data.lazyframe.select(pl.col("*").gt(0))

      def is_positive_element_wise(x: int) -> bool:
           """Take a single value and return a boolean scalar."""
           return x > 0

      schema_with_df_checks = pa.DataFrameSchema(
          columns={
              "a": pa.Column(int),
              "b": pa.Column(int),
          },
          checks=[
              pa.Check(col1_gt_col2, col1="a", col2="b"),
              pa.Check(is_positive_df),
              pa.Check(is_positive_element_wise, element_wise=True),
          ]
      )

      lf = pl.LazyFrame({"a": [2, 3, 4], "b": [1, 2, 3]})
      validated_df = schema_with_df_checks.validate(lf).collect()
      print(validated_df)


   .. testoutput:: polars

      shape: (3, 2)
      ┌─────┬─────┐
      │ a   ┆ b   │
      │ --- ┆ --- │
      │ i64 ┆ i64 │
      ╞═════╪═════╡
      │ 2   ┆ 1   │
      │ 3   ┆ 2   │
      │ 4   ┆ 3   │
      └─────┴─────┘

.. tabbed:: DataFrameModel

   .. testcode:: polars

       class ModelWithDFChecks(pa.DataFrameModel):
           a: int
           b: int

           @pa.dataframe_check
           def cola_gt_colb(cls, data: PolarsData) -> pl.LazyFrame:
               """Return a LazyFrame with a single boolean column."""
               return data.lazyframe.select(pl.col("a").gt(pl.col("b")))

           @pa.dataframe_check
           def is_positive_df(cls, data: PolarsData) -> pl.LazyFrame:
               """Return a LazyFrame with multiple boolean columns."""
               return data.lazyframe.select(pl.col("*").gt(0))

           @pa.dataframe_check(element_wise=True)
           def is_positive_element_wise(cls, x: int) -> bool:
               """Take a single value and return a boolean scalar."""
               return x > 0

       validated_df = ModelWithDFChecks.validate(lf).collect()
       print(validated_df)

   .. testoutput:: polars

      shape: (3, 2)
      ┌─────┬─────┐
      │ a   ┆ b   │
      │ --- ┆ --- │
      │ i64 ┆ i64 │
      ╞═════╪═════╡
      │ 2   ┆ 1   │
      │ 3   ┆ 2   │
      │ 4   ┆ 3   │
      └─────┴─────┘
