.. currentmodule:: pandera.polars

.. _polars:

Data Validation with Polars
================================

*new in 0.19.0*

`Polars <https://docs.pola.rs/>`__ is a blazingly fast DataFrame library for
manipulating structured data. Since the core is written in Rust, you get the
performance of C/C++ with SDKs available for Python, R, and NodeJS.

With the polars integration, you can define pandera schemas to validate polars
dataframes in Python. First, install ``pandera`` with the ``polars`` extra:

.. code:: bash

   pip install pandera[polars]

Then you can use pandera schemas to validate modin dataframes. In the example
below we'll use the :ref:`class-based API <dataframe_models>` to define a
:py:class:`~pandera.api.polars.model.LazyFrame` for validation.

.. testcode:: polars

    import pandera.polars as pa
    import polars as pl

    from pandera.typing.polars import LazyFrame


    class Schema(pa.DataFrameModel):
        state: str
        city: str
        price: int = pa.Field(in_range={"min_value": 5, "max_value": 20})


    # create a modin dataframe that's validated on object initialization
    lf = LazyFrame[Schema](
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
    print(lf.collect())


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


You can also use the :py:func:`~pandera.check_types` decorator to validate
modin dataframes at runtime:


.. testcode:: polars

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


And of course, you can use the object-based API to validate dask dataframes:


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

What data types are supported?
------------------------------

``pandera`` currently supports all the `scalar data types <https://docs.pola.rs/py-polars/html/reference/datatypes.html>`__
`Nested data types <https://docs.pola.rs/py-polars/html/reference/datatypes.html#nested>`__
are not yet supported. Built-in python types like ``str``, ``int``, ``float``,
and ``bool`` will be handled in the same way that ``polars`` handles them:

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


What's different?
------------------

Compared to the way ``pandera`` handles ``pandas`` dataframes, ``pandera``
attempts to leverage the ``polars`` lazy API as much as possible to leverage
its performance optimization benefits (read more about it
[here](https://docs.pola.rs/user-guide/lazy/using/)).

Because ``pandera`` is a run-time validator, it will need to ``.collect()`` the
data values at certain points of the validation pipeline that require operating
on the data values contained in the ``LazyFrame``. Therefore, calling the
``.validate()`` method on a ``LazyFrame`` will trigger multiple ``.collect()``
operations depending on the schema specification.

The ``schema.validate()`` method is effectively an eager operation that converts
the validated data back into a ``polars.LazyFrame`` before returning the output.
At a high level, this is what happens:

- Apply parsers: add missing columns, coerce the datatypes if ``coerce=True``,
  filter columns, and set defaults. This results in multiple ``.collect()``.
  operations.
- Apply checks: run all core, built-in, and custom checks on the data. Checks
  on metadata are done without ``.collect()`` operations, but checks that inspect
  data values do.
- Convert back to ``LazyFrame`` before returning the validated data.

In the context of a lazy computation pipeline, this means that you can use schemas
as eager checkpoints that validate the data. Pandera is designed such that you
can continue to use the ``LazyFrame`` API after the schema validation step.

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

In the event of a validation error, ``pandera`` will raise a ``SchemaError``
eagerly.

.. testcode:: polars

    invalid_lf = pl.LazyFrame({"a": pl.Series(["1", "2", "3"], dtype=pl.Utf8)})
    SimpleModel.validate(invalid_lf)

.. testoutput:: polars

    Traceback (most recent call last):
    ...
    SchemaError: expected column 'a' to have type Int64, got String

And if you use lazy validation ``pandera`` will raise a ``SchemaErrors`` exception.
This is particularly useful when you want to collect all of the validation errors
present in the data.

.. testcode:: polars

    class MyModel(pa.DataFrameModel):
        a: int
        b: str = pa.Field(isin=[*"abc"])
        c: float = pa.Field(ge=0.0, le=1.0)

    invalid_lf = pl.LazyFrame({
        "a": pl.Series(["1", "2", "3"], dtype=pl.Utf8),
        "b": ["d", "e", "f"],
        "c": [0.0, 1.1, -0.1],
    })
    MyModel.validate(invalid_lf, lazy=True)

.. testoutput:: polars

    Traceback (most recent call last):
    ...
    pandera.errors.SchemaErrors: Schema 'MyModel': 4 errors types were found with a total of 6 failures.
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
