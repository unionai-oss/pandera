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

What's different?
------------------

Compared to the way
