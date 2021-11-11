.. currentmodule:: pandera

.. _scaling_koalas:

Data Validation with Koalas
===========================

*new in 0.8.0*

`Koalas <https://koalas.readthedocs.io/en/latest/>`__ is a distributed
compute framework that offers a pandas drop-in replacement dataframe
implementation. You can use pandera to validate :py:func:`~databricks.koalas.DataFrame`
and :py:func:`~databricks.koalas.Series` objects directly. First, install
``pandera`` with the ``dask`` extra:

.. code:: bash

   pip install pandera[koalas]


Then you can use pandera schemas to validate koalas dataframes. In the example
below we'll use the :ref:`class-based API <schema_models>` to define a
:py:class:`SchemaModel` for validation.

.. testcode:: scaling_koalas

    import databricks.koalas as ks
    import pandas as pd
    import pandera as pa

    from pandera.typing.koalas import DataFrame, Series


    class Schema(pa.SchemaModel):
        state: Series[str]
        city: Series[str]
        price: Series[int] = pa.Field(in_range={"min_value": 5, "max_value": 20})


    # create a koalas dataframe that's validated on object initialization
    df = DataFrame[Schema](
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
    print(df)


.. testoutput:: scaling_koalas

      state           city  price
    0    FL        Orlando      8
    1    FL          Miami     12
    2    FL          Tampa     10
    3    CA  San Francisco     16
    4    CA    Los Angeles     20
    5    CA      San Diego     18


You can also use the :py:func:`~pandera.check_types` decorator to validate
koalas dataframes at runtime:


.. testcode:: scaling_koalas

    @pa.check_types
    def function(df: DataFrame[Schema]) -> DataFrame[Schema]:
        return df[df["state"] == "CA"]

    print(function(df))


.. testoutput:: scaling_koalas

      state           city  price
    3    CA  San Francisco     16
    4    CA    Los Angeles     20
    5    CA      San Diego     18


And of course, you can use the object-based API to validate dask dataframes:


.. testcode:: scaling_koalas

    schema = pa.DataFrameSchema({
        "state": pa.Column(str),
        "city": pa.Column(str),
        "price": pa.Column(int, pa.Check.in_range(min_value=5, max_value=20))
    })
    print(schema(df))


.. testoutput:: scaling_koalas

      state           city  price
    0    FL        Orlando      8
    1    FL          Miami     12
    2    FL          Tampa     10
    3    CA  San Francisco     16
    4    CA    Los Angeles     20
    5    CA      San Diego     18
