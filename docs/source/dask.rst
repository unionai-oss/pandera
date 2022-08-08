.. currentmodule:: pandera

.. _scaling_dask:

Data Validation with Dask
=========================

*new in 0.8.0*

`Dask <https://docs.dask.org/en/latest/dataframe.html>`__ is a distributed
compute framework that offers a pandas-like dataframe API.
You can use pandera to validate :py:func:`~dask.dataframe.DataFrame`
and :py:func:`~dask.dataframe.Series` objects directly. First, install
``pandera`` with the ``dask`` extra:

.. code:: bash

   pip install pandera[dask]


Then you can use pandera schemas to validate dask dataframes. In the example
below we'll use the :ref:`class-based API <schema_models>` to define a
:py:class:`SchemaModel` for validation.

.. testcode:: scaling_dask

    import dask.dataframe as dd
    import pandas as pd
    import pandera as pa

    from pandera.typing.dask import DataFrame, Series


    class Schema(pa.SchemaModel):
        state: Series[str]
        city: Series[str]
        price: Series[int] = pa.Field(in_range={"min_value": 5, "max_value": 20})


    ddf = dd.from_pandas(
        pd.DataFrame(
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
        ),
        npartitions=2
    )
    pandera_ddf = Schema(ddf)

    print(pandera_ddf)


.. testoutput:: scaling_dask

    Dask DataFrame Structure:
                    state    city  price
    npartitions=2
    0              object  object  int64
    3                 ...     ...    ...
    5                 ...     ...    ...
    Dask Name: validate, 2 graph layers


As you can see, passing the dask dataframe into ``Schema`` will produce
another dask dataframe which hasn't been evaluated yet. What this means is
that pandera will only validate when the dask graph is evaluated.

.. testcode:: scaling_dask

    print(pandera_ddf.compute())


.. testoutput:: scaling_dask

      state           city  price
    0    FL        Orlando      8
    1    FL          Miami     12
    2    FL          Tampa     10
    3    CA  San Francisco     16
    4    CA    Los Angeles     20
    5    CA      San Diego     18


You can also use the :py:func:`~pandera.check_types` decorator to validate
dask dataframes at runtime:

.. testcode:: scaling_dask

    @pa.check_types
    def function(ddf: DataFrame[Schema]) -> DataFrame[Schema]:
        return ddf[ddf["state"] == "CA"]

    print(function(ddf).compute())


.. testoutput:: scaling_dask

      state           city  price
    3    CA  San Francisco     16
    4    CA    Los Angeles     20
    5    CA      San Diego     18


And of course, you can use the object-based API to validate dask dataframes:


.. testcode:: scaling_dask

    schema = pa.DataFrameSchema({
        "state": pa.Column(str),
        "city": pa.Column(str),
        "price": pa.Column(int, pa.Check.in_range(min_value=5, max_value=20))
    })
    print(schema(ddf).compute())


.. testoutput:: scaling_dask

      state           city  price
    0    FL        Orlando      8
    1    FL          Miami     12
    2    FL          Tampa     10
    3    CA  San Francisco     16
    4    CA    Los Angeles     20
    5    CA      San Diego     18
