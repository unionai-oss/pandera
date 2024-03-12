.. currentmodule:: pandera

.. _scaling_modin:

Data Validation with Modin
==========================

*new in 0.8.0*

`Modin <https://modin.readthedocs.io/en/latest/>`__ is a distributed
compute framework that offers a pandas drop-in replacement dataframe
implementation. You can use pandera to validate :py:func:`~modin.pandas.DataFrame`
and :py:func:`~modin.pandas.Series` objects directly. First, install
``pandera`` with the ``dask`` extra:

.. code:: bash

   pip install pandera[modin]       # installs both ray and dask backends
   pip install pandera[modin-ray]   # only ray backend
   pip install pandera[modin-dask]  # only dask backend


Then you can use pandera schemas to validate modin dataframes. In the example
below we'll use the :ref:`class-based API <dataframe_models>` to define a
:py:class:`~pandera.api.model.pandas.DataFrameModel` for validation.

.. testcode:: scaling_modin
    :skipif: SKIP_MODIN

    import modin.pandas as pd
    import pandas as pd
    import pandera as pa

    from pandera.typing.modin import DataFrame, Series


    class Schema(pa.DataFrameModel):
        state: Series[str]
        city: Series[str]
        price: Series[int] = pa.Field(in_range={"min_value": 5, "max_value": 20})


    # create a modin dataframe that's validated on object initialization
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


.. testoutput:: scaling_modin
    :skipif: SKIP_MODIN

      state           city  price
    0    FL        Orlando      8
    1    FL          Miami     12
    2    FL          Tampa     10
    3    CA  San Francisco     16
    4    CA    Los Angeles     20
    5    CA      San Diego     18


You can also use the :py:func:`~pandera.check_types` decorator to validate
modin dataframes at runtime:


.. testcode:: scaling_modin
    :skipif: SKIP_MODIN

    @pa.check_types
    def function(df: DataFrame[Schema]) -> DataFrame[Schema]:
        return df[df["state"] == "CA"]

    print(function(df))


.. testoutput:: scaling_modin
    :skipif: SKIP_MODIN

      state           city  price
    3    CA  San Francisco     16
    4    CA    Los Angeles     20
    5    CA      San Diego     18


And of course, you can use the object-based API to validate modin dataframes:


.. testcode:: scaling_modin
    :skipif: SKIP_MODIN

    schema = pa.DataFrameSchema({
        "state": pa.Column(str),
        "city": pa.Column(str),
        "price": pa.Column(int, pa.Check.in_range(min_value=5, max_value=20))
    })
    print(schema(df))


.. testoutput:: scaling_modin
    :skipif: SKIP_MODIN

      state           city  price
    0    FL        Orlando      8
    1    FL          Miami     12
    2    FL          Tampa     10
    3    CA  San Francisco     16
    4    CA    Los Angeles     20
    5    CA      San Diego     18
