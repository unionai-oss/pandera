.. currentmodule:: pandera

.. _scaling_pyspark:

Data Validation with Pyspark ⭐️ (New)
=======================================

*new in 0.10.0*

`Pyspark <https://spark.apache.org/docs/3.2.0/api/python/index.html>`__ is a
distributed compute framework that offers a pandas drop-in replacement dataframe
implementation via the `pyspark.pandas API <https://spark.apache.org/docs/3.2.0/api/python/reference/pyspark.pandas/index.html>`__ .
You can use pandera to validate :py:func:`~pyspark.pandas.DataFrame`
and :py:func:`~pyspark.pandas.Series` objects directly. First, install
``pandera`` with the ``pyspark`` extra:

.. code:: bash

   pip install pandera[pyspark]


Then you can use pandera schemas to validate pyspark dataframes. In the example
below we'll use the :ref:`class-based API <schema_models>` to define a
:py:class:`SchemaModel` for validation.

.. testcode:: scaling_pyspark

    import pyspark.pandas as ps
    import pandas as pd
    import pandera as pa

    from pandera.typing.pyspark import DataFrame, Series


    class Schema(pa.SchemaModel):
        state: Series[str]
        city: Series[str]
        price: Series[int] = pa.Field(in_range={"min_value": 5, "max_value": 20})


    # create a pyspark.pandas dataframe that's validated on object initialization
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


.. testoutput:: scaling_pyspark

      state           city  price
    0    FL        Orlando      8
    1    FL          Miami     12
    2    FL          Tampa     10
    3    CA  San Francisco     16
    4    CA    Los Angeles     20
    5    CA      San Diego     18


You can also use the :py:func:`~pandera.check_types` decorator to validate
pyspark pandas dataframes at runtime:


.. testcode:: scaling_pyspark

    @pa.check_types
    def function(df: DataFrame[Schema]) -> DataFrame[Schema]:
        return df[df["state"] == "CA"]

    print(function(df))


.. testoutput:: scaling_pyspark

      state           city  price
    3    CA  San Francisco     16
    4    CA    Los Angeles     20
    5    CA      San Diego     18


And of course, you can use the object-based API to validate dask dataframes:


.. testcode:: scaling_pyspark

    schema = pa.DataFrameSchema({
        "state": pa.Column(str),
        "city": pa.Column(str),
        "price": pa.Column(int, pa.Check.in_range(min_value=5, max_value=20))
    })
    print(schema(df))


.. testoutput:: scaling_pyspark

      state           city  price
    0    FL        Orlando      8
    1    FL          Miami     12
    2    FL          Tampa     10
    3    CA  San Francisco     16
    4    CA    Los Angeles     20
    5    CA      San Diego     18
