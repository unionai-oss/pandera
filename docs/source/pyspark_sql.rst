.. currentmodule:: pandera.pyspark

.. _scaling_pyspark:

Data Validation with Pyspark SQL ⭐️ (New)
=======================================

*new in 0.16.0*

`Pyspark <https://spark.apache.org/docs/3.2.0/api/python/index.html>`__ is a
distributed compute framework that offers a pandas drop-in replacement dataframe
implementation via the `pyspark Dataframe API <https://spark.apache.org/docs/3.2.0/api/python/reference/pyspark.sql.htmll>`__ .
You can use pandera to validate :py:func:`~pyspark.sql.DataFrame`
 objects directly. First, install
``pandera`` with the ``pyspark`` extra:

.. code:: bash

   pip install pandera[pyspark]


Then you can use pandera schemas to validate pyspark dataframes. In the example
below we'll use the :ref:`class-based API <dataframe_models>` to define a
:py:class:`~pandera.api.pyspark.model.DataFrameModel` for validation.

.. testcode:: scaling_pyspark

    import pandera.pyspark as pa

    from pandera.typing.pyspark_sql import DataFrame


    class Schema(pa.DataFrameModel):
        state: str
        city: str
        price: int = pa.Field(in_range={"min_value": 5, "max_value": 20})


    # create a pyspark.pandas dataframe that's validated on object initialization
    df = Schema.validate(spark.createDataFrame([Row(state='FL', city='Orlando', price=8),
                                                Row(state='FL', city='Miami', price=12),
                                               Row(state='FL', city='Tampa', price=10),
                                               Row(state='CA', city='San Francisco', price=16),
                                               Row(state='CA', city='Los Angeles', price=20),
                                               Row(state='CA', city='San Diego', price=18)]
                        )
    df.show()


.. testoutput:: scaling_pyspark

    +-----+-------------+-----+
    |state|         city|price|
    +-----+-------------+-----+
    |   FL|      Orlando|    8|
    |   FL|        Miami|   12|
    |   FL|        Tampa|   10|
    |   CA|San Francisco|   16|
    |   CA|  Los Angeles|   20|
    |   CA|    San Diego|   18|
    +-----+-------------+-----+



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
    print(schema(df).show())


.. testoutput:: scaling_pyspark

    +-----+-------------+-----+
    |state|         city|price|
    +-----+-------------+-----+
    |   FL|      Orlando|    8|
    |   FL|        Miami|   12|
    |   FL|        Tampa|   10|
    |   CA|San Francisco|   16|
    |   CA|  Los Angeles|   20|
    |   CA|    San Diego|   18|
    +-----+-------------+-----+

