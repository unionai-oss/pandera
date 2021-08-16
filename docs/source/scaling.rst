.. currentmodule:: pandera

.. _scaling:

Scaling Pandera to Big Data
=================================

Pandera only support Pandas DataFrames at the moment. However, the same Pandera code
can be used on Spark or Dask with Fugue. Because Dask is built on top of Pandas DataFrames,
it may be easier to

Fugue
-----

Fugue is an abstraction layer that ports Python, Pandas, and SQL code to Spark and Dask. Here,

To install:


Example
-------

In this example, a pandas ``DataFrame`` in created with ``state``, ``city`` and ``price``
columns.

.. testcode:: scaling_pandera

    import pandas as pd

    data = pd.DataFrame({'state': ['FL','FL','FL','CA','CA','CA'],
                        'city': ['Orlando', 'Miami', 'Tampa',
                                'San Francisco', 'Los Angeles', 'San Diego'],
                        'price': [8, 12, 10, 16, 20, 18]})
    print(data)

.. testoutput:: scaling_pandera

    state           city  price
    0    FL        Orlando      8
    1    FL          Miami     12
    2    FL          Tampa     10
    3    CA  San Francisco     16
    4    CA    Los Angeles     20
    5    CA      San Diego     18

Validation is then applied using pandera.

.. testcode:: scaling_pandera

    from pandera import Column, DataFrameSchema, Check

    price_check = DataFrameSchema(
        {"price": Column(int, Check.in_range(min_value=5,max_value=20))}
    )

    def price_validation(data:pd.DataFrame) -> pd.DataFrame:
        price_check.validate(data)
        return data

    price_validation(data)

Bringing this to Spark

.. testcode:: scaling_pandera
    from fugue import transform
    from fugue_spark import SparkExecutionEngine

    spark_df = transform(data, using=price_validation, schema="*", engine=SparkExecutionEngine)
    spark_df.show()

.. testoutput:: scaling_pandera
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


Validation by Partition
-----------------------

There is an interesting use case that comes up

.. testcode:: scaling_pandera

    price_check_FL = DataFrameSchema({
        "price": Column(int, Check.in_range(min_value=7,max_value=13)),
    })

    price_check_CA = DataFrameSchema({
        "price": Column(Int, Check.in_range(min_value=15,max_value=21)),
    })

    price_checks = {'CA': price_check_CA, 'FL': price_check_FL}

.. testcode:: scaling_pandera

    from fugue import FugueWorkflow

    def price_validation(df:pd.DataFrame) -> pd.DataFrame:
        location = df['state'].iloc[0]
        check = price_checks[location]
        check.validate(df)
        return df

    with FugueWorkflow(SparkExecutionEngine) as dag:
        df = dag.df(data)
        df = df.partition(by=["state"]).transform(price_validation)
        df.show()

.. testoutput:: scaling_pandera

    SparkDataFrame
    state:str|city:str                                                                       |price:long
    ---------+-------------------------------------------------------------------------------+----------
    CA       |San Francisco                                                                  |16
    CA       |Los Angeles                                                                    |20
    CA       |San Diego                                                                      |18
    FL       |Orlando                                                                        |8
    FL       |Miami                                                                          |12
    FL       |Tampa                                                                          |10
    Total count: 6
