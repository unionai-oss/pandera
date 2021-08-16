.. currentmodule:: pandera

.. _scaling:

Scaling Pandera to Big Data
=================================

Validation on big data comes in two forms. The first is performing one set of
validations on data that doesn't fit in memory. The second happens when a large dataset
is comprised of multiple groups that require different validations. In Pandas semantics,
this is equivalent to a groupby-validate operation. This section will cover using
Pandera for both of these scenarios.

Pandera only supports Pandas DataFrames at the moment. However, the same Pandera code
can be used on top of Spark or Dask engines with Fugue. These computation engines
allow validation to be performed in a distributed setting (with some limitations that
will be explained). Because Dask dataframes are built on top of Pandas dataframes,
bringing Pandera to Dask is relatively easier than with Spark if coded from scratch.

In this example, we'll explore using Fugue, abstraction layer that ports Python, Pandas,
and SQL code to Spark and Dask.

Fugue
-----

Fugue was created to be an easy interface to Spark and Dask.

To run the code below, Fugue needs to:

`pip install fugue[spark]`


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

.. testcode:: scaling_pandera
    :skipif: SKIP_SCALING

    from fugue import transform
    from fugue_spark import SparkExecutionEngine

    spark_df = transform(data, price_validation, schema="*", engine=SparkExecutionEngine)
    spark_df.show()

.. testoutput:: scaling_pandera
    :skipif: SKIP_SCALING

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
        "price": Column(int, Check.in_range(min_value=15,max_value=21)),
    })

    price_checks = {'CA': price_check_CA, 'FL': price_check_FL}

.. testcode:: scaling_pandera
    :skipif: SKIP_SCALING

    from fugue import FugueWorkflow

    def price_validation(df:pd.DataFrame) -> pd.DataFrame:
        location = df['state'].iloc[0]
        check = price_checks[location]
        check.validate(df)
        return df

    with FugueWorkflow(SparkExecutionEngine) as dag:
        df = dag.df(data)
        df = df.partition(by=["state"]).transform(price_validation, schema="*")
        df.show()

.. testoutput:: scaling_pandera
    :skipif: SKIP_SCALING

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
