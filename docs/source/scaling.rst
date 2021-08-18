.. currentmodule:: pandera

.. _scaling:

Scaling Pandera to Big Data
=================================

Validation on big data comes in two forms. The first is performing one set of
validations on data that doesn't fit in memory. The second happens when a large dataset
is comprised of multiple groups that require different validations. In pandas semantics,
this would be the equivalent of a ``groupby-validate`` operation. This section will cover
using ``pandera`` for both of these scenarios.

``Pandera`` only supports pandas ``DataFrames`` at the moment. However, the same ``pandera``
code can be used on top of ``Spark`` or ``Dask`` engines with
`Fugue <https://github.com/fugue-project/fugue/>`_ . These computation engines allow validation
to be performed in a distributed setting. ``Fugue`` is an open source abstraction layer that
ports ``Python``, ``pandas``, and ``SQL`` code to ``Spark`` and ``Dask``.

Fugue
-----

``Fugue`` serves as an interface to distributed computing. Because of its non-invasive design,
existing ``Python`` code can be scaled to a distributed setting without significant changes.

To run the example, ``Fugue`` needs to installed separately. Using pip:

.. code:: bash

    pip install fugue[spark]

This will also install ``PySpark`` because of the ``spark`` extra. ``Dask`` is available
with the ``dask`` extra.


Example
-------

In this example, a pandas ``DataFrame`` is created with ``state``, ``city`` and ``price``
columns. ``Pandera`` will be used to validate that the ``price`` column values are within
a certain range.

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


Validation is then applied using pandera. A ``price_validation`` function is
created that runs the validation. None of this will be new.

.. testcode:: scaling_pandera

    from pandera import Column, DataFrameSchema, Check

    price_check = DataFrameSchema(
        {"price": Column(int, Check.in_range(min_value=5,max_value=20))}
    )

    def price_validation(data:pd.DataFrame) -> pd.DataFrame:
        price_check.validate(data)
        return data

The ``transform`` function in ``Fugue`` is the easiest way to use ``Fugue`` with existing ``Python``
functions as seen in the following code snippet. The first two arguments are the ``DataFrame`` and
function to apply. The keyword argument ``schema`` is required because schema is strictly enforced
in distributed settings. Here, the ``schema`` is simply `*` because no new columns are added.

The last part of the ``transform`` function is the ``engine``. Here, the ``SparkExecutionEngine`` is used
to run the code on top of ``Spark``. ``Fugue`` also has a ``DaskExecutionEngine``, and passing nothing uses
the default pandas-based ``ExecutionEngine``. Because the ``SparkExecutionEngine`` is used, the result
becomes a ``Spark DataFrame``.

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

There is an interesting use case that arises with bigger datasets. Frequently, there are logical
groupings of data that require different validations. In the earlier sample data, the
price range for the records with ``state`` FL is lower than the range for the ``state`` CA.
Two :class:`~pandera.schemas.DataFrameSchema` will be created to reflect this. Notice their ranges
for the :class:`~pandera.checks.Check` differ.

.. testcode:: scaling_pandera

    price_check_FL = DataFrameSchema({
        "price": Column(int, Check.in_range(min_value=7,max_value=13)),
    })

    price_check_CA = DataFrameSchema({
        "price": Column(int, Check.in_range(min_value=15,max_value=21)),
    })

    price_checks = {'CA': price_check_CA, 'FL': price_check_FL}

A slight modification is needed to our ``price_validation`` function. ``Fugue`` will partition
the whole dataset into multiple pandas ``DataFrames``. Think of this as a ``groupby``. By the
time ``price_validation`` is used, it only contains the data for one ``state``. The appropriate
``DataFrameSchema`` is pulled and then applied.

To partition our data by ``state``, all we need to do is pass it into the ``transform`` function
through the ``partition`` argument. This splits up the data across different workers before they
each run the ``price_validation`` function. Again, this is like a groupby-validation.

.. testcode:: scaling_pandera
    :skipif: SKIP_SCALING

    def price_validation(df:pd.DataFrame) -> pd.DataFrame:
        location = df['state'].iloc[0]
        check = price_checks[location]
        check.validate(df)
        return df

    spark_df = transform(data,
              price_validation,
              schema="*",
              partition=dict(by="state"),
              engine=SparkExecutionEngine)

    spark_df.show()

.. testoutput:: scaling_pandera
    :skipif: SKIP_SCALING

    SparkDataFrame
    state:str|city:str                                                 |price:long
    ---------+---------------------------------------------------------+----------
    CA       |San Francisco                                            |16
    CA       |Los Angeles                                              |20
    CA       |San Diego                                                |18
    FL       |Orlando                                                  |8
    FL       |Miami                                                    |12
    FL       |Tampa                                                    |10
    Total count: 6

.. note::

    Because operations in a distributed setting are applied per partition, statistical
    validators will be applied on each partition rather than the global dataset. If no
    partitioning scheme is specified, ``Spark`` and ``Dask`` use default partitions. Be
    careful about using operations like mean, min, and max without partitioning beforehand.

    All row-wise validations scale well with this set-up.
