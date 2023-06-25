.. currentmodule:: pandera.pyspark

.. _native_pyspark:

Data Validation with Pyspark SQL ⭐️ (New)
==========================================

*new in 0.16.0*

Apache Spark is an open-source unified analytics engine for large-scale data processing. Spark provides an interface for programming clusters with implicit data parallelism and fault tolerance. 
`Pyspark <https://spark.apache.org/docs/3.2.0/api/python/index.html>`__ is the Python API for Apache Spark, an open source, distributed computing framework and set of libraries for real-time, large-scale data processing.

You can use pandera to validate ``pyspark.sql.DataFrame`` objects directly. First, install ``pandera`` with the ``pyspark`` extra:

.. code:: bash

   pip install pandera[pyspark]

What is different?
------------------

There are some small changes to support nuances of pyspark SQL and expected usage, they are as follow:

1. The output of ``schema.validate`` will produce a dataframe in pyspark SQL even in case of errors during validation.
   Instead of raising the error, the errors are collected and can be accessed via the ``dataframe.pandera.errors`` attribute  as shown in this example. 

   .. note::
      This design decision is based on the expectation that most use cases for pyspark SQL dataframes means entails a production ETL setting.
      In these settings, pandera prioritizes completing the production load and saving the data quality issues for downstream rectification.

2. Unlike the pandera pandas schemas, the default behaviour of the pyspark SQL version for errors is ``lazy=True``, i.e. all the errors would be collected instead of raising at first error instance.

3. There is no support for lambda based vectorized checks since in spark lambda checks needs UDFs, which is inefficient. However pyspark sql does support custom checks via the :func:`~pandera.extensions.register_check_method` decorator.

4. The custom check has to return a scalar boolean value instead of a series.

5. In defining the type annotation, there is limited support for default python data types such as int, str etc. When using the ``pandera.pyspark`` API, using ``pyspark.sql.types`` based datatypes such as ``StringType``, ``IntegerType``, etc. is highly recommended.


Basic Usage
-----------

In this section, lets look at an end to end example of how pandera would work in a native pyspark implementation.

.. testcode:: native_pyspark

    import pandera.pyspark as pa
    import pyspark.sql.types as T

    from decimal import Decimal
    from pyspark.sql import SparkSession
    from pyspark.sql import DataFrame
    from pandera.pyspark import DataFrameModel

    spark = SparkSession.builder.getOrCreate()

    class PanderaSchema(DataFrameModel):
        id: T.IntegerType() = pa.Field(gt=5)
        product_name: T.StringType() = pa.Field(str_startswith="B")
        price: T.DecimalType(20, 5) = pa.Field()
        description: T.ArrayType(T.StringType()) = pa.Field()
        meta: T.MapType(T.StringType(), T.StringType()) = pa.Field()

    data = [
        (5, "Bread", Decimal(44.4), ["description of product"], {"product_category": "dairy"}),
        (15, "Butter", Decimal(99.0), ["more details here"], {"product_category": "bakery"}),
    ]

    spark_schema = T.StructType(
        [
            T.StructField("id", T.IntegerType(), False),
            T.StructField("product", T.StringType(), False),
            T.StructField("price", T.DecimalType(20, 5), False),
            T.StructField("description", T.ArrayType(T.StringType(), False), False),
            T.StructField(
                "meta", T.MapType(T.StringType(), T.StringType(), False), False
            ),
        ],
    )
    df = spark.createDataFrame(data, spark_schema)
    df.show()


.. testoutput:: native_pyspark

    +---+-------+--------+--------------------+--------------------+
    | id|product|   price|         description|                meta|
    +---+-------+--------+--------------------+--------------------+
    |  5|  Bread|44.40000|[description of p...|{product_category...|
    | 15| Butter|99.00000| [more details here]|{product_category...|
    +---+-------+--------+--------------------+--------------------+

In above example, ``PanderaSchema`` class inherits from ``DataFrameModel`` base class. It has type annotations for 5 fields with 2 of the fields having checks enforced e.g. ``gt=5`` and ``str_startswith="B"``.

Just to simulate some schema and data validations, we also defined native spark's schema ``spark_schema`` and enforced it on our dataframe ``df``.

Next, you can use the :py:func:`~PanderaSchema.validate` function to validate pyspark sql dataframes at runtime.

.. testcode:: native_pyspark

    df_out = PanderaSchema.validate(check_obj=df)

After running :py:func:`~PanderaSchema.validate`, the returned object ``df_out`` will be a ``pyspark`` dataframe extended to hold validation results  on it.

Introducing Error Report
------------------------

*new in 0.16.0*

You can print the validation results as follows:

.. testcode:: native_pyspark

    import json

    df_out_errors = df_out.pandera.errors
    print(json.dumps(dict(df_out_errors), indent=4))

.. testoutput:: native_pyspark

    {
        "SCHEMA": {
            "COLUMN_NOT_IN_DATAFRAME": [
                {
                    "schema": "PanderaSchema",
                    "column": "PanderaSchema",
                    "check": "column_in_dataframe",
                    "error": "column 'product_name' not in dataframe\nRow(id=5, product='Bread', price=Decimal('44.40000'), description=['description of product'], meta={'product_category': 'dairy'})"
                }
            ],
            "WRONG_DATATYPE": [
                {
                    "schema": "PanderaSchema",
                    "column": "description",
                    "check": "dtype('ArrayType(StringType(), True)')",
                    "error": "expected column 'description' to have type ArrayType(StringType(), True), got ArrayType(StringType(), False)"
                },
                {
                    "schema": "PanderaSchema",
                    "column": "meta",
                    "check": "dtype('MapType(StringType(), StringType(), True)')",
                    "error": "expected column 'meta' to have type MapType(StringType(), StringType(), True), got MapType(StringType(), StringType(), False)"
                }
            ]
        },
        "DATA": {
            "DATAFRAME_CHECK": [
                {
                    "schema": "PanderaSchema",
                    "column": "id",
                    "check": "greater_than(5)",
                    "error": "column 'id' with type IntegerType() failed validation greater_than(5)"
                }
            ]
        }
    }

As seen above, the error report is aggregated on 2 levels in a `python dictionary` object:
1. type of validation (``SCHEMA`` or ``DATA``) and 
2. category of errors such as ``DATAFRAME_CHECK`` or ``WRONG_DATATYPE``, etc. 

so as to be easily consumed by downstream applications such as timeseries visualization of errors over time.

.. important::
    It's critical to extract errors report from `df_out.pandera.errors` as any further `pyspark` operations may reset it.


Granular Control of Pandera's Execution
----------------------------------------

*new in 0.16.0*

By default, error report is generated for both schema and data level validation.
In *0.16.0* we also introduced a more granular control over the execution of Pandera's validation flow. This is achieved by introducing configurable settings using environment variables that allow you to control execution at three different levels:

1.	``SCHEMA_ONLY`` - to perform schema validations only. It checks that data conforms to the schema definition, but does not perform any data-level validations on dataframe.
2.	``DATA_ONLY`` - to perform data-level validations only. It validates that data conforms to the defined `checks`, but does not validate the schema.
3.	``SCHEMA_AND_DATA``: (**default**) - to perform both schema and data level validations. It runs most exhaustive validation and could be compute intensive.

You can override default behaviour by setting an environment variable from terminal before running the `pandera` process as:

    `export PANDERA_VALIDATION_DEPTH=SCHEMA_ONLY`

This will be picked up by `pandera` to only enforce SCHEMA level validations.


Switching Validation On and Off
-------------------------------

*new in 0.16.0*

It's very common in production to enable or disable certain services to save computing resources. We thought about it and thus introduced a switch to enable or disable pandera in production.

You can override default behaviour by setting an environment variable from terminal before running  the `pandera` process as follow:

    `export PANDERA_VALIDATION_ENABLED=False`

This will be picked up by `pandera` to disable all validations in the application.


By default, validations are enabled and depth is set to ``SCHEMA_AND_DATA`` which can be changed to ``SCHEMA_ONLY`` or ``DATA_ONLY`` as required by the use case.


Registering Custom Checks
-------------------------

``pandera`` already offers an interface to register custom checks functions so
that they're available in the :class:`~pandera.api.checks.Check` namespace. See
:ref:`the extensions<extensions>` document for more information.

Unlike the pandas version, pyspark sql does not support lambda function inside ``check``.
It is because to implement lambda functions would mean introducing spark UDF which is expensive operation due to serialization, hence it is better to create native pyspark function.

Note: The output of the function should be a boolean value ``True`` for passed and ``False`` for failure. Unlike the Pandas version which expect it to be a series of boolean values.

.. testcode:: native_pyspark

    from pandera.extensions import register_check_method
    import pyspark.sql.types as T

    @register_check_method
    def new_pyspark_check(pyspark_obj, *, max_value) -> bool:
        """Ensure values of the data are strictly below a maximum value.
        :param max_value: Upper bound not to be exceeded. Must be
            a type comparable to the dtype of the column datatype of pyspark
        """

        cond = col(pyspark_obj.column_name) <= max_value
        return pyspark_obj.dataframe.filter(~cond).limit(1).count() == 0

    class Schema(DataFrameModel):
            """Schema"""

            product: T.StringType()
            code: T.IntegerType() = pa.Field(
                new_pyspark_check={
                    "max_value": 30
                }
            )

Metadata at Dataframe and Field level
-------------------------------------

*new in 0.16.0*

In real world use cases, we often need to embed additional information on objects. Thats why we introduced a new feature in Pandera that allows users to store additional metadata at ``Field`` and ``Schema`` / ``Model`` levels. This feature is designed to provide greater context and information about the data, which can be leveraged by other applications.

For example, by storing details about a specific column, such as data type, format, or units, developers can ensure that downstream applications are able to interpret and use the data correctly. Similarly, by storing information about which columns of a schema are needed for a specific use case, developers can optimize data processing pipelines, reduce storage costs, and improve query performance.

.. testcode:: native_pyspark

    import pyspark.sql.types as T
    class PanderaSchema(DataFrameModel):
        """Pandera Schema Class"""

        product_id: T.IntegerType() = pa.Field()
        product_class: T.StringType() = pa.Field(
            metadata={
                "search_filter": "product_pricing",
            },
        )
        product_name: T.StringType() = pa.Field()
        price: T.DecimalType(20, 5) = pa.Field()

        class Config:
            """Config of pandera class"""

            name = "product_info"
            strict = True
            coerce = True
            metadata = {"category": "product-details"}


As seen in above example, ``product_class`` field has additional embedded information such as ``search_filter``. This metadata can be leveraged to search and filter multiple schemas for certain keywords.
This is clearly a very basic example, but the possibilities are endless with having metadata at ``Field`` and `DataFrame` levels.

We also provided a helper function to extract metadata from a schema as follows:

.. testcode:: native_pyspark
    
    PanderaSchema.get_metadata()

This feature is available for ``pyspark.sql`` and ``pandas`` both.
