.. currentmodule:: pandera.pyspark

.. _native_pyspark:

Data Validation with Pyspark SQL ⭐️ (New)
=======================================

*new in 0.16.0*

Apache Spark is an open-source unified analytics engine for large-scale data processing. Spark provides an interface for programming clusters with implicit data parallelism and fault tolerance. 
`Pyspark <https://spark.apache.org/docs/3.2.0/api/python/index.html>`__ is the Python API for Apache Spark, an open source, distributed computing framework and set of libraries for real-time, large-scale data processing.

You can use pandera to validate :py:func:`~pyspark.sql.DataFrame`
 objects directly. First, install
``pandera`` with the ``pyspark`` extra:

.. code:: bash

   pip install pandera[pyspark]

What is different?
------------------
There are some small changes to support nuances of pyspark SQL and expected usage, they are as follow:-

1. The output will a dataframe in pyspark SQL even in case of errors during validation. Instead of raising the error, the errors are collected and can be accessed via attribute as shown in example :any:`Registering Custom Checks` .This decision is based on expectation that most use case of pyspark SQL implementation would be in production where data quality information may be used later, such cases prioritise completing the production load and data quality issue might be solved at a later stage.

2. Unlike the pandas version the default behaviour of the pyspark SQL version for errors is lazy=True. i.e. All the errors would be collected instead of raising at first error instance.

3. No support for lambda based vectorized checks since in spark lambda checks needs UDF which is inefficient. However pyspark sql does support custom check via register custom check method.

4. The custom check has to return a boolean value instead of a series.

5. In defining the type annotation, there is limited support for default python data types such as int, str etc.


Basic Usage
-----------

Then you can use pandera schemas to validate pyspark dataframes. In the example
below we'll use the :ref:`class-based API <dataframe_models>` to define a
:py:class:`~pandera.api.pyspark.model.DataFrameModel` for validation.

.. testcode:: native_pyspark

    import pandera.pyspark as pa
    import pyspark.sql.types as T

    from decimal import Decimal
    from pyspark.sql import DataFrame
    from pandera.pyspark import DataFrameModel

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

    +---+-------+-----+------------------------+----------------------------+
    |id |product|price|description             |meta                        |
    +---+-------+-----+------------------------+----------------------------+
    |5  |Bread  |null |[description of product]|{product_category -> dairy} |
    |15 |Butter |null |[more details here]     |{product_category -> bakery}|
    +---+-------+-----+------------------------+----------------------------+



You can use the :py:func:`~PanderaSchema.validate` function to validate
pyspark sql dataframes at runtime.
If you notice in below code the output is expected to be dataframe with an appended attribute "pandera", which itself contains an errors attribute that stores the error report.

.. testcode:: native_pyspark

    df_out = PanderaSchema.validate(check_obj=df)
    df_out_errors = df_out.pandera.errors
    print(df_out_errors)


.. testoutput:: native_pyspark

    {
        "SCHEMA":{
            "COLUMN_NOT_IN_DATAFRAME":[
                {
                    "schema":"PanderaSchema",
                    "column":"PanderaSchema",
                    "check":"column_in_dataframe",
                    "error":"column 'product_name' not in dataframe Row(id=5, product='Bread', price=None, description=['description of product'], meta={'product_category': 'dairy'})"
                }
            ],
            "WRONG_DATATYPE":[
                {
                    "schema":"PanderaSchema",
                    "column":"description",
                    "check":"dtype('ArrayType(StringType(), True)')",
                    "error":"expected column 'description' to have type ArrayType(StringType(), True), got ArrayType(StringType(), False)"
                },
                {
                    "schema":"PanderaSchema",
                    "column":"meta",
                    "check":"dtype('MapType(StringType(), StringType(), True)')",
                    "error":"expected column 'meta' to have type MapType(StringType(), StringType(), True), got MapType(StringType(), StringType(), False)"
                }
            ]
        },
        "DATA":{
            "DATAFRAME_CHECK":[
                {
                    "schema":"PanderaSchema",
                    "column":"id",
                    "check":"greater_than(5)",
                    "error":"column 'id' with type IntegerType() failed validation greater_than(5)"
                }
            ]
        }
    }

As seen above, the error report is grouped on type of validation (schema or data) and category of errors such as dataframe_check or wrong_dtype, etc. for easy processing by downstream applications.

Note: further operations on same dataframe may drop errors information as expected. So recommendation is to extract the `errors` dictionary object and persist before continuing.


Granular Control of Pandera's Execution
----------------------------------------
*new in 0.16.0*

By default, error report is generated for both schema and data level validation.
In *0.16.0* we also introduced a more granular control over the execution of Pandera's validation flow. This is achieved by introducing configurable settings set using environment variables that allow you to control execution at three different levels:

1.	SCHEMA_ONLY: This setting performs schema validations only. It checks that the data conforms to the schema definition, but does not perform any additional data-level validations.

2.	DATA_ONLY: This setting performs data-level validations only. It checks the data against the defined constraints and rules, but does not validate the schema.

3.	SCHEMA_DATA_BOTH: This setting performs both schema and data-level validations. It checks the data against both the schema definition and the defined constraints and rules.

By configuring `PANDERA_DEPTH` parameter, you can choose the level of validation that best fits your specific use case. For example, if the main concern is to ensure that the data conforms to the defined schema, the SCHEMA_ONLY setting can be used to reduce the overall processing time. Alternatively, if the data is known to conform to the schema and the focus is on ensuring data quality, the DATA_ONLY setting can be used to prioritize data-level validations.

.. testcode:: native_pyspark
    {
        'PANDERA_VALIDATION': 'ENABLE',
        'PANDERA_DEPTH': 'SCHEMA_AND_DATA',
    }


By default, validations are enabled and depth is set to `SCHEMA_AND_DATA` which can be changed to `SCHEMA_ONLY` or `DATA_ONLY` as required by the use case.


Registering Custom Checks
-------------------------

``pandera`` already offers an interface to register custom checks functions so
that they're available in the :class:`~pandera.api.checks.Check` namespace. See
:ref:`the extensions<extensions>` document for more information.

Pyspark SQL supports custom check using only this way. Unlike the pandas version, pyspark sql does not support lambda function in check.
It is because to implement lambda functions would mean introducing spark UDF which is expensive operation due to serialization, hence it is better to create native pyspark function.

Note: The output of the function should be a boolean value "True" for passed and "False" for failure. Unlike the Pandas version which expect it to be a series of boolean values.

.. testcode:: native_pyspark

    from pandera.extensions import register_check_method

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

            product: StringType()
            code: IntegerType() = Field(
                new_pyspark_check={
                    "max_value": 30
                }
            )

