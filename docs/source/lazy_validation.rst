.. currentmodule:: pandera

.. _lazy_validation:

Lazy Validation
===============

*New in version 0.4.0*

By default, when you call the ``validate`` method on schema or schema component
objects, a :class:`~pandera.errors.SchemaError` is raised as soon as one of the
assumptions specified in the schema is falsified. For example, for a
:class:`~pandera.schemas.DataFrameSchema` object, the following situations will raise an
exception:

* a column specified in the schema is not present in the dataframe.
* if ``strict=True``, a column in the dataframe is not specified in the schema.
* the ``pandas_dtype`` does not match.
* if ``coerce=True``, the dataframe column cannot be coerced into the specified
  ``pandas_dtype``.
* the :class:`~pandera.checks.Check` specified in one of the columns returns ``False`` or
  a boolean series containing at least one ``False`` value.


For example:

.. testcode:: non_lazy_validation

   import pandas as pd
   import pandera as pa

   from pandera import Check, Column, DataFrameSchema

   df = pd.DataFrame({"column": ["a", "b", "c"]})

   schema = pa.DataFrameSchema({"column": Column(pa.Int)})
   schema.validate(df)

.. testoutput:: non_lazy_validation

    Traceback (most recent call last):
    ...
    SchemaError: expected series 'column' to have type int64, got object


For more complex cases, it is useful to see all of the errors raised during
the ``validate`` call so that you can debug the causes of errors on different
columns and checks. The ``lazy`` keyword argument in the ``validate`` method
of all schemas and schema components gives you the option of doing just this:

.. testcode:: lazy_validation

    import pandas as pd
    import pandera as pa
    
    from pandera import Check, Column, DataFrameSchema

    schema = pa.DataFrameSchema(
        columns={
            "int_column": Column(pa.Int),
            "float_column": Column(pa.Float, Check.greater_than(0)),
            "str_column": Column(pa.String, Check.equal_to("a")),
            "date_column": Column(pa.DateTime),
        },
        strict=True
    )
    
    df = pd.DataFrame({
        "int_column": ["a", "b", "c"],
        "float_column": [0, 1, 2],
        "str_column": ["a", "b", "d"],
        "unknown_column": None,
    })
    
    schema.validate(df, lazy=True)

.. testoutput:: lazy_validation

    Traceback (most recent call last):
    ...
    pandera.errors.SchemaErrors: A total of 5 schema errors were found.

    Error Counts
    ------------
    - column_not_in_schema: 1
    - column_not_in_dataframe: 1
    - schema_component_check: 3

    Schema Error Summary
    --------------------
                                                             failure_cases  n_failure_cases
    schema_context  column       check
    DataFrameSchema <NA>         column_in_dataframe         [date_column]                1
                                 column_in_schema         [unknown_column]                1
    Column          float_column pandas_dtype('float64')           [int64]                1
                    int_column   pandas_dtype('int64')            [object]                1
                    str_column   equal_to(a)                        [b, d]                2

    Usage Tip
    ---------

    Directly inspect all errors by catching the exception:

    ```
    try:
        schema.validate(dataframe, lazy=True)
    except SchemaErrors as err:
        err.schema_errors  # dataframe of schema errors
        err.data  # invalid dataframe
    ```

As you can see from the output above, a :class:`~pandera.errors.SchemaErrors`
exception is raised with a summary of the error counts and failure cases
caught by the schema. You can also see from the **Usage Tip** that you can
catch these errors and inspect the failure cases in a more granular form:


.. testcode:: lazy_validation
    :skipif: SKIP
    
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        print("Schema errors and failure cases:")
        print(err.schema_errors.head())
        print("\nDataFrame object that failed validation:")
        print(err.data.head())

.. testoutput:: lazy_validation
    :skipif: SKIP

    Schema errors and failure cases:
        schema_context        column                    check check_number  \
    0  DataFrameSchema          None         column_in_schema         None
    1  DataFrameSchema          None      column_in_dataframe         None
    2           Column    int_column    pandas_dtype('int64')         None
    3           Column  float_column  pandas_dtype('float64')         None
    4           Column    str_column              equal_to(a)            0

         failure_case index
    0  unknown_column  None
    1     date_column  None
    2          object  None
    3           int64  None
    4               b     1

    DataFrame object that failed validation:
      int_column  float_column str_column unknown_column
    0          a             0          a           None
    1          b             1          b           None
    2          c             2          d           None
