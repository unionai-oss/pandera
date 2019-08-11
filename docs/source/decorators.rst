.. pandera documentation for check_input and check_output decorators

.. _decorators:

Decorators for Existing Workflows
=================================

If you have an existing data pipeline that uses pandas data structures,
you can use the ``check_input`` and ``check_output`` decorators to
easily check function arguments or returned variables from existing
functions.

``check_input``
~~~~~~~~~~~~~~~

Validates input pandas DataFrame/Series before entering the wrapped
function.

.. code:: python

   import pandas as pd

   from pandera import DataFrameSchema, Column, Check, Int, Float, check_input


   df = pd.DataFrame({
       "column1": [1, 4, 0, 10, 9],
       "column2": [-1.3, -1.4, -2.9, -10.1, -20.4],
   })

   in_schema = DataFrameSchema({
       "column1": Column(Int, Check(lambda x: 0 <= x <= 10)),
       "column2": Column(Float, Check(lambda x: x < -1.2)),
   })


   # by default, assumes that the first argument is dataframe/series.
   @check_input(in_schema)
   def preprocessor(dataframe):
       dataframe["column4"] = dataframe["column1"] + dataframe["column2"]
       return dataframe


   # or you can provide the argument name as a string
   @check_input(in_schema, "dataframe")
   def preprocessor(dataframe):
       ...


   # or integer representing index in the positional arguments.
   @check_input(in_schema, 1)
   def preprocessor(foo, dataframe):
       ...


   preprocessed_df = preprocessor(df)
   print(preprocessed_df)

   #  Output:
   #     column1  column2  column3  column4
   #  0        1     -1.3  value_1     -0.3
   #  1        4     -1.4  value_2      2.6
   #  2        0     -2.9  value_3     -2.9
   #  3       10    -10.1  value_2     -0.1
   #  4        9    -20.4  value_1    -11.4

``check_output``
~~~~~~~~~~~~~~~~

The same as ``check_input``, but this decorator checks the output
DataFrame/Series of the decorated function.

.. code:: python

   from pandera import DataFrameSchema, Column, Check, Int, check_output


   preprocessed_df = ...

   # assert that all elements in "column1" are zero
   out_schema = DataFrameSchema({
       "column1": Column(Int, Check(lambda x: x == 0))
   })


   # by default assumes that the pandas DataFrame/Schema is the only output
   @check_output(out_schema)
   def zero_column_1(df):
       df["column1"] = 0
       return df


   # you can also specify in the index of the argument if the output is list-like
   @check_output(out_schema, 1)
   def zero_column_1_arg(df):
       df["column1"] = 0
       return "foobar", df


   # or the key containing the data structure to verify if the output is dict-like
   @check_output(out_schema, "out_df")
   def zero_column_1_dict(df):
       df["column1"] = 0
       return {"out_df": df, "out_str": "foobar"}


   # for more complex outputs, you can specify a function
   @check_output(out_schema, lambda x: x[1]["out_df"])
   def zero_column_1_custom(df):
       df["column1"] = 0
       return ("foobar", {"out_df": df})


   zero_column_1(preprocessed_df)
   zero_column_1_arg(preprocessed_df)
   zero_column_1_dict(preprocessed_df)
   zero_column_1_custom(preprocessed_df)
