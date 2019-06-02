.. pandera documentation master file

Checks
======

Example Usage
-------------

``DataFrameSchema``
~~~~~~~~~~~~~~~~~~~

.. code:: python

   import pandas as pd

   from pandera import Column, DataFrameSchema, Float, Int, String, Check


   # validate columns
   schema = DataFrameSchema({
       # the check function expects a series argument and should output a boolean
       # or a boolean Series.
       "column1": Column(Int, Check(lambda s: s <= 10)),
       "column2": Column(Float, Check(lambda s: s < -1.2)),
       # you can provide a list of validators
       "column3": Column(String, [
           Check(lambda s: s.str.startswith("value_")),
           Check(lambda s: s.str.split("_", expand=True).shape[1] == 2)
       ]),
   })

   # alternatively, you can pass strings representing the legal pandas datatypes:
   # http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
   schema = DataFrameSchema({
       "column1": Column("int64", Check(lambda s: s <= 10)),
       ...
   })

   df = pd.DataFrame({
       "column1": [1, 4, 0, 10, 9],
       "column2": [-1.3, -1.4, -2.9, -10.1, -20.4],
       "column3": ["value_1", "value_2", "value_3", "value_2", "value_1"]
   })

   validated_df = schema.validate(df)
   print(validated_df)

   #     column1  column2  column3
   #  0        1     -1.3  value_1
   #  1        4     -1.4  value_2
   #  2        0     -2.9  value_3
   #  3       10    -10.1  value_2
   #  4        9    -20.4  value_1

Vectorized vs. Element-wise Checks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the functions passed into ``Check``\ s are expected to have
the following signature: ``pd.Series -> bool|pd.Series[bool]``. For the
``Check`` to pass, all of the elements in the boolean series must
evaluate to ``True``.

If you want to make atomic checks for each element in the Column, then
you can provide the ``element_wise=True`` keyword argument:

.. code:: python

   import pandas as pd

   from pandera import Check, Column, DataFrameSchema, Int

   schema = DataFrameSchema({
       "a": Column(Int, [
           # a vectorized check that returns a bool
           Check(lambda s: s.mean() > 5, element_wise=False),
           # a vectorized check that returns a boolean series
           Check(lambda s: s > 0, element_wise=False),
           # an element-wise check that returns a bool
           Check(lambda x: x > 0, element_wise=True),
       ]),
   })

   df = pd.DataFrame({"a": [4, 4, 5, 6, 6, 7, 8, 9]})
   schema.validate(df)

By default ``element_wise=False`` so that you can take advantage of the
speed gains provided by the ``pandas.Series`` API by writing vectorized
checks.


Column Check Groups
-------------------

``Column`` ``Check``\ s support grouping by a different column so that
you can make assertions about subsets of the ``Column`` of interest.
This changes the function signature of the ``Check`` function so that
its input is a dict where keys are the group names and keys are subsets
of the ``Column`` series.

Specifying ``groupby`` as a column name, list of column names, or
callable changes the expected signature of the ``Check`` function
argument to ``dict[Any|tuple[Any], Series] -> bool|Series[bool]`` where
the dict keys are the discrete keys in the ``groupby`` columns.

.. code:: python

   import pandas as pd

   from pandera import DataFrameSchema, Column, Check, Bool, Float, Int, String


   schema = DataFrameSchema({
       "height_in_feet": Column(Float, [
           # groupby as a single column
           Check(lambda g: g[False].mean() > 6, groupby="age_less_than_20"),
           # define multiple groupby columns
           Check(lambda g: g[(True, "F")].sum() == 9.1,
                 groupby=["age_less_than_20", "sex"]),
           # groupby as a callable with signature (DataFrame) -> DataFrameGroupBy
           Check(lambda g: g[(False, "M")].median() == 6.75,
                 groupby=lambda df: (
                   df
                   .assign(age_less_than_15=lambda d: d["age"] < 15)
                   .groupby(["age_less_than_15", "sex"]))),
       ]),
       "age": Column(Int, Check(lambda s: s > 0)),
       "age_less_than_20": Column(Bool),
       "sex": Column(String, Check(lambda s: s.isin(["M", "F"])))
   })

   df = (
       pd.DataFrame({
           "height_in_feet": [6.5, 7, 6.1, 5.1, 4],
           "age": [25, 30, 21, 18, 13],
           "sex": ["M", "M", "F", "F", "F"]
       })
       .assign(age_less_than_20=lambda x: x["age"] < 20)
   )

   schema.validate(df)

In the above example we define a ``DataFrameSchema`` with column checks
for ``height_in_feet`` using a single column, multiple columns, and a
more complex groupby function that creates a new column
``age_less_than_15`` on the fly.
