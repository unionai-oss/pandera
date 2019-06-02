.. pandera documentation master file

pandera
=======

A light-weight and flexible validation package for
`pandas <http://pandas.pydata.org>`__ data structures.

Why?
----

Because pandas data structures hide a lot of information, and explicitly
validating them in production-critical or reproducible research settings
is a good idea.

And it also makes it easier to review pandas code :)

Install
-------

.. code:: bash

   pip install pandera

Example Usage
-------------

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

Informative Errors
------------------

If the dataframe does not pass validation checks, ``pandera`` provides
useful error messages. An ``error`` argument can also be supplied to
``Check`` for custom error messages.

.. code:: python

   import pandas as pd

   from pandera import Column, DataFrameSchema, Int, Check

   simple_schema = DataFrameSchema({
       "column1": Column(
           Int, Check(lambda x: 0 <= x <= 10, error="range checker [0, 10]"))
   })

   # validation rule violated
   fail_check_df = pd.DataFrame({
       "column1": [-20, 5, 10, 30],
   })

   simple_schema.validate(fail_check_df)

   # schema.SchemaError: series failed element-wise validator 0:
   # <lambda>: range checker [0, 10]
   # failure cases:
   #              index  count
   # failure_case
   # -20            [0]      1
   #  30            [3]      1


   # column name mis-specified
   wrong_column_df = pd.DataFrame({
       "foo": ["bar"] * 10,
       "baz": [1] * 10
   })

   simple_schema.validate(wrong_column_df)

   #  SchemaError: column 'column1' not in dataframe
   #     foo  baz
   #  0  bar    1
   #  1  bar    1
   #  2  bar    1
   #  3  bar    1
   #  4  bar    1

Contributing
------------

All contributions, bug reports, bug fixes, documentation improvements,
enhancements and ideas are welcome.

A detailed overview on how to contribute can be found in the
`contributing
guide <https://github.com/cosmicBboy/pandera/blob/master/.github/CONTRIBUTING.md>`__
on GitHub.

Issues
------

Submit issues, feature requests or bugfixes on
`github <https://github.com/cosmicBboy/pandera/issues>`__.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   self
   dataframe_schemas
   series_schemas
   checks
   hypothesis
   decorators
   pandera

Indices and tables
==================

* :ref:`genindex`