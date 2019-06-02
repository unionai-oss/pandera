.. pandera documentation master file

Datatype and Column-like property tests
=======================================

A light-weight and flexible validation package for
`pandas <http://pandas.pydata.org>`__ data structures.

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

Column Validation
-----------------

Nullable Columns
~~~~~~~~~~~~~~~~

By default, SeriesSchema/Column objects assume that values are not
nullable. In order to accept null values, you need to explicitly specify
``nullable=True``, or else you’ll get an error.

.. code:: python

   import numpy as np
   import pandas as pd

   from pandera import Check, Column, DataFrameSchema, Int

   df = pd.DataFrame({"column1": [5, 1, np.nan]})

   non_null_schema = DataFrameSchema({
       "column1": Column(Int, Check(lambda x: x > 0))
   })

   non_null_schema.validate(df)

   # SchemaError: non-nullable series contains null values: {2: nan}

**NOTE:** Due to a known limitation in
`pandas <http://pandas.pydata.org/pandas-docs/stable/gotchas.html#support-for-integer-na>`__,
integer arrays cannot contain ``NaN`` values, so this schema will return
a dataframe where ``column1`` is of type ``float``.

.. code:: python

   from pandera import Check, Column, DataFrameSchema, Int

   df = ...
   null_schema = DataFrameSchema({
       "column1": Column(Int, Check(lambda x: x > 0), nullable=True)
   })

   null_schema.validate(df)

   #    column1
   # 0      5.0
   # 1      1.0
   # 2      NaN

Coercing Types on Columns
~~~~~~~~~~~~~~~~~~~~~~~~~

If you specify ``Column(dtype, ..., coerce=True)`` as part of the
DataFrameSchema definition, calling ``schema.validate`` will first
coerce the column into the specified ``dtype``.

.. code:: python

   import pandas as pd

   from pandera import Column, DataFrameSchema, String

   df = pd.DataFrame({"column1": [1, 2, 3]})
   schema = DataFrameSchema({"column1": Column(String, coerce=True)})

   validated_df = schema.validate(df)
   assert isinstance(validated_df.column1.iloc[0], str)

Note the special case of integers columns not supporting ``nan`` values.
In this case, ``schema.validate`` will complain if ``coerce == True``
and null values are allowed in the column.

The best way to handle this case is to simply specify the column as a
``Float`` or ``Object``.

.. code:: python

   import pandas as pd

   from pandera import Column, DataFrameSchema, Float, Int, Object

   df = pd.DataFrame({"column1": [1., 2., 3, pd.np.nan]})
   schema = DataFrameSchema({"column1": Column(Int, coerce=True, nullable=True)})

   validated_df = schema.validate(df)
   # ValueError: cannot convert float NaN to integer


   schema_object = DataFrameSchema({
       "column1": Column(Object, coerce=True, nullable=True)})
   schema_float = DataFrameSchema({
       "column1": Column(Float, coerce=True, nullable=True)})

   schema_object.validate(df).dtypes
   # column1    object


   schema_float.validate(df).dtypes
   # column1    float64

If you want to coerce all of the columns specified in the
``DataFrameSchema``, you can specify the ``coerce`` argument with
``DataFrameSchema(..., coerce=True)``.

Required Columns
~~~~~~~~~~~~~~~~

By default all columns specified in the schema are required, meaning
that if a column is missing in the input dataframe an exception will be
thrown. If you want to make a column optional specify ``required=False``
in the column constructor:

.. code:: python

   import pandas as pd

   from pandera import Column, DataFrameSchema, Int, String

   df = pd.DataFrame({"column2": ["hello", "pandera"]})
   schema = DataFrameSchema({
       "column1": Column(Int, required=False),
       "column2": Column(String)
   })

   validated_df = schema.validate(df)
   # list(validated_df.columns) == ["column2"]

Handling of Dataframe Columns not in the Schema
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, columns that aren’t specified in the schema aren’t checked.
If you want to check that the dataframe *only* contains columns in the
schema, specify ``strict=True``:

.. code:: python

   import pandas as pd
   from pandera import Column, DataFrameSchema, Int

   schema = DataFrameSchema({"column1": Column(Int, nullable=True)},
                            strict=True)
   df = pd.DataFrame({"column2": [1, 2, 3]})

   schema.validate(df)

   # SchemaError: column 'column2' not in DataFrameSchema {'column1': <Schema Column: 'None' type=int64>}



Index Validation
----------------

You can also specify an ``Index`` in the ``DataFrameSchema``.

.. code:: python

   import pandas as pd

   from pandera import Column, DataFrameSchema, Index, Int, String, Check

   schema = DataFrameSchema(
       columns={"a": Column(Int)},
       index=Index(
           String,
           Check(lambda x: x.startswith("index_"))))

   df = pd.DataFrame({"a": [1, 2, 3]}, index=["index_1", "index_2", "index_3"])

   print(schema.validate(df))

   #          a
   # index_1  1
   # index_2  2
   # index_3  3


   df.index = ["foo1", "foo2", "foo3"]
   schema.validate(df)

   # SchemaError: <Schema Index> failed element-wise validator 0:
   # <lambda>
   # failure cases:
   #              index  count
   # failure_case
   # foo1           [0]      1
   # foo2           [1]      1
   # foo3           [2]      1

``SeriesSchema``
~~~~~~~~~~~~~~~~

.. code:: python

   import pandas as pd

   from pandera import Check, SeriesSchema, String

   # specify multiple validators
   schema = SeriesSchema(String, [
       Check(lambda x: "foo" in x),
       Check(lambda x: x.endswith("bar")),
       Check(lambda x: len(x) > 3)])

   schema.validate(pd.Series(["1_foobar", "2_foobar", "3_foobar"]))

   #  0    1_foobar
   #  1    2_foobar
   #  2    3_foobar
   #  dtype: object