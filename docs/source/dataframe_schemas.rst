.. pandera documentation for DataFrameSchemas

.. _DataFrameSchemas:

DataFrame Schemas
=================

DataFrameSchemas enable the specification of a schema that a dataframe is
validated against.

The DataFrameSchema object consists of |column|_\s, |index|_\s, whether to
|coerced|_ the types of all of the columns and |strict|_ which if True will
error if the DataFrame contains columns that aren’t in the DataFrameSchema.


.. |column| replace:: ``Column``
.. |index| replace:: ``Index``
.. |coerced| replace:: ``coerce``
.. |strict| replace:: ``strict``

.. code:: python

   schema = DataFrameSchema({
       "column1": Column(Int),
       "column2": Column(Float, Check(lambda s: s < -1.2)),
       # you can provide a list of validators
       "column3": Column(String, [
           Check(lambda s: s.str.startswith("value")),
           Check(lambda s: s.str.split("_", expand=True).shape[1] == 2)
       ]),
   },
      strict = True,
      coerce = True,
   )
   })

.. _column:

Column Validation
-----------------

A ``Column`` must contain a *type* to be validated. It can be optionally
verified for `null values`_ or duplicate values. The column can be coerced_ into
the specified type, and the required_ parameter allows control over whether or
not the column is allowed to be missing.

:ref:`Column Checks<checks>` allow for the DataFrame's values to be
checked against a user provided function. ``Check``\ s support
:ref:`grouping<grouping>` by a different column so that the user can make
assertions about subsets of the ``Column`` of interest.

:ref:`Column Hypothesis test<hypothesis>` support testing different
column so that assertions can be made about the relationships between
``Column``\s.

.. _null values:

Null Values in Columns
~~~~~~~~~~~~~~~~~~~~~~

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

.. note:: Due to a known limitation in
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

.. _coerced:

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

.. _required:

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

.. _strict:

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

.. _index:

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

MultiIndex Validation
---------------------

``pandera`` also supports multi-index column and index validation.


MultiIndex Columns
~~~~~~~~~~~~~~~~~~

Specifying multi-index columns follows the ``pandas`` syntax of specifying tuples
for each level in the index hierarchy:

.. code:: python

  import pandas as pd

  from pandera import Column, DataFrameSchema, Index, Int, String

  schema = DataFrameSchema({
      ("foo", "bar"): Column(Int),
      ("foo", "baz"): Column(String)
  })

  df = pd.DataFrame({
      ("foo", "bar"): [1, 2, 3],
      ("foo", "baz"): ["a", "b", "c"],
  })

  schema.validate(df)

  #   foo
  #   bar baz
  # 0   1   a
  # 1   2   b
  # 2   3   c


MultiIndex Indexes
~~~~~~~~~~~~~~~~~~

The ``pandera.MultiIndex`` class allows you to define multi-index indexes by
composing a list of ``pandera.Index`` objects.

.. code:: python

  import pandas as pd

  from pandera import Column, DataFrameSchema, Index, MultiIndex, Int, \
      String, Check

  schema = DataFrameSchema(
      columns={"column1": Column(Int)},
      index=MultiIndex([
          Index(String,
                Check(lambda s: s.isin(["foo", "bar"])),
                name="index0"),
          Index(Int, name="index1"),
      ])
  )

  df = pd.DataFrame(
      data={"column1": [1, 2, 3]},
      index=pd.MultiIndex(
          levels=[["foo", "bar"], [0, 1, 2, 3, 4]],
          labels=[[0, 1, 0], [0, 1, 2]],
          names=["index0", "index1"],
      )
  )

  schema.validate(df)

  #                column1
  # index0 index1
  # foo    0             1
  # bar    1             2
  # foo    2             3
