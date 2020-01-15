.. pandera documentation for DataFrameSchemas

.. _DataFrameSchemas:

DataFrame Schemas
=================

The ``DataFrameSchema`` class enables the specification of a schema that
verifies the columns and index of a ``pd.DataFrame`` object.

The ``DataFrameSchema`` object consists of |column|_\s and an |index|_.

.. |column| replace:: ``Column``
.. |index| replace:: ``Index``
.. |coerced| replace:: ``coerce``
.. |strict| replace:: ``strict``

.. testcode:: dataframe_schemas

    import pandera as pa

    from pandera import Column, DataFrameSchema, Check, Index

    schema = DataFrameSchema(
        {
            "column1": Column(pa.Int),
            "column2": Column(pa.Float, Check(lambda s: s < -1.2)),
            # you can provide a list of validators
            "column3": Column(pa.String, [
               Check(lambda s: s.str.startswith("value")),
               Check(lambda s: s.str.split("_", expand=True).shape[1] == 2)
            ]),
        },
        index=Index(pa.Int),
        strict=True,
        coerce=True,
    )

.. _column:

Column Validation
-----------------

A ``Column`` must specify the properties of a column in a dataframe object.
It can be optionally verified for its data type, `null values`_ or duplicate
values. The column can be coerced_ into the specified type, and the
required_ parameter allows control over whether or not the column is allowed to
be missing.

:ref:`Column checks<checks>` allow for the DataFrame's values to be
checked against a user provided function. ``Check`` objects also support
:ref:`grouping<grouping>` by a different column so that the user can make
assertions about subsets of the ``Column`` of interest.

Column Hypotheses enable you to perform statistical hypothesis tests on a
DataFrame in either wide or tidy format. See
:ref:`Hypothesis Testing<hypothesis>` for more details.


.. _null values:

Null Values in Columns
~~~~~~~~~~~~~~~~~~~~~~

By default, SeriesSchema/Column objects assume that values are not
nullable. In order to accept null values, you need to explicitly specify
``nullable=True``, or else you’ll get an error.

.. testcode:: null_values_in_columns

   import numpy as np
   import pandas as pd
   import pandera as pa

   from pandera import Check, Column, DataFrameSchema

   df = pd.DataFrame({"column1": [5, 1, np.nan]})

   non_null_schema = DataFrameSchema({
       "column1": Column(pa.Int, Check(lambda x: x > 0))
   })

   non_null_schema.validate(df)

.. testoutput:: null_values_in_columns

    Traceback (most recent call last):
    ...
    SchemaError: non-nullable series contains null values: {2: nan}

.. note:: Due to a known limitation in
    `pandas <http://pandas.pydata.org/pandas-docs/stable/gotchas.html#support-for-integer-na>`__,
    integer arrays cannot contain ``NaN`` values, so this schema will return
    a DataFrame where ``column1`` is of type ``float``.

.. testcode:: null_values_in_columns

   null_schema = DataFrameSchema({
       "column1": Column(pa.Int, Check(lambda x: x > 0), nullable=True)
   })

   print(null_schema.validate(df))

.. testoutput:: null_values_in_columns

       column1
    0      5.0
    1      1.0
    2      NaN

.. _coerced:

Coercing Types on Columns
~~~~~~~~~~~~~~~~~~~~~~~~~

If you specify ``Column(dtype, ..., coerce=True)`` as part of the
DataFrameSchema definition, calling ``schema.validate`` will first
coerce the column into the specified ``dtype``.

.. testcode:: coercing_types_on_columns

    import pandas as pd
    import pandera as pa

    from pandera import Column, DataFrameSchema

    df = pd.DataFrame({"column1": [1, 2, 3]})
    schema = DataFrameSchema({"column1": Column(pa.String, coerce=True)})

    validated_df = schema.validate(df)
    assert isinstance(validated_df.column1.iloc[0], str)

.. note:: Note the special case of integers columns not supporting ``nan``
    values. In this case, ``schema.validate`` will complain if ``coerce == True``
    and null values are allowed in the column.

.. testcode:: coercing_types_on_columns

    df = pd.DataFrame({"column1": [1., 2., 3, pd.np.nan]})
    schema = DataFrameSchema({
        "column1": Column(pa.Int, coerce=True, nullable=True)
    })

    validated_df = schema.validate(df)

.. testoutput:: coercing_types_on_columns

    Traceback (most recent call last):
    ...
    ValueError: cannot convert float NaN to integer


The best way to handle this case is to simply specify the column as a
``Float`` or ``Object``.


.. testcode:: coercing_types_on_columns

    schema_object = DataFrameSchema({
        "column1": Column(pa.Object, coerce=True, nullable=True)
    })
    schema_float = DataFrameSchema({
        "column1": Column(pa.Float, coerce=True, nullable=True)
    })

    print(schema_object.validate(df).dtypes)
    print(schema_float.validate(df).dtypes)

.. testoutput:: coercing_types_on_columns

    column1    object
    dtype: object
    column1    float64
    dtype: object

If you want to coerce all of the columns specified in the
``DataFrameSchema``, you can specify the ``coerce`` argument with
``DataFrameSchema(..., coerce=True)``.

.. _required:

Required Columns
~~~~~~~~~~~~~~~~

By default all columns specified in the schema are required, meaning
that if a column is missing in the input DataFrame an exception will be
thrown. If you want to make a column optional, specify ``required=False``
in the column constructor:

.. testcode:: required_columns

   import pandas as pd
   import pandera as pa

   from pandera import Column, DataFrameSchema

   df = pd.DataFrame({"column2": ["hello", "pandera"]})
   schema = DataFrameSchema({
       "column1": Column(pa.Int, required=False),
       "column2": Column(pa.String)
   })

   validated_df = schema.validate(df)
   print(validated_df)

.. testoutput:: required_columns

       column2
    0    hello
    1  pandera


Since ``required=True`` by default, missing columns would raise an error:

.. testcode:: required_columns

    schema = DataFrameSchema({
        "column1": Column(pa.Int),
        "column2": Column(pa.String),
    })

    schema.validate(df)

.. testoutput:: required_columns

    Traceback (most recent call last):
    ...
    pandera.SchemaError: column 'column1' not in dataframe
       column2
    0    hello
    1  pandera


.. _column validation:

Stand-alone Column Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to being used in the context of a ``DataFrameSchema``, ``Column``
objects can also be used to validate columns in a dataframe on its own:

.. testcode:: dataframe_schemas

    import pandas as pd
    import pandera as pa

    from pandera import Column, Check

    df = pd.DataFrame({
        "column1": [1, 2, 3],
        "column2": ["a", "b", "c"],
    })

    column1_schema = Column(pa.Int, name="column1")
    column2_schema = Column(pa.String, name="column2")

    # pass the dataframe as an argument to the Column object callable
    df = column1_schema(df)
    validated_df = column2_schema(df)

    # or explicitly use the validate method
    df = column1_schema.validate(df)
    validated_df = column2_schema.validate(df)

    # use the DataFrame.pipe method to validate two columns
    validated_df = df.pipe(column1_schema).pipe(column2_schema)


For multi-column use cases, the ``DataFrameSchema`` is still recommended, but
if you have one or a small number of columns to verify, using ``Column``
objects by themselves is appropriate.


.. _strict:

Handling Dataframe Columns not in the Schema
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, columns that aren’t specified in the schema aren’t checked.
If you want to check that the DataFrame *only* contains columns in the
schema, specify ``strict=True``:

.. testcode:: handling_columns_not_in_schema

    import pandas as pd
    import pandera as pa

    from pandera import Column, DataFrameSchema

    schema = DataFrameSchema(
        {"column1": Column(pa.Int)},
        strict=True)

    df = pd.DataFrame({"column2": [1, 2, 3]})

    schema.validate(df)

.. testoutput:: handling_columns_not_in_schema

    Traceback (most recent call last):
    ...
    SchemaError: column 'column2' not in DataFrameSchema {'column1': <Schema Column: 'None' type=int64>}


.. _index:

Index Validation
----------------

You can also specify an ``Index`` in the ``DataFrameSchema``.

.. testcode:: index_validation

    import pandas as pd
    import pandera as pa

    from pandera import Column, DataFrameSchema, Index, Check

    schema = DataFrameSchema(
       columns={"a": Column(pa.Int)},
       index=Index(
           pa.String,
           Check(lambda x: x.str.startswith("index_"))))

    df = pd.DataFrame(
        data={"a": [1, 2, 3]},
        index=["index_1", "index_2", "index_3"])

    print(schema.validate(df))

.. testoutput:: index_validation

             a
    index_1  1
    index_2  2
    index_3  3


In the case that the DataFrame index doesn't pass the ``Check``.

.. testcode:: index_validation

    df = pd.DataFrame(
        data={"a": [1, 2, 3]},
        index=["foo1", "foo2", "foo3"])

    schema.validate(df)

.. testoutput:: index_validation

    Traceback (most recent call last):
    ...
    SchemaError: <Schema Index> failed element-wise validator 0:
    <lambda>
    failure cases:
                 index  count
    failure_case
    foo1           [0]      1
    foo2           [1]      1
    foo3           [2]      1

MultiIndex Validation
---------------------

``pandera`` also supports multi-index column and index validation.


MultiIndex Columns
~~~~~~~~~~~~~~~~~~

Specifying multi-index columns follows the ``pandas`` syntax of specifying
tuples for each level in the index hierarchy:

.. testcode:: multiindex_columns

    import pandas as pd
    import pandera as pa

    from pandera import Column, DataFrameSchema, Index

    schema = DataFrameSchema({
        ("foo", "bar"): Column(pa.Int),
        ("foo", "baz"): Column(pa.String)
    })

    df = pd.DataFrame({
        ("foo", "bar"): [1, 2, 3],
        ("foo", "baz"): ["a", "b", "c"],
    })

    print(schema.validate(df))

.. testoutput:: multiindex_columns
    :options: +NORMALIZE_WHITESPACE

      foo
      bar baz
    0   1   a
    1   2   b
    2   3   c

.. _multiindex:

MultiIndex Indexes
~~~~~~~~~~~~~~~~~~

The ``pandera.MultiIndex`` class allows you to define multi-index indexes by
composing a list of ``pandera.Index`` objects.

.. testcode:: multiindex_indexes

  import pandas as pd
  import pandera as pa

  from pandera import Column, DataFrameSchema, Index, MultiIndex, Check

  schema = DataFrameSchema(
      columns={"column1": Column(pa.Int)},
      index=MultiIndex([
          Index(pa.String,
                Check(lambda s: s.isin(["foo", "bar"])),
                name="index0"),
          Index(pa.Int, name="index1"),
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

  print(schema.validate(df))

.. testoutput:: multiindex_indexes
    :options: +NORMALIZE_WHITESPACE

                   column1
    index0 index1
    foo    0             1
    bar    1             2
    foo    2             3


Pandas DType
---------------------

Pandas provides a `dtype` parameter for casting a dataframe to a specific dtype
schema. ``DataFrameSchema`` provides a `dtype` property which returns a pandas
style dict. The keys of the dict are column names and values are the dtype.

Some examples of where this can be provided to pandas are:

- https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
- https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html

.. testcode:: dataframe_dtype

  import pandas as pd
  import pandera as pa

  from pandera import Column, DataFrameSchema, Index, MultiIndex, Check

  schema = DataFrameSchema(
      columns={
        "column1": Column(pa.Int),
        "column2": Column(pa.Category),
        "column3": Column(pa.Bool)
      },
  )

  df = pd.DataFrame.from_dict({
    "a": {"column1": 1, "column2": "valueA", "column3": True},
    "b": {"column1": 1, "column2": "valueB", "column3": True},
    },
    orient="index"
  ).astype(schema.dtype).sort_index(axis=1)

  print(schema.validate(df))

.. testoutput:: dataframe_dtype
    :options: +NORMALIZE_WHITESPACE

       column1 column2  column3
    a        1  valueA     True
    b        1  valueB     True



DataFrameSchema Transformations
-------------------------------

Pandera supports transforming a schema using ``.add_columns`` and ``.remove_columns``.

``.add_columns`` expects a ``Dict[str, Any]``, i.e. the same as when defining ``Columns`` in a ``DataFrameSchema``:

.. testcode:: add_columns

  from pandera import DataFrameSchema, Column, Int, Check, String, Object

  schema = DataFrameSchema({
    "col1": Column(Int, Check(lambda s: s >= 0)),
    }, strict=True)

  new_schema = schema.add_columns({
    "col2": Column(String, Check(lambda x: x <= 0)),
    "col3": Column(Object, Check(lambda x: x == 0))
    })

  expected_schema = schema.add_columns({
    "col1": Column(Int, Check(lambda s: s >= 0)),
    "col2": Column(String, Check(lambda x: x <= 0)),
    "col3": Column(Object, Check(lambda x: x == 0))
    })

  print(new_schema == expected_schema)

.. testoutput:: add_columns
    :options: +NORMALIZE_WHITESPACE

    True

``.remove_columns`` expects a list of one or more Column names:

.. testcode:: remove_columns

  from pandera import DataFrameSchema, Column, Int, Check, String, Object

  schema = DataFrameSchema({
    "col1": Column(Int, Check(lambda s: s >= 0)),
    "col2": Column(String, Check(lambda x: x <= 0)),
    "col3": Column(Object, Check(lambda x: x == 0))
    }, strict=True)

  new_schema = schema.remove_columns(["col2", "col3"])

  print(new_schema)

.. testoutput:: remove_columns
    :options: +NORMALIZE_WHITESPACE

    DataFrameSchema(
        columns={
            "col1": "<Schema Column: 'col1' type=int64>"
        },
    index=None,
    transformer=None,
    coerce=False,
    strict=True
    )
