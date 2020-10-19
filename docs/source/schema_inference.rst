.. currentmodule:: pandera

.. _schema_inference:

Schema Inference
================

*New in version 0.4.0*

.. warning::
   
   This functionality is experimental. Use with caution!

With simple use cases, writing a schema definition manually is pretty
straight-forward with pandera. However, it can get tedious to do this with
dataframes that have many columns of various data types.

To help you handle these cases, the :func:`~pandera.schema_inference.infer_schema` function enables
you to quickly infer a draft schema from a pandas dataframe or series. Below
is a simple example:

.. testcode:: infer_dataframe_schema
   :skipif: SKIP

   import pandas as pd
   import pandera as pa

   from pandera import Check, Column, DataFrameSchema

   df = pd.DataFrame({
       "column1": [5, 10, 20],
       "column2": ["a", "b", "c"],
       "column3": pd.to_datetime(["2010", "2011", "2012"]),
   })
   schema = pa.infer_schema(df)
   print(schema)

.. testoutput:: infer_dataframe_schema
   :skipif: SKIP

    DataFrameSchema(
        columns={
            "column1": "<Schema Column: 'column1' type=int64>",
            "column2": "<Schema Column: 'column2' type=string>",
            "column3": "<Schema Column: 'column3' type=datetime64[ns]>"
        },
        checks=[],
        index=<Schema Index>,
        transformer=None,
        coerce=True,
        strict=False
    )


These inferred schemas are **rough drafts** that shouldn't be used for
validation without modification. You can modify the inferred schema to
obtain the schema definition that you're satisfied with.

For :class:`~pandera.schemas.DataFrameSchema` objects, the following methods create
modified copies of the schema:

* :func:`~pandera.schemas.DataFrameSchema.add_columns`
* :func:`~pandera.schemas.DataFrameSchema.remove_columns`
* :func:`~pandera.schemas.DataFrameSchema.update_column`

For :class:`~pandera.schemas.SeriesSchema` objects:

* :func:`~pandera.schemas.SeriesSchema.set_checks`

The section below describes two workflows for persisting and modifying an
inferred schema.

Schema Persistence
------------------

There are two ways of persisting schemas, inferred or otherwise.

Write to a Python script
~~~~~~~~~~~~~~~~~~~~~~~~

You can also write your schema to a python script with :func:`~pandera.io.to_script`:

.. testcode:: infer_dataframe_schema
   :skipif: SKIP

   # supply a file-like object, Path, or str to write to a file. If not
   # specified, to_script will output the code as a string.
   schema_script = schema.to_script()
   print(schema_script)

.. testoutput:: infer_dataframe_schema
   :skipif: SKIP

    from pandas import Timestamp
    from pandera import (
        DataFrameSchema,
        Column,
        Check,
        Index,
        MultiIndex,
        PandasDtype,
    )

    schema = DataFrameSchema(
        columns={
            "column1": Column(
                pandas_dtype=PandasDtype.Int64,
                checks=[
                    Check.greater_than_or_equal_to(min_value=5.0),
                    Check.less_than_or_equal_to(max_value=20.0),
                ],
                nullable=False,
                allow_duplicates=True,
                coerce=False,
                required=True,
                regex=False,
            ),
            "column2": Column(
                pandas_dtype=PandasDtype.String,
                checks=None,
                nullable=False,
                allow_duplicates=True,
                coerce=False,
                required=True,
                regex=False,
            ),
            "column3": Column(
                pandas_dtype=PandasDtype.DateTime,
                checks=[
                    Check.greater_than_or_equal_to(
                        min_value=Timestamp("2010-01-01 00:00:00")
                    ),
                    Check.less_than_or_equal_to(
                        max_value=Timestamp("2012-01-01 00:00:00")
                    ),
                ],
                nullable=False,
                allow_duplicates=True,
                coerce=False,
                required=True,
                regex=False,
            ),
        },
        index=Index(
            pandas_dtype=PandasDtype.Int64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                Check.less_than_or_equal_to(max_value=2.0),
            ],
            nullable=False,
            coerce=False,
            name=None,
        ),
        coerce=True,
        strict=False,
        name=None,
    )

As a python script, you can iterate on an inferred schema and use it to
validate data once you are satisfied with your schema definition.


Write to YAML
~~~~~~~~~~~~~

You can also write the schema object to a yaml file with :func:`~pandera.io.to_yaml`,
and you can then read it into memory with :func:`~pandera.io.from_yaml`. The
:func:`~pandera.schemas.DataFrameSchema.to_yaml` and :func:`~pandera.schemas.DataFrameSchema.from_yaml`
is a convenience method for this functionality.

.. testcode:: infer_dataframe_schema
   :skipif: SKIP

   # supply a file-like object, Path, or str to write to a file. If not
   # specified, to_yaml will output a yaml string.
   yaml_schema = schema.to_yaml()
   print(yaml_schema)

.. testoutput:: infer_dataframe_schema
   :skipif: SKIP

    schema_type: dataframe
    version: 0.4.5
    columns:
      column1:
        pandas_dtype: int64
        nullable: false
        checks:
          greater_than_or_equal_to: 5.0
          less_than_or_equal_to: 20.0
        allow_duplicates: true
        coerce: false
        required: true
        regex: false
      column2:
        pandas_dtype: string
        nullable: false
        checks: null
        allow_duplicates: true
        coerce: false
        required: true
        regex: false
      column3:
        pandas_dtype: datetime64[ns]
        nullable: false
        checks:
          greater_than_or_equal_to: '2010-01-01 00:00:00'
          less_than_or_equal_to: '2012-01-01 00:00:00'
        allow_duplicates: true
        coerce: false
        required: true
        regex: false
    index:
    - pandas_dtype: int64
      nullable: false
      checks:
        greater_than_or_equal_to: 0.0
        less_than_or_equal_to: 2.0
      name: null
      coerce: false
    coerce: true
    strict: false

You can edit this yaml file by specifying column names under the ``column``
key. The respective values map onto key-word arguments in the
:class:`~pandera.schema_components.Column` class.

.. note::

   Currently, only built-in :class:`~pandera.checks.Check` methods are supported under the
   ``checks`` key.
