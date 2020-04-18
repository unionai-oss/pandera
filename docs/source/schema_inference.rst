.. currentmodule:: pandera

.. _schema_inference:

Schema Inference
================

.. warning::
   
   This functionality is experimental and not feature-complete, use with
   caution!

The :py:func:`infer_schema` enables you to quickly infer a draft schema from
a pandas dataframe or series.


.. testcode:: infer_dataframe_schema

   import pandas as pd
   import pandera as pa

   from pandera import Check, Column, DataFrameSchema

   df = pd.DataFrame({
       "column1": [5, 10, 20],
       "column2": [5., 1., 3.],
       "column3": ["a", "b", "c"],
   })
   schema = pa.infer_schema(df)

   print(schema)

.. testoutput:: infer_dataframe_schema

   DataFrameSchema(
       columns={
           "column1": "<Schema Column: 'column1' type=int64>",
           "column2": "<Schema Column: 'column2' type=float64>",
           "column3": "<Schema Column: 'column3' type=string>"
       },
       index=None,
       transformer=None,
       coerce=True,
       strict=False
   )


You can then be modify the inferred schema with to obtain the schema definition
that you're satisfied with.

For :py:class:`DataFrameSchema` objects, you can use the
:py:func:`DataFrameSchema.add_columns`,
:py:func:`DataFrameSchema.remove_columns`, and
:py:func:`DataFrameSchema.update_column` methods to produce updated copies
of the original inferred schema. For :py:class:`SeriesSchema` objects, you
can update series checks with the :py:func:`SeriesSchema.set_checks` method.
