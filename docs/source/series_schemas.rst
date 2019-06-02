.. pandera documentation for seriesschemas

Series Schemas
==============

``SeriesSchema``\s allow for the validation of series against a schema. They are
very similiar to :ref:`columns<column>` and :ref:`indexes<index>` specified
in :ref:`DataFrameSchemas<DataFrameSchemas>`.

Series Validation
~~~~~~~~~~~~~~~~~

Schemas can be validated by creating

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

