.. pandera documentation for seriesschemas

Series Schemas
==============

``SeriesSchema``\s allow for the validation of series against a schema. They are
very similar to :ref:`columns<column>` and :ref:`indexes<index>` specified
in :ref:`DataFrameSchemas<DataFrameSchemas>`.

Series Validation
~~~~~~~~~~~~~~~~~

Schemas can be validated by creating

.. testcode:: series_validation

    import pandas as pd
    import pandera as pa

    from pandera import Check, SeriesSchema

    # specify multiple validators
    schema = SeriesSchema(pa.String, [
        Check(lambda s: s.str.startswith("foo")),
        Check(lambda s: s.str.endswith("bar")),
        Check(lambda x: len(x) > 3, element_wise=True)])

    print(schema.validate(pd.Series(["foobar", "foobar", "foobar"])))

.. testoutput:: series_validation

    0    foobar
    1    foobar
    2    foobar
    dtype: object
