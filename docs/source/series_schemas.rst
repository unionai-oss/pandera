.. pandera documentation for seriesschemas

.. currentmodule:: pandera

.. _SeriesSchemas:

Series Schemas
==============

The :class:`~pandera.schemas.SeriesSchema` class allows for the validation of pandas
``Series`` objects, and are very similar to :ref:`columns<column>` and
:ref:`indexes<index>` described in :ref:`DataFrameSchemas<DataFrameSchemas>`.


.. testcode:: series_validation

    import pandas as pd
    import pandera as pa


    # specify multiple validators
    schema = pa.SeriesSchema(
        pa.String,
        checks=[
            pa.Check(lambda s: s.str.startswith("foo")),
            pa.Check(lambda s: s.str.endswith("bar")),
            pa.Check(lambda x: len(x) > 3, element_wise=True)
        ],
        nullable=False,
        allow_duplicates=True,
        name="my_series")

    validated_series = schema.validate(
        pd.Series(["foobar", "foobar", "foobar"], name="my_series"))
    print(validated_series)

.. testoutput:: series_validation

    0    foobar
    1    foobar
    2    foobar
    Name: my_series, dtype: object
