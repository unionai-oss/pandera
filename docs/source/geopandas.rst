.. currentmodule:: pandera

.. _scaling_geopandas:

Data Validation with GeoPandas
===========================

*new in 0.9.0*

`GeoPandas <https://geopandas.org/en/stable/docs.html>`__ is an extension of Pandas that adds
support for geospatial data. You can use pandera to validate :py:func:`~geopandas.GeoDataFrame`
and :py:func:`~geopandas.GeoSeries` objects directly. First, install
``pandera`` with the ``geopandas`` extra:

.. code:: bash

   pip install pandera[geopandas]


Then you can use pandera schemas to validate geodataframes. In the example
below we'll use the :ref:`class-based API <schema_models>` to define a
:py:class:`SchemaModel` for validation.

.. testcode:: scaling_geopandas

    import geopandas as gpd
    import pandas as pd
    import pandera as pa
    from shapely.geometry import Polygon

    from pandera.typing.geopandas import GeoDataFrame, GeoSeries


    class Schema(pa.SchemaModel):
        geometry: GeoSeries
        region: Series[str]


    # create a geodataframe that's validated on object initialization
    df = GeoDataFrame[Schema](
        {
            'geometry': [
                Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
                Polygon(((0, 0), (0, -1), (-1, -1), (-1, 0)))
            ],
            'region': ['NA','SA']
        }
    )
    print(df)


.. testoutput:: scaling_geopandas

                                                geometry region
    0  POLYGON ((0.00000 0.00000, 0.00000 1.00000, 1....     NA
    1  POLYGON ((0.00000 0.00000, 0.00000 -1.00000, -...     SA
