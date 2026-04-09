.. _api-geopandas:

GeoPandas
=========

Entry point for GeoPandas-aware schemas and models. Requires the ``geopandas``
extra. Implementations live in :mod:`pandera.api.geopandas`; this top-level
module re-exports the public API.

:mod:`pandera.geopandas` re-exports the full :mod:`pandera.pandas` namespace in
addition to the GeoPandas-specific classes below.

.. autosummary::
   :toctree: generated
   :template: class.rst

   pandera.geopandas.GeoDataFrameModel
   pandera.geopandas.GeoDataFrameSchema
