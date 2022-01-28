.. empty

{{ fullname | escape | underline }}

.. currentmodule:: {{ fullname }}

.. automodule:: {{ fullname }}

   {% block classes %}
   .. rubric:: Pandas Types

   .. autosummary::

      DataFrame
      Index
      Series

   .. rubric:: GeoPandas Types

   .. autosummary::

      geopandas.GeoDataFrame
      geopandas.GeoSeries

   .. rubric:: Dask Types

   .. autosummary::

      dask.DataFrame
      dask.Series
      dask.Index

   .. rubric:: Koalas Types

   .. autosummary::

      koalas.DataFrame
      koalas.Series
      koalas.Index

   .. rubric:: Modin Types

   .. autosummary::

      koalas.DataFrame
      koalas.Series
      koalas.Index

   .. rubric:: FastAPI Types

   .. autosummary::
      :toctree: generated

      fastapi.UploadFile

   .. rubric:: Serialization Formats

   .. autosummary::
      :toctree: generated

      formats.Formats

   {% endblock %}

   {% block attributes %}
   .. rubric:: DataTypes

   .. autosummary::

      Bool
      DateTime
      Timedelta
      Category
      Float
      Float16
      Float32
      Float64
      Int
      Int8
      Int16
      Int32
      Int64
      UInt8
      UInt16
      UInt32
      UInt64
      INT8
      INT16
      INT32
      INT64
      UINT8
      UINT16
      UINT32
      UINT64
      Object
      String
      STRING

    {% endblock %}
