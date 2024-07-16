.. _api-dataframe-models:

DataFrame Models
================

DataFrame Model
---------------

.. autosummary::
   :toctree: generated
   :template: class.rst

   pandera.api.pandas.model.DataFrameModel
   pandera.api.polars.model.DataFrameModel
   pandera.api.pyspark.model.DataFrameModel
   pandera.api.dataframe.model.DataFrameModel

Model Components
----------------

.. autosummary::
   :toctree: generated

   pandera.api.dataframe.model_components.Field
   pandera.api.dataframe.model_components.check
   pandera.api.dataframe.model_components.dataframe_check
   pandera.api.dataframe.model_components.parser
   pandera.api.dataframe.model_components.dataframe_parser


Config
------

.. autosummary::
   :toctree: generated
   :template: model_component_class.rst
   :nosignatures:

   pandera.api.pandas.model_config.BaseConfig
   pandera.api.polars.model_config.BaseConfig
   pandera.api.pyspark.model_config.BaseConfig


Typing
------

Pandas
******

.. autosummary::
   :toctree: generated
   :template: class.rst

   pandera.typing.DataFrame
   pandera.typing.Series
   pandera.typing.Index

Geopandas
*********

.. autosummary::
   :toctree: generated
   :template: class.rst

   pandera.typing.geopandas.GeoDataFrame
   pandera.typing.geopandas.GeoSeries

Dask
****

.. autosummary::
   :toctree: generated
   :template: class.rst

   pandera.typing.dask.DataFrame
   pandera.typing.dask.Series
   pandera.typing.dask.Index

Pyspark
*******

.. autosummary::
   :toctree: generated
   :template: class.rst

   pandera.typing.pyspark.DataFrame
   pandera.typing.pyspark.Series
   pandera.typing.pyspark.Index

Modin
*****

.. autosummary::
   :toctree: generated
   :template: class.rst

   pandera.typing.modin.DataFrame
   pandera.typing.modin.Series
   pandera.typing.modin.Index

FastAPI
*******

.. autosummary::
   :toctree: generated
   :template: class.rst

   pandera.typing.fastapi.UploadFile


Serialization Formats
*********************

.. autosummary::
   :toctree: generated
   :template: class.rst

   pandera.typing.formats.Formats
