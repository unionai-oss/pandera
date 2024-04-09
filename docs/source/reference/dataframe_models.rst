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

Model Components
----------------

.. autosummary::
   :toctree: generated

   pandera.api.dataframe.model_components.Field
   pandera.api.dataframe.model_components.check
   pandera.api.dataframe.model_components.dataframe_check
   pandera.api.dataframe.model_components.parse
   pandera.api.dataframe.model_components.dataframe_parse

Typing
------

.. autosummary::
   :toctree: generated
   :template: typing_module.rst
   :nosignatures:

   pandera.typing

Config
------

.. autosummary::
   :toctree: generated
   :template: model_component_class.rst
   :nosignatures:

   pandera.api.pandas.model_config.BaseConfig
   pandera.api.polars.model_config.BaseConfig
   pandera.api.pyspark.model_config.BaseConfig
