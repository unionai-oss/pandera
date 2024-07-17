.. _api-core:

Core
====

Schemas
-------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.api.pandas.container.DataFrameSchema
   pandera.api.pandas.array.SeriesSchema
   pandera.api.polars.container.DataFrameSchema
   pandera.api.pyspark.container.DataFrameSchema
   pandera.api.dataframe.container.DataFrameSchema

Schema Components
-----------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.api.pandas.components.Column
   pandera.api.pandas.components.Index
   pandera.api.pandas.components.MultiIndex
   pandera.api.polars.components.Column
   pandera.api.pyspark.components.Column
   pandera.api.dataframe.components.ComponentSchema

Checks
------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.api.checks.Check
   pandera.api.hypotheses.Hypothesis

Data Objects
------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.api.polars.types.PolarsData
   pandera.api.pyspark.types.PysparkDataframeColumnObject

Configuration
-------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.config.PanderaConfig
   pandera.config.ValidationDepth
   pandera.config.ValidationScope
   pandera.config.config_context
   pandera.config.get_config_context
