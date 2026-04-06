.. _api-io-utils:

IO Utilities
============

Schema serialization lives in explicit submodules so optional backends are not
imported from ``pandera.io`` itself. Install extras as needed; see
:ref:`installation<installation>`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   pandera.io.pandas_io.from_yaml
   pandera.io.pandas_io.to_yaml
   pandera.io.pandas_io.to_script
   pandera.io.pandas_io.from_json
   pandera.io.pandas_io.to_json
   pandera.io.polars_io.from_yaml
   pandera.io.polars_io.to_yaml
   pandera.io.polars_io.from_json
   pandera.io.polars_io.to_json
   pandera.io.pyspark_sql_io.from_yaml
   pandera.io.pyspark_sql_io.to_yaml
   pandera.io.pyspark_sql_io.from_json
   pandera.io.pyspark_sql_io.to_json
