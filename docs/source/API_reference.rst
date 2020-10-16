.. pandera package index documentation toctree

.. currentmodule:: pandera

API Reference
=============

Schemas
-------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   DataFrameSchema
   SeriesSchema


Schema Components
-----------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   Column
   Index
   MultiIndex


Checks
------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   Check
   Hypothesis


Pandas Data Types
-----------------

.. autosummary::
   :toctree: generated
   :template: pandas_dtype_class.rst
   :nosignatures:

   PandasDtype


Decorators
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   check_input
   check_output
   check_io


Schema Inference
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   infer_schema


IO Utils
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   io.from_yaml
   io.to_yaml
   io.to_script


Errors
------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   errors.SchemaError
   errors.SchemaErrors
   errors.SchemaInitError
   errors.SchemaDefinitionError
