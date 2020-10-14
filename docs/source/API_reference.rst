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


Schema Models
-------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   SchemaModel

**Model Components**

.. autosummary::
   :toctree: generated
   :nosignatures:

   Field
   check
   dataframe_check

**Typing**

.. autosummary::
   :toctree: generated
   :template: typing_class.rst
   :nosignatures:

   typing.DataFrame
   typing.Series
   typing.Index

**Base Classes**

.. autosummary::
   :toctree: generated
   :template: model_component_class.rst
   :nosignatures:

   model.BaseConfig
   model_components.FieldInfo


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
   check_types


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
