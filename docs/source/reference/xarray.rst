.. _api-xarray:

Xarray
======

Schemas and components for validating :class:`xarray.DataArray` and
:class:`xarray.Dataset`. Typical imports use :mod:`pandera.xarray`; the
implementations live under :mod:`pandera.api.xarray`. See :ref:`xarray-guide`
for usage examples.

Schemas
-------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.api.xarray.container.DataArraySchema
   pandera.api.xarray.container.DatasetSchema

Schema components
-----------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.api.xarray.components.DataVar
   pandera.api.xarray.components.Coordinate

Declarative models
------------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.api.xarray.model.DataArrayModel
   pandera.api.xarray.model.DatasetModel

.. autosummary::
   :toctree: generated
   :nosignatures:

   pandera.api.xarray.model_components.Field

Typing
------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.typing.xarray.Coordinate
   pandera.typing.xarray.DataArray
   pandera.typing.xarray.Dataset


Model configuration
---------------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.api.xarray.model_config.DataArrayConfig
   pandera.api.xarray.model_config.DatasetConfig

Abstract base classes
---------------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.api.xarray.base.BaseDataArraySchema
   pandera.api.xarray.base.BaseDatasetSchema
   pandera.api.xarray.base.BaseDataTreeSchema

Types and helpers
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   pandera.api.xarray.types.XarrayData
   pandera.api.xarray.utils.get_validation_depth
