.. _api-xarray:

Xarray
======

*New in 0.31.0*

Schemas and components for validating :class:`xarray.DataArray`,
:class:`xarray.Dataset`, and :class:`xarray.DataTree`. Typical imports use
:mod:`pandera.xarray`; implementations live under :mod:`pandera.api.xarray`.
See :ref:`xarray-guide` for usage examples.

The :mod:`pandera.xarray` entry point also re-exports
:class:`~pandera.api.checks.Check`, :class:`~pandera.api.parsers.Parser`, the
:ref:`decorators <api-decorators>` (:func:`~pandera.decorators.check_input`,
etc.), and :mod:`pandera.errors` — see :ref:`api-core` for those APIs.

Schemas
-------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.api.xarray.container.DataArraySchema
   pandera.api.xarray.container.DatasetSchema
   pandera.api.xarray.container.DataTreeSchema

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
   pandera.api.xarray.model.DataTreeModel

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
   pandera.typing.xarray.DataTree
   pandera.typing.xarray.XarrayAnnotationBase

Model configuration
-------------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.api.xarray.model_config.DataArrayConfig
   pandera.api.xarray.model_config.DatasetConfig
   pandera.api.xarray.model_config.DataTreeConfig

Abstract base classes
---------------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.api.xarray.base.BaseDataArraySchema
   pandera.api.xarray.base.BaseDatasetSchema
   pandera.api.xarray.base.BaseDataTreeSchema

Check object types
------------------

Types accepted by xarray :class:`~pandera.api.checks.Check` backends and the
:data:`~pandera.api.xarray.types.XARRAY_CHECK_OBJECT_TYPES` registry tuple.

.. autosummary::
   :toctree: generated
   :nosignatures:

   pandera.api.xarray.types.XarrayData
   pandera.api.xarray.types.XARRAY_CHECK_OBJECT_TYPES
   pandera.api.xarray.types.XarrayCheckObjects

Configuration
-------------

See also :ref:`api-core` for the full configuration API.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.config.ValidationDepth
   pandera.config.ValidationScope

Utilities
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   pandera.api.xarray.utils.get_validation_depth

Schema inference
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   pandera.schema_inference.xarray.infer_schema

IO (YAML / JSON)
----------------

Serialization helpers for xarray schemas (also re-exported from
:mod:`pandera.xarray`).

.. autosummary::
   :toctree: generated
   :nosignatures:

   pandera.io.xarray_io.to_yaml
   pandera.io.xarray_io.from_yaml
   pandera.io.xarray_io.to_json
   pandera.io.xarray_io.from_json

Hypothesis strategies
---------------------

`Hypothesis <https://hypothesis.readthedocs.io/>`_ strategies for generating
``DataArray`` / ``Dataset`` objects that match a schema. Requires the
``strategies`` extra.

.. autosummary::
   :toctree: generated
   :nosignatures:

   pandera.strategies.xarray_strategies.xarray_dtype_strategy
   pandera.strategies.xarray_strategies.data_array_strategy
   pandera.strategies.xarray_strategies.dataset_strategy
   pandera.strategies.xarray_strategies.data_array_schema_strategy
   pandera.strategies.xarray_strategies.dataset_schema_strategy

Dtypes
------

See :ref:`api-dtypes` for :class:`~pandera.engines.xarray_engine.DataType` and
:class:`~pandera.engines.xarray_engine.Engine`.

See also
--------

- :ref:`xarray-guide` — tutorials and examples
- :ref:`api-core` — :class:`~pandera.api.checks.Check`, configuration, errors
- :ref:`api-dtypes` — dtype engines including xarray
- :ref:`api-decorators` — validation decorators
