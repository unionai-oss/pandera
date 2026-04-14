.. _api-pytorch:

PyTorch
=======

*New in 0.32.0*

Schemas and components for validating :class:`tensordict.TensorDict` and
:class:`tensordict.tensorclass`. Typical imports use :mod:`pandera.tensordict`;
implementations live under :mod:`pandera.api.tensordict`.
See :ref:`pytorch-guide` for usage examples.

The :mod:`pandera.tensordict` entry point also re-exports
:class:`~pandera.api.checks.Check` and :mod:`pandera.errors` —
see :ref:`api-core` for those APIs.

Schemas
-------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.api.tensordict.container.TensorDictSchema

Schema components
-----------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.api.tensordict.components.Tensor

Declarative models
------------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.api.tensordict.model.TensorDictModel

.. autosummary::
   :toctree: generated
   :nosignatures:

   pandera.api.tensordict.model_components.Field

Typing
------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.typing.tensordict.TensorDict
   pandera.typing.tensordict.Tensorclass

Abstract base classes
---------------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.api.tensordict.base.BaseTensorDictSchema

Check object types
------------------

Types accepted by TensorDict :class:`~pandera.api.checks.Check` backends.

.. autosummary::
   :toctree: generated
   :nosignatures:

   pandera.api.tensordict.types.TensorDictData
   pandera.api.tensordict.types.TENSORDICT_CHECK_OBJECT_TYPES

Configuration
-------------

See also :ref:`api-core` for the full configuration API.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.config.ValidationDepth
   pandera.config.ValidationScope

Dtypes
------

See :ref:`api-dtypes` for :class:`~pandera.engines.tensordict_engine.DataType` and
:class:`~pandera.engines.tensordict_engine.Engine`.

See also
--------

- :ref:`pytorch-guide` — tutorials and examples
- :ref:`api-core` — :class:`~pandera.api.checks.Check`, configuration, errors
- :ref:`api-dtypes` — dtype engines including PyTorch
- :ref:`api-decorators` — validation decorators