% pandera package index documentation toctree

```{eval-rst}
.. currentmodule:: pandera
```

# API

## DataFrames

```{eval-rst}
.. list-table::
   :widths: 30 70

   * - :ref:`Core <api-core>`
     - The core objects for defining pandera schemas
   * - :ref:`GeoPandas <api-geopandas>`
     - ``GeoDataFrameSchema`` and ``GeoDataFrameModel`` entry point
   * - :ref:`Data Types <api-dtypes>`
     - Data types for type checking and coercion.
   * - :ref:`DataFrame Models <api-dataframe-models>`
     - Alternative class-based API for defining types for tabular/array-like data.
   * - :ref:`Decorators <api-decorators>`
     - Decorators for integrating pandera schemas with python functions.
   * - :ref:`Schema Inference <api-schema-inference>`
     - Bootstrap schemas from real data
   * - :ref:`IO Utilities <api-io-utils>`
     - Utility functions for reading/writing schemas
   * - :ref:`Data Synthesis Strategies <api-strategies>`
     - Module of functions for generating data from schemas.
   * - :ref:`Extensions <api-extensions>`
     - Utility functions for extending pandera functionality
   * - :ref:`Errors <api-errors>`
     - Pandera-specific exceptions
```

## Configuration

```{eval-rst}
.. list-table::
   :widths: 30 70

   * - :class:`~pandera.config.PanderaConfig`
     - Global configuration (validation enabled, depth, caching)
   * - :class:`~pandera.config.ValidationDepth`
     - Whether to apply checks at schema-level, data-level, or both
   * - :class:`~pandera.config.ValidationScope`
     - Indicates whether a check operates at schema or data level
   * - :func:`~pandera.config.config_context`
     - Context manager to temporarily override config settings
   * - :func:`~pandera.config.get_config_context`
     - Get the current configuration context
   * - :func:`~pandera.config.get_config_global`
     - Get the global configuration
   * - :func:`~pandera.config.reset_config_context`
     - Reset context configuration to the global default
```

See :ref:`api-core` for full details.

## Multi-dimensional arrays

```{eval-rst}
.. list-table::
   :widths: 30 70

   * - :ref:`Xarray <api-xarray>`
     - Schemas for labelled N-dimensional :mod:`xarray` arrays, datasets, and datatrees
   * - :ref:`PyTorch <api-pytorch>`
     - Schemas for :mod:`tensordict` TensorDict and tensorclass objects
```

```{toctree}
:hidden: true

core
geopandas
dtypes
dataframe_models
decorators
schema_inference
io
strategies
extensions
errors
xarray
pytorch
```
