% pandera package index documentation toctree

```{eval-rst}
.. currentmodule:: pandera
```

# API

```{eval-rst}
.. list-table::
   :widths: 30 70

   * - :ref:`Core <api-core>`
     - The core objects for defining pandera schemas
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

```{toctree}
:hidden: true

core
dtypes
dataframe_models
decorators
schema_inference
io
strategies
extensions
errors
```
