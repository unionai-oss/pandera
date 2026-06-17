.. _api-narwhals:

Narwhals Backend
================

*new in 0.32.0*

Opt-in `Narwhals <https://narwhals-dev.github.io/narwhals/>`__-powered
validation backend that powers the Polars, Ibis, and PySpark SQL
integrations behind a single unified code path. Requires the ``narwhals``
extra. Enable with ``PANDERA_USE_NARWHALS_BACKEND=True`` or
:func:`~pandera.set_config`. Backends register lazily on first schema use;
changing the flag at runtime triggers automatic re-registration. See
:ref:`Backend registration <narwhals-backend-registration>` and
:ref:`Narwhals-powered backends <narwhals-backend>` for the user-facing guide.

Data Objects
------------

Objects passed to custom check functions when the Narwhals backend is
active.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.api.narwhals.types.NarwhalsData
   pandera.api.narwhals.types.NarwhalsCheckResult

Backends
--------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.backends.narwhals.base.NarwhalsSchemaBackend
   pandera.backends.narwhals.container.DataFrameSchemaBackend
   pandera.backends.narwhals.components.ColumnBackend
   pandera.backends.narwhals.checks.NarwhalsCheckBackend

Narwhals Dtypes
---------------

.. autosummary::
   :toctree: generated
   :template: dtype.rst
   :nosignatures:

   pandera.engines.narwhals_engine.DataType
   pandera.engines.narwhals_engine.Int8
   pandera.engines.narwhals_engine.Int16
   pandera.engines.narwhals_engine.Int32
   pandera.engines.narwhals_engine.Int64
   pandera.engines.narwhals_engine.UInt8
   pandera.engines.narwhals_engine.UInt16
   pandera.engines.narwhals_engine.UInt32
   pandera.engines.narwhals_engine.UInt64
   pandera.engines.narwhals_engine.Float32
   pandera.engines.narwhals_engine.Float64
   pandera.engines.narwhals_engine.String
   pandera.engines.narwhals_engine.Bool
   pandera.engines.narwhals_engine.Date
   pandera.engines.narwhals_engine.DateTime
   pandera.engines.narwhals_engine.Duration
   pandera.engines.narwhals_engine.Categorical
   pandera.engines.narwhals_engine.List
   pandera.engines.narwhals_engine.Struct

Engine
------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.engines.narwhals_engine.Engine

Built-in Check Expressions
--------------------------

Narwhals expression implementations of the built-in checks, shared by the
Polars, Ibis, and PySpark SQL integrations when the Narwhals backend is
enabled.

.. autosummary::
   :toctree: generated
   :nosignatures:

   pandera.backends.narwhals.builtin_checks.equal_to
   pandera.backends.narwhals.builtin_checks.not_equal_to
   pandera.backends.narwhals.builtin_checks.greater_than
   pandera.backends.narwhals.builtin_checks.greater_than_or_equal_to
   pandera.backends.narwhals.builtin_checks.less_than
   pandera.backends.narwhals.builtin_checks.less_than_or_equal_to
   pandera.backends.narwhals.builtin_checks.in_range
   pandera.backends.narwhals.builtin_checks.isin
   pandera.backends.narwhals.builtin_checks.notin
   pandera.backends.narwhals.builtin_checks.str_matches
   pandera.backends.narwhals.builtin_checks.str_contains
   pandera.backends.narwhals.builtin_checks.str_startswith
   pandera.backends.narwhals.builtin_checks.str_endswith
   pandera.backends.narwhals.builtin_checks.str_length
