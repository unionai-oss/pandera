.. _api-dtypes:

Data Types
==========

Library-agnostic dtypes
-----------------------

.. autosummary::
   :toctree: generated
   :template: dtype.rst
   :nosignatures:

   pandera.dtypes.DataType
   pandera.dtypes.Bool
   pandera.dtypes.Timestamp
   pandera.dtypes.DateTime
   pandera.dtypes.Timedelta
   pandera.dtypes.Category
   pandera.dtypes.Float
   pandera.dtypes.Float16
   pandera.dtypes.Float32
   pandera.dtypes.Float64
   pandera.dtypes.Float128
   pandera.dtypes.Int
   pandera.dtypes.Int8
   pandera.dtypes.Int16
   pandera.dtypes.Int32
   pandera.dtypes.Int64
   pandera.dtypes.UInt
   pandera.dtypes.UInt8
   pandera.dtypes.UInt16
   pandera.dtypes.UInt32
   pandera.dtypes.UInt64
   pandera.dtypes.Complex
   pandera.dtypes.Complex64
   pandera.dtypes.Complex128
   pandera.dtypes.Complex256
   pandera.dtypes.String


Pandas-specific Dtypes
----------------------

Listed here for compatibility with pandera versions < 0.7.
Passing native pandas dtypes to pandera components is preferred.

.. autosummary::
   :toctree: generated
   :template: dtype.rst
   :nosignatures:

   pandera.engines.pandas_engine.BOOL
   pandera.engines.pandas_engine.INT8
   pandera.engines.pandas_engine.INT16
   pandera.engines.pandas_engine.INT32
   pandera.engines.pandas_engine.INT64
   pandera.engines.pandas_engine.UINT8
   pandera.engines.pandas_engine.UINT16
   pandera.engines.pandas_engine.UINT32
   pandera.engines.pandas_engine.UINT64
   pandera.engines.pandas_engine.STRING
   pandera.engines.numpy_engine.Object

GeoPandas-specific Dtypes
-------------------------

*new in 0.9.0*

.. autosummary::
   :toctree: generated
   :template: dtype.rst
   :nosignatures:

   pandera.engines.pandas_engine.Geometry

Utility functions
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   pandera.dtypes.is_subdtype
   pandera.dtypes.is_float
   pandera.dtypes.is_int
   pandera.dtypes.is_uint
   pandera.dtypes.is_complex
   pandera.dtypes.is_numeric
   pandera.dtypes.is_bool
   pandera.dtypes.is_string
   pandera.dtypes.is_datetime
   pandera.dtypes.is_timedelta
   pandera.dtypes.immutable

Engines
-------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.engines.engine.Engine
   pandera.engines.numpy_engine.Engine
   pandera.engines.pandas_engine.Engine


PandasDtype Enum
----------------

.. warning::

   This class deprecated and will be removed from the pandera API in ``0.9.0``

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.engines.pandas_engine.PandasDtype
