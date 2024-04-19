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
   pandera.dtypes.Decimal
   pandera.dtypes.String

Pandas Dtypes
-------------

Listed here for compatibility with pandera versions \< 0.7.
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
   pandera.engines.pandas_engine.DateTime
   pandera.engines.pandas_engine.Date
   pandera.engines.pandas_engine.Decimal
   pandera.engines.pandas_engine.Category

GeoPandas Dtypes
----------------

*new in 0.9.0*

.. autosummary::
   :toctree: generated
   :template: dtype.rst
   :nosignatures:

   pandera.engines.pandas_engine.Geometry

Pydantic Dtypes
---------------

*new in 0.10.0*

.. autosummary::
   :toctree: generated
   :template: dtype.rst
   :nosignatures:

   pandera.engines.pandas_engine.PydanticModel

Polars Dtypes
-------------

*new in 0.19.0*

.. autosummary::
   :toctree: generated
   :template: dtype.rst
   :nosignatures:

   pandera.engines.polars_engine.Int8
   pandera.engines.polars_engine.Int16
   pandera.engines.polars_engine.Int32
   pandera.engines.polars_engine.Int64
   pandera.engines.polars_engine.UInt8
   pandera.engines.polars_engine.UInt16
   pandera.engines.polars_engine.UInt32
   pandera.engines.polars_engine.UInt64
   pandera.engines.polars_engine.Float32
   pandera.engines.polars_engine.Float64
   pandera.engines.polars_engine.Decimal
   pandera.engines.polars_engine.Date
   pandera.engines.polars_engine.DateTime
   pandera.engines.polars_engine.Time
   pandera.engines.polars_engine.Timedelta
   pandera.engines.polars_engine.Array
   pandera.engines.polars_engine.List
   pandera.engines.polars_engine.Struct
   pandera.engines.polars_engine.Bool
   pandera.engines.polars_engine.String
   pandera.engines.polars_engine.Categorical
   pandera.engines.polars_engine.Category
   pandera.engines.polars_engine.Null
   pandera.engines.polars_engine.Object


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
