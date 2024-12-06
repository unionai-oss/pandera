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

Passing native pandas dtypes to pandera components is preferred, and will be
converted to the following pandera-native dtypes. See :ref:`here <dtype-validation>`
for more details.

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
   pandera.engines.pandas_engine.Decimal
   pandera.engines.pandas_engine.Category
   pandera.engines.pandas_engine.STRING
   pandera.engines.pandas_engine.NpString
   pandera.engines.pandas_engine.DateTime
   pandera.engines.pandas_engine.Date
   pandera.engines.pandas_engine.Period
   pandera.engines.pandas_engine.Sparse
   pandera.engines.pandas_engine.Interval
   pandera.engines.pandas_engine.PydanticModel
   pandera.engines.pandas_engine.PythonDict
   pandera.engines.pandas_engine.PythonList
   pandera.engines.pandas_engine.PythonTuple
   pandera.engines.pandas_engine.PythonTypedDict
   pandera.engines.pandas_engine.PythonNamedTuple

Pyarrow Dtypes
--------------

*new in 0.20.0*

Pyarrow datatypes are available with the pandas validation engine. Passing
`native Pyarrow dtypes <https://arrow.apache.org/docs/python/api/datatypes.html>`__
are preferred, and will be converted to the following pandera-native dtypes.
See :ref:`here <pyarrow-dtypes>` for more details.

.. autosummary::
   :toctree: generated
   :template: dtype.rst
   :nosignatures:

   pandera.engines.pandas_engine.ArrowBool
   pandera.engines.pandas_engine.ArrowInt64
   pandera.engines.pandas_engine.ArrowInt32
   pandera.engines.pandas_engine.ArrowInt16
   pandera.engines.pandas_engine.ArrowInt8
   pandera.engines.pandas_engine.ArrowString
   pandera.engines.pandas_engine.ArrowUInt64
   pandera.engines.pandas_engine.ArrowUInt32
   pandera.engines.pandas_engine.ArrowUInt16
   pandera.engines.pandas_engine.ArrowUInt8
   pandera.engines.pandas_engine.ArrowFloat64
   pandera.engines.pandas_engine.ArrowFloat32
   pandera.engines.pandas_engine.ArrowFloat16
   pandera.engines.pandas_engine.ArrowDecimal128
   pandera.engines.pandas_engine.ArrowTimestamp
   pandera.engines.pandas_engine.ArrowDictionary
   pandera.engines.pandas_engine.ArrowList
   pandera.engines.pandas_engine.ArrowStruct
   pandera.engines.pandas_engine.ArrowNull
   pandera.engines.pandas_engine.ArrowDate32
   pandera.engines.pandas_engine.ArrowDate64
   pandera.engines.pandas_engine.ArrowDuration
   pandera.engines.pandas_engine.ArrowTime32
   pandera.engines.pandas_engine.ArrowTime64
   pandera.engines.pandas_engine.ArrowTimestamp
   pandera.engines.pandas_engine.ArrowBinary
   pandera.engines.pandas_engine.ArrowLargeBinary
   pandera.engines.pandas_engine.ArrowLargeString


GeoPandas Dtypes
----------------

*new in 0.9.0*

.. autosummary::
   :toctree: generated
   :template: dtype.rst
   :nosignatures:

   pandera.engines.geopandas_engine.Geometry

Pydantic Dtypes
---------------

*new in 0.10.0*

.. autosummary::
   :toctree: generated
   :template: dtype.rst
   :nosignatures:

   pandera.engines.pandas_engine.PydanticModel

.. _polars-dtypes:

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
   pandera.engines.polars_engine.Enum
   pandera.engines.polars_engine.Categorical
   pandera.engines.polars_engine.Category
   pandera.engines.polars_engine.Binary
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
   pandera.engines.polars_engine.Engine
   pandera.engines.pyspark_engine.Engine
