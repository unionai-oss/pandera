"""A flexible and expressive pandas validation library."""

from . import errors, constants, io
from .checks import Check
from .hypotheses import Hypothesis
from .decorators import check_input, check_output
from .dtypes import PandasDtype
from .schemas import DataFrameSchema, SeriesSchema
from .schema_components import Column, Index, MultiIndex
from .schema_inference import infer_schema
from .version import __version__


# pylint: disable=invalid-name
Bool = PandasDtype.Bool
DateTime = PandasDtype.DateTime
Category = PandasDtype.Category
Float = PandasDtype.Float
Float16 = PandasDtype.Float16
Float32 = PandasDtype.Float32
Float64 = PandasDtype.Float64
Int = PandasDtype.Int
Int8 = PandasDtype.Int8
Int16 = PandasDtype.Int16
Int32 = PandasDtype.Int32
Int64 = PandasDtype.Int64
UInt8 = PandasDtype.UInt8
UInt16 = PandasDtype.UInt16
UInt32 = PandasDtype.UInt32
UInt64 = PandasDtype.UInt64
INT8 = PandasDtype.INT8
INT16 = PandasDtype.INT16
INT32 = PandasDtype.INT32
INT64 = PandasDtype.INT64
UINT8 = PandasDtype.UINT8
UINT16 = PandasDtype.UINT16
UINT32 = PandasDtype.UINT32
UINT64 = PandasDtype.UINT64
Object = PandasDtype.Object
String = PandasDtype.String
Timedelta = PandasDtype.Timedelta
