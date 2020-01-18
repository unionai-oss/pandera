"""Schema datatypes."""
# pylint: disable=C0103

from enum import Enum


class PandasDtype(Enum):
    """Enumerate all valid pandas data types."""

    Bool = "bool"
    DateTime = "datetime64[ns]"
    Category = "category"
    Float = "float64"
    Float16 = "float16"
    Float32 = "float32"
    Float64 = "float64"
    Int = "int64"
    Int8 = "int8"
    Int16 = "int16"
    Int32 = "int32"
    Int64 = "int64"
    UInt8 = "uint8"
    UInt16 = "uint16"
    UInt32 = "uint32"
    UInt64 = "uint64"
    Object = "object"
    # the string datatype doesn't map to a unique string representation and is
    # representated as a numpy object array. This will change after pandas 1.0,
    # but for now will need to handle this as a special case.
    String = "string"
    Timedelta = "timedelta64[ns]"


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
Object = PandasDtype.Object
String = PandasDtype.String
Timedelta = PandasDtype.Timedelta
