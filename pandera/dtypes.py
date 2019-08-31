"""Schema datatypes."""

from enum import Enum


class PandasDtype(Enum):
    Bool = "bool"
    DateTime = "datetime64[ns]"
    Category = "category"
    Float = "float64"
    Int = "int64"
    Object = "object"
    String = "object"
    Timedelta = "timedelta64[ns]"


Bool = PandasDtype.Bool
DateTime = PandasDtype.DateTime
Category = PandasDtype.Category
Float = PandasDtype.Float
Int = PandasDtype.Int
Object = PandasDtype.Object
String = PandasDtype.String
Timedelta = PandasDtype.Timedelta
