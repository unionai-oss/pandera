"""A flexible and expressive pandas validation library."""
import platform

from pandera.dtypes import (
    Bool,
    Category,
    Complex,
    Complex64,
    Complex128,
    DataType,
    DateTime,
    Float,
    Float16,
    Float32,
    Float64,
    Int,
    Int8,
    Int16,
    Int32,
    Int64,
    String,
    Timedelta,
    Timestamp,
    UInt,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
)
from pandera.engines.numpy_engine import Object
from pandera.engines.pandas_engine import (
    BOOL,
    INT8,
    INT16,
    INT32,
    INT64,
    PANDAS_1_3_0_PLUS,
    STRING,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
)
from pandera.engines.pandas_engine import _PandasDtype as PandasDtype
from pandera.engines.pandas_engine import pandas_version

from . import constants, errors, pandas_accessor
from .checks import Check
from .decorators import check_input, check_io, check_output, check_types
from .hypotheses import Hypothesis
from .model import SchemaModel
from .model_components import Field, check, dataframe_check
from .schema_components import Column, Index, MultiIndex
from .schema_inference import infer_schema
from .schemas import DataFrameSchema, SeriesSchema
from .version import __version__

if platform.system() != "Windows":
    from pandera.dtypes import Complex256, Float128
