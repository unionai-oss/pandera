import builtins
import datetime
from dataclasses import field
from typing import Any, List
import numpy as np

import pandera.dtypes_
from pandera.dtypes_ import *
from pandera.engines.engine import Engine
from typing import Any


@immutable(init=True)
class NumpyDataType(DataType):
    type: np.dtype = field(default=np.dtype("object"), repr=False)

    def __post_init__(self):
        object.__setattr__(self, "type", np.dtype(self.type))

    def coerce(self, arr: np.ndarray) -> np.ndarray:
        return arr.astype(self.type)

    def __str__(self) -> str:
        return self.type.name

    def __repr__(self) -> str:
        return f"NumpyDataType({self})"


class NumpyEngine(metaclass=Engine, base_datatype=NumpyDataType):
    @classmethod
    def dtype(cls, data_type: Any) -> "NumpyDataType":
        try:
            return Engine.dtype(cls, data_type)
        except TypeError:
            try:
                np_dtype = np.dtype(data_type).type
            except TypeError:
                raise TypeError(
                    f"data type '{data_type}' not understood by {cls.__name__}."
                ) from None
            try:
                return Engine.dtype(cls, np_dtype)
            except TypeError:
                return NumpyDataType(data_type)


################################################################################
# boolean
################################################################################


@NumpyEngine.register_dtype(equivalents=["bool", bool, np.bool_, Bool, Bool()])
@immutable
class NumpyBool(DisableInitMixin, NumpyDataType, Bool):
    """Numpy representation of a boolean data type."""

    type = np.dtype("bool")


def _build_number_equivalents(
    builtin_name: str, pandera_name: str, sizes: List[int]
) -> None:
    """Return a dict of equivalent builtin, numpy, pandera dtypes
    indexed by size in bit_width."""
    builtin_type = getattr(builtins, builtin_name, None)
    default_np_dtype = np.dtype(builtin_name)
    default_size = int(default_np_dtype.name.replace(builtin_name, ""))

    default_equivalents = [
        # e.g.: np.int64
        np.dtype(builtin_name).type,
        # e.g: pandera.dtypes.Int
        getattr(pandera.dtypes_, pandera_name),
    ]
    if builtin_type:
        default_equivalents.append(builtin_type)

    return {
        bit_width: set(
            (
                # e.g.: numpy.int64
                getattr(np, f"{builtin_name}{bit_width}"),
                # e.g.: pandera.dtypes.Int64
                getattr(pandera.dtypes_, f"{pandera_name}{bit_width}"),
                getattr(pandera.dtypes_, f"{pandera_name}{bit_width}")(),
                # e.g.: pandera.dtypes.Int(64)
                getattr(pandera.dtypes_, pandera_name)(),
            )
        )
        | set(default_equivalents if bit_width == default_size else [])
        for bit_width in sizes
    }


################################################################################
## signed integer
################################################################################

_int_equivalents = _build_number_equivalents(
    builtin_name="int", pandera_name="Int", sizes=[64, 32, 16, 8]
)


@NumpyEngine.register_dtype(equivalents=_int_equivalents[64])
@immutable
class NumpyInt64(DisableInitMixin, NumpyDataType, Int64):
    type = np.dtype("int64")
    bit_width: int = 64


@NumpyEngine.register_dtype(equivalents=_int_equivalents[32])
@immutable
class NumpyInt32(NumpyInt64):
    type = np.dtype("int32")
    bit_width: int = 32


@NumpyEngine.register_dtype(equivalents=_int_equivalents[16])
@immutable
class NumpyInt16(NumpyInt32):
    type = np.dtype("int16")
    bit_width: int = 16


@NumpyEngine.register_dtype(equivalents=_int_equivalents[8])
@immutable
class NumpyInt8(NumpyInt16):
    type = np.dtype("int8")
    bit_width: int = 8


################################################################################
## unsigned integer
################################################################################

_uint_equivalents = _build_number_equivalents(
    builtin_name="uint",
    pandera_name="UInt",
    sizes=[64, 32, 16, 8],
)


@NumpyEngine.register_dtype(equivalents=_uint_equivalents[64])
@immutable
class NumpyUInt64(DisableInitMixin, NumpyDataType, UInt64):
    type = np.dtype("uint64")
    bit_width: int = 64


@NumpyEngine.register_dtype(equivalents=_uint_equivalents[32])
@immutable
class NumpyUInt32(NumpyUInt64):
    type = np.dtype("uint32")
    bit_width: int = 32


@NumpyEngine.register_dtype(equivalents=_uint_equivalents[16])
@immutable
class NumpyUInt16(NumpyUInt32):
    type = np.dtype("uint16")
    bit_width: int = 16


@NumpyEngine.register_dtype(equivalents=_uint_equivalents[8])
@immutable
class NumpyUInt8(NumpyUInt16):
    type = np.dtype("uint8")
    bit_width: int = 8


################################################################################
## float
################################################################################

_float_equivalents = _build_number_equivalents(
    builtin_name="float",
    pandera_name="Float",
    sizes=[128, 64, 32, 16],
)


@NumpyEngine.register_dtype(equivalents=_float_equivalents[128])
@immutable
class NumpyFloat128(DisableInitMixin, NumpyDataType, Float128):
    type = np.dtype("float128")
    bit_width: int = 128


@NumpyEngine.register_dtype(equivalents=_float_equivalents[64])
@immutable
class NumpyFloat64(NumpyFloat128):
    type = np.dtype("float64")
    bit_width: int = 64


@NumpyEngine.register_dtype(equivalents=_float_equivalents[32])
@immutable
class NumpyFloat32(NumpyFloat64):
    type = np.dtype("float32")
    bit_width: int = 32


@NumpyEngine.register_dtype(equivalents=_float_equivalents[16])
@immutable
class NumpyFloat16(NumpyFloat32):
    type = np.dtype("float16")
    bit_width: int = 16


################################################################################
## complex
################################################################################

_complex_equivalents = _build_number_equivalents(
    builtin_name="complex",
    pandera_name="Complex",
    sizes=[256, 128, 64],
)


@NumpyEngine.register_dtype(equivalents=_complex_equivalents[256])
@immutable
class NumpyComplex256(DisableInitMixin, NumpyDataType, Complex256):
    type = np.dtype("complex256")
    bit_width: int = 256


@NumpyEngine.register_dtype(equivalents=_complex_equivalents[128])
@immutable
class NumpyComplex128(NumpyComplex256):
    type = np.dtype("complex128")
    bit_width: int = 128


@NumpyEngine.register_dtype(equivalents=_complex_equivalents[64])
@immutable
class NumpyComplex64(NumpyComplex128):
    type = np.dtype("complex64")
    bit_width: int = 64


################################################################################
# string
################################################################################


@NumpyEngine.register_dtype(equivalents=["str", "string", str, np.str_])
@immutable
class NumpyString(DisableInitMixin, NumpyDataType, String):
    type = np.dtype("str")

    def coerce(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(object)
        notna = ~arr.isna()
        arr[notna] = arr[notna].astype(str)
        return arr

    def check(self, datatype: "DataType") -> bool:
        return isinstance(datatype, (NumpyObject, type(self)))


################################################################################
# object
################################################################################


@NumpyEngine.register_dtype(equivalents=["object", "O", object, np.object_])
@immutable
class NumpyObject(DisableInitMixin, NumpyDataType):
    type = np.dtype("object")


Object = NumpyObject

################################################################################
# time
################################################################################


@NumpyEngine.register_dtype(
    equivalents=[
        datetime.datetime,
        np.datetime64,
        Timestamp,
        Timestamp(),
    ]
)
@immutable
class NumpyDateTime64(DisableInitMixin, NumpyDataType, Timestamp):
    type = np.dtype("datetime64")


@NumpyEngine.register_dtype(
    equivalents=[
        datetime.datetime,
        np.timedelta64,
        Timedelta,
        Timedelta(),
    ]
)
@immutable
class NumpyTimedelta64(DisableInitMixin, NumpyDataType, Timedelta):
    type = np.dtype("timedelta64")
