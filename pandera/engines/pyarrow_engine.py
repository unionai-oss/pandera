# pylint: disable=cyclic-import,unexpected-keyword-arg,no-value-for-parameter
"""Pyarrow data types for the pandas type engine."""

import dataclasses
import datetime
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import pandas as pd
import pyarrow

from pandera import dtypes
from pandera.dtypes import immutable
from pandera.engines.pandas_engine import Engine, DataType, BOOL
from pandera.engines.type_aliases import PandasObject


class ArrowDataType(DataType):
    """Base `DataType` for boxing Pandas Arrow data types."""

    def coerce_value(self, value: Any) -> Any:
        """Coerce a value to a particular type."""
        return pyarrow.scalar(
            value,
            type=(
                self.type.pyarrow_dtype  # pylint: disable=E1101
                if self.type
                else None
            ),
        )


@Engine.register_dtype(
    equivalents=[
        "bool[pyarrow]",
        pyarrow.bool_,
        pd.ArrowDtype(pyarrow.bool_()),
    ]
)
@immutable
class ArrowBool(ArrowDataType, BOOL):
    """Semantic representation of a :class:`pyarrow.bool_`."""

    type = pd.ArrowDtype(pyarrow.bool_())


@Engine.register_dtype(
    equivalents=[
        "int64[pyarrow]",
        pyarrow.int64,
        pd.ArrowDtype(pyarrow.int64()),
    ]
)
@immutable
class ArrowInt64(ArrowDataType, dtypes.Int):
    """Semantic representation of a :class:`pyarrow.int64`."""

    type = pd.ArrowDtype(pyarrow.int64())
    bit_width: int = 64


@Engine.register_dtype(
    equivalents=[
        "int32[pyarrow]",
        pyarrow.int32,
        pd.ArrowDtype(pyarrow.int32()),
    ]
)
@immutable
class ArrowInt32(ArrowInt64):
    """Semantic representation of a :class:`pyarrow.int32`."""

    type = pd.ArrowDtype(pyarrow.int32())
    bit_width: int = 32


@Engine.register_dtype(
    equivalents=[
        "int16[pyarrow]",
        pyarrow.int16,
        pd.ArrowDtype(pyarrow.int16()),
    ]
)
@immutable
class ArrowInt16(ArrowInt32):
    """Semantic representation of a :class:`pyarrow.int16`."""

    type = pd.ArrowDtype(pyarrow.int16())
    bit_width: int = 16


@Engine.register_dtype(
    equivalents=[
        "int8[pyarrow]",
        pyarrow.int8,
        pd.ArrowDtype(pyarrow.int8()),
    ]
)
@immutable
class ArrowInt8(ArrowInt16):
    """Semantic representation of a :class:`pyarrow.int8`."""

    type = pd.ArrowDtype(pyarrow.int8())
    bit_width: int = 8


@Engine.register_dtype(
    equivalents=[
        pyarrow.string,
        pyarrow.utf8,
        pd.ArrowDtype(pyarrow.string()),
        pd.ArrowDtype(pyarrow.utf8()),
    ]
)
@immutable
class ArrowString(ArrowDataType, dtypes.String):
    """Semantic representation of a :class:`pyarrow.string`."""

    type = pd.ArrowDtype(pyarrow.string())


@Engine.register_dtype(
    equivalents=[
        "uint64[pyarrow]",
        pyarrow.uint64,
        pd.ArrowDtype(pyarrow.uint64()),
    ]
)
@immutable
class ArrowUInt64(ArrowDataType, dtypes.UInt):
    """Semantic representation of a :class:`pyarrow.uint64`."""

    type = pd.ArrowDtype(pyarrow.uint64())
    bit_width: int = 64


@Engine.register_dtype(
    equivalents=[
        "uint32[pyarrow]",
        pyarrow.uint32,
        pd.ArrowDtype(pyarrow.uint32()),
    ]
)
@immutable
class ArrowUInt32(ArrowUInt64):
    """Semantic representation of a :class:`pyarrow.uint32`."""

    type = pd.ArrowDtype(pyarrow.uint32())
    bit_width: int = 32


@Engine.register_dtype(
    equivalents=[
        "uint16[pyarrow]",
        pyarrow.uint16,
        pd.ArrowDtype(pyarrow.uint16()),
    ]
)
@immutable
class ArrowUInt16(ArrowUInt32):
    """Semantic representation of a :class:`pyarrow.uint16`."""

    type = pd.ArrowDtype(pyarrow.uint16())
    bit_width: int = 16


@Engine.register_dtype(
    equivalents=[
        "uint8[pyarrow]",
        pyarrow.uint8,
        pd.ArrowDtype(pyarrow.uint8()),
    ]
)
@immutable
class ArrowUInt8(ArrowUInt16):
    """Semantic representation of a :class:`pyarrow.uint8`."""

    type = pd.ArrowDtype(pyarrow.uint8())
    bit_width: int = 8


@Engine.register_dtype(
    equivalents=[
        "double[pyarrow]",
        pyarrow.float64,
        pd.ArrowDtype(pyarrow.float64()),
    ]
)
@immutable
class ArrowFloat64(ArrowDataType, dtypes.Float):
    """Semantic representation of a :class:`pyarrow.float64`."""

    type = pd.ArrowDtype(pyarrow.float64())
    bit_width: int = 64


@Engine.register_dtype(
    equivalents=[
        "float[pyarrow]",
        pyarrow.float32,
        pd.ArrowDtype(pyarrow.float32()),
    ]
)
@immutable
class ArrowFloat32(ArrowFloat64):
    """Semantic representation of a :class:`pyarrow.float32`."""

    type = pd.ArrowDtype(pyarrow.float32())
    bit_width: int = 32


@Engine.register_dtype(
    equivalents=[
        "halffloat[pyarrow]",
        pyarrow.float16,
        pd.ArrowDtype(pyarrow.float16()),
    ]
)
@immutable
class ArrowFloat16(ArrowFloat32):
    """Semantic representation of a :class:`pyarrow.float16`."""

    type = pd.ArrowDtype(pyarrow.float16())
    bit_width: int = 16


@Engine.register_dtype(
    equivalents=[pyarrow.decimal128, pyarrow.Decimal128Type]
)
@immutable(init=True)
class ArrowDecimal128(ArrowDataType, dtypes.Decimal):
    """Semantic representation of a :class:`pyarrow.decimal128`."""

    type: Optional[pd.ArrowDtype] = dataclasses.field(default=None, init=False)
    precision: int = 28
    scale: int = 0

    def __post_init__(self):
        type_ = pd.ArrowDtype(pyarrow.decimal128(self.precision, self.scale))
        object.__setattr__(self, "type", type_)

    @classmethod
    def from_parametrized_dtype(
        cls,
        pyarrow_dtype: pyarrow.Decimal128Type,
    ):
        return cls(precision=pyarrow_dtype.precision, scale=pyarrow_dtype.scale)  # type: ignore


@Engine.register_dtype(equivalents=[pyarrow.timestamp, pyarrow.TimestampType])
@immutable(init=True)
class ArrowTimestamp(ArrowDataType, dtypes.Timestamp):
    """Semantic representation of a :class:`pyarrow.timestamp`."""

    type: Optional[pd.ArrowDtype] = dataclasses.field(default=None, init=False)
    unit: Optional[str] = "ns"
    tz: Optional[datetime.tzinfo] = None

    def __post_init__(self):
        type_ = pd.ArrowDtype(pyarrow.timestamp(self.unit, self.tz))
        object.__setattr__(self, "type", type_)

    @classmethod
    def from_parametrized_dtype(cls, pyarrow_dtype: pyarrow.TimestampType):
        return cls(unit=pyarrow_dtype.unit, tz=pyarrow_dtype.tz)  # type: ignore


@Engine.register_dtype(
    equivalents=[pyarrow.dictionary, pyarrow.DictionaryType]
)
@immutable(init=True)
class ArrowDictionary(ArrowDataType):
    """Semantic representation of a :class:`pyarrow.dictionary`."""

    type: Optional[pd.ArrowDtype] = dataclasses.field(default=None, init=False)
    index_type: Optional[pyarrow.DataType] = pyarrow.int64()
    value_type: Optional[pyarrow.DataType] = pyarrow.int64()
    ordered: bool = False

    def __post_init__(self):
        type_ = pd.ArrowDtype(
            pyarrow.dictionary(
                self.index_type,
                self.value_type,
                self.ordered,
            )
        )
        object.__setattr__(self, "type", type_)

    @classmethod
    def from_parametrized_dtype(cls, pyarrow_dtype: pyarrow.DictionaryType):
        return cls(
            index_type=pyarrow_dtype.index_type,  # type: ignore
            value_type=pyarrow_dtype.value_type,  # type: ignore
            ordered=pyarrow_dtype.ordered,  # type: ignore
        )


@Engine.register_dtype(
    equivalents=[
        pyarrow.list_,
        pyarrow.ListType,
        pyarrow.FixedSizeListType,
    ]
)
@immutable(init=True)
class ArrowList(ArrowDataType):
    """Semantic representation of a :class:`pyarrow.list_`."""

    type: Optional[pd.ArrowDtype] = dataclasses.field(default=None, init=False)
    value_type: Optional[Union[pyarrow.DataType, pyarrow.Field]] = (
        pyarrow.string()
    )
    list_size: Optional[int] = -1

    def __post_init__(self):
        type_ = pd.ArrowDtype(pyarrow.list_(self.value_type, self.list_size))
        object.__setattr__(self, "type", type_)

    @classmethod
    def from_parametrized_dtype(
        cls,
        pyarrow_dtype: Union[pyarrow.ListType, pyarrow.FixedSizeListType],
    ):
        try:
            _dtype = cls(
                value_type=pyarrow_dtype.value_type,  # type: ignore
                list_size=pyarrow_dtype.list_size,  # type: ignore
            )
        except AttributeError:
            _dtype = cls(value_type=pyarrow_dtype.value_type)  # type: ignore
        return _dtype


@Engine.register_dtype(equivalents=[pyarrow.struct, pyarrow.StructType])
@immutable(init=True)
class ArrowStruct(ArrowDataType):
    """Semantic representation of a :class:`pyarrow.struct`."""

    type: Optional[pd.ArrowDtype] = dataclasses.field(default=None, init=False)
    fields: Optional[
        Union[
            Iterable[Union[pyarrow.Field, Tuple[str, pyarrow.DataType]]],
            Dict[str, pyarrow.DataType],
        ]
    ] = tuple()

    def __post_init__(self):
        type_ = pd.ArrowDtype(pyarrow.struct(self.fields))
        object.__setattr__(self, "type", type_)

    @classmethod
    def from_parametrized_dtype(cls, pyarrow_dtype: pyarrow.StructType):
        return cls(
            fields=[pyarrow_dtype.field(i) for i in range(pyarrow_dtype.num_fields)]  # type: ignore
        )


@Engine.register_dtype(
    equivalents=[
        "null[pyarrow]",
        pyarrow.null,
        pd.ArrowDtype(pyarrow.null()),
    ]
)
@immutable
class ArrowNull(ArrowDataType):
    """Semantic representation of a :class:`pyarrow.null`."""

    type = pd.ArrowDtype(pyarrow.null())


@Engine.register_dtype(
    equivalents=[
        "date32[day][pyarrow]",
        pyarrow.date32,
        pd.ArrowDtype(pyarrow.date32()),
    ]
)
@immutable
class ArrowDate32(ArrowDataType, dtypes.Date):
    """Semantic representation of a :class:`pyarrow.date32`."""

    type = pd.ArrowDtype(pyarrow.date32())


@Engine.register_dtype(
    equivalents=[
        "date64[ms][pyarrow]",
        pyarrow.date64,
        pd.ArrowDtype(pyarrow.date64()),
    ]
)
@immutable
class ArrowDate64(ArrowDataType, dtypes.Date):
    """Semantic representation of a :class:`pyarrow.date64`."""

    type = pd.ArrowDtype(pyarrow.date64())


@Engine.register_dtype(equivalents=[pyarrow.duration, pyarrow.DurationType])
@immutable(init=True)
class ArrowDuration(ArrowDataType):
    """Semantic representation of a :class:`pyarrow.duration`."""

    type: Optional[pd.ArrowDtype] = dataclasses.field(default=None, init=False)
    unit: Optional[str] = "ns"

    def __post_init__(self):
        type_ = pd.ArrowDtype(pyarrow.duration(self.unit))
        object.__setattr__(self, "type", type_)

    @classmethod
    def from_parametrized_dtype(cls, pyarrow_dtype: pyarrow.DurationType):
        return cls(unit=pyarrow_dtype.unit)  # type: ignore


@Engine.register_dtype(equivalents=[pyarrow.time32, pyarrow.Time32Type])
@immutable(init=True)
class ArrowTime32(ArrowDataType):
    """Semantic representation of a :class:`pyarrow.time32`."""

    type: Optional[pd.ArrowDtype] = dataclasses.field(default=None, init=False)
    unit: Optional[str] = "ms"

    def __post_init__(self):
        type_ = pd.ArrowDtype(pyarrow.time32(self.unit))
        object.__setattr__(self, "type", type_)

    @classmethod
    def from_parametrized_dtype(cls, pyarrow_dtype: pyarrow.Time32Type):
        return cls(unit=pyarrow_dtype.unit)  # type: ignore

    def coerce(self, data_container: PandasObject) -> PandasObject:
        if data_container.dtype == self.type:
            return data_container
        else:
            return data_container.astype(
                pd.ArrowDtype(pyarrow.int32())
            ).astype(self.type)


@Engine.register_dtype(equivalents=[pyarrow.time64, pyarrow.Time64Type])
@immutable(init=True)
class ArrowTime64(ArrowDataType):
    """Semantic representation of a :class:`pyarrow.time64`."""

    type: Optional[pd.ArrowDtype] = dataclasses.field(default=None, init=False)
    unit: Optional[str] = "ns"

    def __post_init__(self):
        type_ = pd.ArrowDtype(pyarrow.time64(self.unit))
        object.__setattr__(self, "type", type_)

    @classmethod
    def from_parametrized_dtype(cls, pyarrow_dtype: pyarrow.Time64Type):
        return cls(unit=pyarrow_dtype.unit)  # type: ignore

    def coerce(self, data_container: PandasObject) -> PandasObject:
        if data_container.dtype == self.type:
            return data_container
        else:
            return data_container.astype(
                pd.ArrowDtype(pyarrow.int64())
            ).astype(self.type)


@Engine.register_dtype(equivalents=[pyarrow.map_, pyarrow.MapType])
@immutable(init=True)
class ArrowMap(ArrowDataType):
    """Semantic representation of a :class:`pyarrow.map_`."""

    type: Optional[pd.ArrowDtype] = dataclasses.field(default=None, init=False)
    key_type: Optional[pyarrow.DataType] = pyarrow.int64()
    item_type: Optional[pyarrow.DataType] = pyarrow.int64()
    keys_sorted: bool = False

    def __post_init__(self):
        type_ = pd.ArrowDtype(
            pyarrow.map_(
                self.key_type,
                self.item_type,
                self.keys_sorted,
            )
        )
        object.__setattr__(self, "type", type_)

    @classmethod
    def from_parametrized_dtype(cls, pyarrow_dtype: pyarrow.MapType):
        return cls(
            key_type=pyarrow_dtype.key_type,  # type: ignore
            item_type=pyarrow_dtype.item_type,  # type: ignore
            keys_sorted=pyarrow_dtype.keys_sorted,  # type: ignore
        )


@Engine.register_dtype(
    equivalents=[
        "binary[pyarrow]",
        pyarrow.binary,
        pyarrow.FixedSizeBinaryType,
        pd.ArrowDtype(pyarrow.binary()),
    ]
)
@immutable(init=True)
class ArrowBinary(ArrowDataType, dtypes.Binary):
    """Semantic representation of a :class:`pyarrow.binary`."""

    type: Optional[pd.ArrowDtype] = dataclasses.field(default=None, init=False)
    length: Optional[int] = -1

    def __post_init__(self):
        type_ = pd.ArrowDtype(pyarrow.binary(self.length))
        object.__setattr__(self, "type", type_)

    @classmethod
    def from_parametrized_dtype(
        cls,
        pyarrow_dtype: Union[pyarrow.DataType, pyarrow.FixedSizeBinaryType],
    ):
        try:
            _dtype = cls(length=pyarrow_dtype.byte_width)  # type: ignore
        except (ValueError, AttributeError):
            _dtype = cls()  # type: ignore
        return _dtype


@Engine.register_dtype(
    equivalents=[
        "large_binary[pyarrow]",
        pyarrow.large_binary,
        pd.ArrowDtype(pyarrow.large_binary()),
    ]
)
@immutable
class ArrowLargeBinary(ArrowDataType):
    """Semantic representation of a :class:`pyarrow.large_binary`."""

    type = pd.ArrowDtype(pyarrow.large_binary())


@Engine.register_dtype(
    equivalents=[
        "large_string[pyarrow]",
        pyarrow.large_string,
        pyarrow.large_utf8,
        pd.ArrowDtype(pyarrow.large_string()),
        pd.ArrowDtype(pyarrow.large_utf8()),
    ]
)
@immutable
class ArrowLargeString(ArrowDataType, dtypes.String):
    """Semantic representation of a :class:`pyarrow.large_string`."""

    type = pd.ArrowDtype(pyarrow.large_string())
