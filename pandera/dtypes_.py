import dataclasses
import functools
from typing import Any, Tuple, Type


class DataType:
    def __init__(self):
        if self.__class__ is DataType:
            raise TypeError(
                f"{self.__class__.__name__} may not be instantiated."
            )

    def __call__(self, obj: Any):
        """Coerce object to the dtype."""
        return self.coerce(obj)

    def coerce(self, obj: Any):
        """Coerce object to the dtype."""
        raise NotImplementedError()

    def check(self, pandera_dtype: "DataType") -> bool:
        if not isinstance(pandera_dtype, DataType):
            return False
        return self == pandera_dtype

    def __repr__(self) -> str:
        return f"DataType({str(self)})"

    def __str__(self) -> str:
        """Must be implemented by subclasses."""
        raise NotImplementedError()

    def __hash__(self) -> int:
        raise NotImplementedError()


def immutable(
    dtype: Type[DataType] = None, **dataclass_kwargs: Any
) -> Type[DataType]:
    """:func:`dataclasses.dataclass` decorator with different default values:
    `frozen=True`, `init=False`, `repr=False`.

    In addition, `init=False` disables inherited `__init__` method to ensure
    the DataType's default attributes are not altered during initialization.

    :param dtype: :class:`DataType` to decorate.
    :param dataclass_kwargs: Keywords arguments forwarded to
        :func:`dataclasses.dataclass`.
    :returns: Immutable :class:`DataType`
    """
    kwargs = {"frozen": True, "init": False, "repr": False}
    kwargs.update(dataclass_kwargs)

    def _wrapper(dtype):
        immutable_dtype = dataclasses.dataclass(**kwargs)(dtype)
        if not kwargs["init"]:

            def __init__(self):
                pass

            # delattr(immutable_dtype, "__init__") doesn't work because
            # super.__init__ would still exist.
            setattr(immutable_dtype, "__init__", __init__)

        return immutable_dtype

    if dtype is None:
        return _wrapper

    return _wrapper(dtype)


################################################################################
# boolean
################################################################################


@immutable
class Bool(DataType):
    """Semantic representation of a boolean data type."""

    def __str__(self) -> str:
        return "bool"


Boolean = Bool

################################################################################
# number
################################################################################


@immutable
class _Number(DataType):
    continuous: bool = None
    exact: bool = None

    def check(self, datatype: "DataType") -> bool:
        if self.__class__ is _Number:
            return isinstance(datatype, (Int, Float, Complex))
        return super().check(datatype)


@immutable
class _PhysicalNumber(_Number):
    bit_width: int = None
    _base_name: str = dataclasses.field(default=None, init=False, repr=False)

    def __eq__(self, obj: object) -> bool:
        if isinstance(obj, type(self)):
            return obj.bit_width == self.bit_width
        return super().__eq__(obj)

    def __str__(self) -> str:
        return f"{self._base_name}{self.bit_width}"


################################################################################
## signed integer
################################################################################


@immutable(eq=False)
class Int(_PhysicalNumber):
    _base_name = "int"
    continuous = False
    exact = True
    bit_width = 64
    signed: bool = dataclasses.field(default=True, init=False)


@immutable
class Int64(Int, _PhysicalNumber):
    bit_width = 64


@immutable
class Int32(Int64):
    bit_width = 32


@immutable
class Int16(Int32):
    bit_width = 16


@immutable
class Int8(Int16):
    bit_width = 8


################################################################################
## unsigned integer
################################################################################


@immutable
class UInt(Int):
    _base_name = "uint"
    signed: bool = dataclasses.field(default=False, init=False)


@immutable
class UInt64(UInt):
    bit_width = 64


@immutable
class UInt32(UInt64):
    bit_width = 32


@immutable
class UInt16(UInt32):
    bit_width = 16


@immutable
class UInt8(UInt16):
    bit_width = 8


################################################################################
## float
################################################################################


@immutable(eq=False)
class Float(_PhysicalNumber):
    _base_name = "float"
    continuous = True
    exact = False
    bit_width = 64


@immutable
class Float128(Float):
    bit_width = 128


@immutable
class Float64(Float128):
    bit_width = 64


@immutable
class Float32(Float64):
    bit_width = 32


@immutable
class Float16(Float32):
    bit_width = 16


################################################################################
## complex
################################################################################


@immutable(eq=False)
class Complex(_PhysicalNumber):
    _base_name = "complex"
    bit_width = 128


@immutable
class Complex256(Complex):
    bit_width = 256


@immutable
class Complex128(Complex):
    bit_width = 128


@immutable
class Complex64(Complex128):
    bit_width = 64


################################################################################
# nominal
################################################################################


@immutable(init=True)
class Category(DataType):
    categories: Tuple[Any] = None  # immutable sequence to ensure safe hash
    ordered: bool = False

    def __post_init__(self) -> "Category":
        if self.categories is not None and not isinstance(
            self.categories, tuple
        ):
            object.__setattr__(self, "categories", tuple(self.categories))

    def check(self, datatype: "DataType") -> bool:
        if (
            isinstance(datatype, Category)
            and self.categories is None
            or datatype.categories is None
        ):
            # Category without categories is a superset of any Category
            # Allow end-users to not list categories when validating.
            return True

        return super().check(datatype)

    def __str__(self) -> str:
        return "category"


@immutable
class String(DataType):
    def __str__(self) -> str:
        return "string"


################################################################################
# time
################################################################################


@immutable
class Date(DataType):
    def __str__(self) -> str:
        return "date"


@immutable
class Timestamp(Date):
    def __str__(self) -> str:
        return "timestamp"


DateTime = Timestamp


@immutable
class Timedelta(DataType):
    def __str__(self) -> str:
        return "timedelta"
