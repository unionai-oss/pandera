"""Common typing functionality."""

import copy
import inspect
import typing
from typing import (  # type: ignore[attr-defined]
    TYPE_CHECKING,
    Any,
    Generic,
    Optional,
    TypeVar,
    Union,
    _GenericAlias,
    get_args,
    get_origin,
    overload,
)

import typing_inspect

from pandera import dtypes, errors

Bool = dtypes.Bool  #: ``"bool"`` numpy dtype
Date = dtypes.Date  #: ``datetime.date`` object dtype
DateTime = dtypes.DateTime  #: ``"datetime64[ns]"`` numpy dtype
Decimal = dtypes.Decimal  #: ``decimal.Decimal`` object dtype
Timedelta = dtypes.Timedelta  #: ``"timedelta64[ns]"`` numpy dtype
Category = dtypes.Category  #: pandas ``"categorical"`` datatype
Float = dtypes.Float  #: ``"float"`` numpy dtype
Float16 = dtypes.Float16  #: ``"float16"`` numpy dtype
Float32 = dtypes.Float32  #: ``"float32"`` numpy dtype
Float64 = dtypes.Float64  #: ``"float64"`` numpy dtype
Int = dtypes.Int  #: ``"int"`` numpy dtype
Int8 = dtypes.Int8  #: ``"int8"`` numpy dtype
Int16 = dtypes.Int16  #: ``"int16"`` numpy dtype
Int32 = dtypes.Int32  #: ``"int32"`` numpy dtype
Int64 = dtypes.Int64  #: ``"int64"`` numpy dtype
UInt8 = dtypes.UInt8  #: ``"uint8"`` numpy dtype
UInt16 = dtypes.UInt16  #: ``"uint16"`` numpy dtype
UInt32 = dtypes.UInt32  #: ``"uint32"`` numpy dtype
UInt64 = dtypes.UInt64  #: ``"uint64"`` numpy dtype


GenericDtype = TypeVar(  # type: ignore
    "GenericDtype",
    bound=Union[
        bool,
        int,
        str,
        float,
        Bool,
        Date,
        DateTime,
        Decimal,
        Timedelta,
        Category,
        Float,
        Float16,
        Float32,
        Float64,
        Int,
        Int8,
        Int16,
        Int32,
        Int64,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
    ],
)

DataFrameModel = TypeVar("DataFrameModel", bound="DataFrameModel")  # type: ignore
ColumnDtype = TypeVar("ColumnDtype")


if TYPE_CHECKING:
    T = TypeVar("T")  # pragma: no cover
else:
    T = DataFrameModel


__orig_generic_alias_call = copy.copy(_GenericAlias.__call__)


def __patched_generic_alias_call(self, *args, **kwargs):
    """
    Patched implementation of _GenericAlias.__call__ so that validation errors
    can be raised when instantiating an instance of pandera DataFrame generics,
    e.g. DataFrame[A](data).
    """
    if DataFrameBase not in self.__origin__.__bases__:
        return __orig_generic_alias_call(self, *args, **kwargs)

    if not self._inst:
        raise TypeError(
            f"Type {self._name} cannot be instantiated; "
            f"use {self.__origin__.__name__}() instead"
        )
    result = self.__origin__(*args, **kwargs)
    try:
        result.__orig_class__ = self
    # Limit the patched behavior to subset of exception types
    except (
        TypeError,
        errors.SchemaError,
        errors.SchemaErrors,
        errors.SchemaInitError,
        errors.SchemaDefinitionError,
    ):
        raise
    # In python 3.11.9, all exceptions when setting attributes when defining
    # _GenericAlias subclasses are caught and ignored.
    except Exception:
        pass
    return result


_GenericAlias.__call__ = __patched_generic_alias_call


class DataFrameBase(Generic[T]):
    """
    Pandera Dataframe base class for validating dataframes on
    initialization.
    """

    default_dtype: type | None = None

    def _before_schema_validate(self, schema):
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name == "__orig_class__":
            orig_class = value
            class_args = getattr(orig_class, "__args__", None)
            if class_args is not None and any(
                x.__name__ == "DataFrameModel"
                for x in inspect.getmro(class_args[0])
            ):
                schema_model = value.__args__[0]
                schema = schema_model.to_schema()
            else:
                raise TypeError("Could not find DataFrameModel in class args")

            # prevent the double validation problem by preventing checks for
            # dataframes with a defined pandera.schema
            pandera_accessor = getattr(self, "pandera", None)

            if (
                pandera_accessor is None
                or pandera_accessor.schema is None
                or pandera_accessor.schema != schema
            ):
                check_obj = self._before_schema_validate(schema)
                self.__dict__.update(schema.validate(check_obj).__dict__)
                if pandera_accessor is None:
                    pandera_accessor = getattr(self, "pandera")
                pandera_accessor.add_schema(schema)


class SeriesBase(Generic[GenericDtype]):
    """Pandera Series base class to use for all pandas-like APIs."""

    default_dtype: type | None = None

    def __get__(
        self, instance: object, owner: type
    ) -> str:  # pragma: no cover
        raise AttributeError("Series should resolve to Field-s")


class IndexBase(Generic[GenericDtype]):
    """Representation of pandas.Index, only used for type annotation.

    *new in 0.5.0*
    """

    default_dtype: type | None = None

    def __get__(
        self, instance: object, owner: type
    ) -> str:  # pragma: no cover
        raise AttributeError("Indexes should resolve to pa.Index-s")


class Column(Generic[ColumnDtype]):
    """Typing-only descriptor for a :class:`DataFrameModel` column.

    The annotation's generic argument is the dtype consumed by Pandera's
    runtime model parser. The descriptor models the runtime replacement made
    by ``DataFrameModel``: required class-level access is a column name, while
    instance-level access retains the generic dtype.
    """

    @overload
    def __get__(self, instance: None, owner: type[Any]) -> str: ...

    @overload
    def __get__(self, instance: object, owner: type[Any]) -> ColumnDtype: ...

    def __get__(
        self, instance: object | None, owner: type[Any]
    ) -> str | ColumnDtype:
        raise AttributeError("Column should resolve to a DataFrameModel field")


class AnnotationInfo:
    """Captures extra information about an annotation.

    Attributes:
        origin: The non-parameterized generic class.
        args: All generic types for accessing as an iterable.
        arg: The first generic type (DataFrameModel does not support more than
            1 argument).
        literal: Whether the annotation is a literal.
        optional: Whether the annotation is optional.
        nullable: Whether the annotation's values are nullable.
        is_column: Whether the annotation uses ``Column[T]``.
        raw_annotation: The raw annotation.
        metadata: Extra arguments passed to :data:`typing.Annotated`.
    """

    def __init__(self, raw_annotation: type) -> None:
        self._parse_annotation(raw_annotation)

    @property
    def is_generic_df(self) -> bool:
        """True if the annotation is a DataFrameBase subclass."""
        try:
            if self.origin is None:
                return False
            return issubclass(self.origin, DataFrameBase)
        except TypeError:
            return False

    @property
    def is_generic_xarray(self) -> bool:
        """True if the annotation is an xarray model container."""
        try:
            from pandera.typing.xarray import XarrayAnnotationBase

            if self.origin is None:
                return False
            return issubclass(self.origin, XarrayAnnotationBase)
        except (TypeError, ImportError):
            return False

    @property
    def is_generic_model(self) -> bool:
        """True if the annotation wraps a pandera model class."""
        return self.is_generic_df or self.is_generic_xarray

    def _parse_annotation(self, raw_annotation: type) -> None:
        """Parse key information from annotation.

        :param annotation: A subscripted type.
        :returns: Annotation
        """
        self.raw_annotation = raw_annotation
        self.origin = self.arg = None
        self.is_annotated_type = False
        self.is_column = False
        self.nullable = False

        self.optional = typing_inspect.is_optional_type(raw_annotation)
        if self.optional and typing_inspect.is_union_type(raw_annotation):
            # Annotated with Optional or Union[..., NoneType]
            # get_args -> (pandera.typing.Index[str], <class 'NoneType'>)
            raw_annotation = get_args(raw_annotation)[0]
            self.raw_annotation = raw_annotation

        metadata = getattr(raw_annotation, "__metadata__", None)
        if metadata:
            self.is_annotated_type = True
            raw_annotation = get_args(raw_annotation)[0]

        self.origin = get_origin(raw_annotation)
        # Replace empty tuple returned from get_args by None
        args = get_args(raw_annotation) or None
        self.args = args
        self.arg = (
            args[0]
            if args
            else raw_annotation
            if self.is_annotated_type
            else None
        )

        if metadata and self.arg is not None:
            try:
                inspect.signature(self.arg)
            except (TypeError, ValueError):
                # Preserve the historical behavior for arbitrary Annotated
                # metadata that is not a dtype constructor argument.
                metadata = None

        if not metadata and (
            nested_metadata := getattr(self.arg, "__metadata__", None)
        ):
            metadata = nested_metadata
            self.arg = get_args(self.arg)[0]

        self.metadata = metadata

        if raw_annotation is Column:
            self.is_column = True
            self.literal = False
            self.args = None
            self.arg = None
            self.default_dtype = None
            return

        if self.origin is Column:
            self.is_column = True
            self.literal = False
            self.arg, self.nullable = _unwrap_column_dtype(self.arg)
            self.args = (self.arg,)
            self.default_dtype = getattr(self.arg, "default_dtype", None)
            return

        self.literal = get_origin(self.arg) is typing.Literal

        if self.literal:
            self.arg = get_args(self.arg)[0]
        elif self.origin is None and self.metadata is None:
            if isinstance(raw_annotation, type) and issubclass(
                raw_annotation, SeriesBase
            ):
                # handle case where the provided annotation is just a pandera Series generic.
                self.arg = Any
            else:
                # otherwise assume that the annotation is the data type itself.
                self.arg = raw_annotation
        self.default_dtype = getattr(raw_annotation, "default_dtype", None)


def _unwrap_column_dtype(dtype: Any) -> tuple[Any, bool]:
    """Unwrap ``Column[T | None]`` and return its value-nullability."""
    if typing_inspect.is_optional_type(dtype) and typing_inspect.is_union_type(
        dtype
    ):
        return get_args(dtype)[0], True
    return dtype, False


def is_column_annotation(annotation: AnnotationInfo) -> bool:
    """Return whether an annotation uses the shared ``Column[T]`` marker."""
    return annotation.is_column
