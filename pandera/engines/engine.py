"""Data types engine interface."""

# https://github.com/PyCQA/pylint/issues/3268
import functools
import inspect
import sys
from abc import ABCMeta
from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Optional,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

import typing_inspect

from pandera.dtypes import DataType

# register different TypedDict type depending on python version
if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict  # noqa


_DataType = TypeVar("_DataType", bound=DataType)
_Engine = TypeVar("_Engine", bound="Engine")
_EngineType = type[_Engine]


if TYPE_CHECKING:  # pragma: no cover

    class Dispatch:
        """Only used for type annotation."""

        def __call__(self, data_type: Any, **kwds: Any) -> Any:
            pass

        @staticmethod
        def register(
            data_type: Any, func: Callable[[Any], DataType]
        ) -> Callable[[Any], DataType]:
            """Register a new implementation for the given cls."""

else:
    Dispatch = Callable[[Any], DataType]


def _is_typeddict(x: type) -> bool:
    return x.__class__.__name__ == "_TypedDictMeta"


def _is_namedtuple(x: type) -> bool:
    return tuple in getattr(x, "__bases__", ()) and hasattr(
        x, "__annotations__"
    )


@dataclass
class _DtypeRegistry:
    dispatch: Dispatch
    equivalents: dict[Any, DataType]
    strict_equivalents: dict[Any, DataType]

    def get_equivalent(self, data_type: Any) -> DataType | None:
        if (data_type, type(data_type)) in self.strict_equivalents:
            return self.strict_equivalents.get((data_type, type(data_type)))
        return self.equivalents.get(data_type)


@dataclass
class StrictEquivalent:
    """
    Represents data types that are equivalent to the pandera DataType that
    are meant to be evaluated strictly, i.e. not with `data_type == "string_alias`
    """

    dtype: DataType


class Engine(ABCMeta):
    """Base Engine metaclass.

    Keep a registry of concrete Engines.
    """

    _registry: dict["Engine", _DtypeRegistry] = {}
    _registered_dtypes: set[type[DataType]]
    _base_pandera_dtypes: tuple[type[DataType]]

    def __new__(mcs, name, bases, namespace, **kwargs):
        base_pandera_dtypes = kwargs.pop("base_pandera_dtypes")
        try:
            namespace["_base_pandera_dtypes"] = tuple(base_pandera_dtypes)
        except TypeError:
            namespace["_base_pandera_dtypes"] = (base_pandera_dtypes,)

        namespace["_registered_dtypes"] = set()
        engine = super().__new__(mcs, name, bases, namespace, **kwargs)

        @functools.singledispatch
        def dtype(data_type: Any) -> DataType:
            raise ValueError(f"Data type '{data_type}' not understood")

        mcs._registry[engine] = _DtypeRegistry(
            dispatch=dtype, equivalents={}, strict_equivalents={}
        )
        return engine

    def _check_source_dtype(cls, data_type: Any) -> None:
        if isinstance(data_type, cls._base_pandera_dtypes) or (
            inspect.isclass(data_type)
            and issubclass(data_type, cls._base_pandera_dtypes)
        ):
            base_names = [
                f"{base.__module__}.{base.__qualname__}"
                for base in cls._base_pandera_dtypes
            ]
            raise ValueError(
                f"Subclasses of {base_names} cannot be registered"
                f" with {cls.__name__}."
            )

    def _register_from_parametrized_dtype(
        cls,
        pandera_dtype_cls: type[DataType],
    ) -> None:
        method = pandera_dtype_cls.__dict__["from_parametrized_dtype"]
        if not isinstance(method, classmethod):
            raise ValueError(
                f"{pandera_dtype_cls.__name__}.from_parametrized_dtype "
                + "must be a classmethod."
            )
        func = method.__func__
        annotations = get_type_hints(func).values()
        dtype = next(iter(annotations))  # get 1st annotation
        # parse typing.Union
        dtypes = get_args(dtype) or [dtype]

        def _method(*args, **kwargs):
            return func(pandera_dtype_cls, *args, **kwargs)

        for source_dtype in dtypes:
            cls._check_source_dtype(source_dtype)
            cls._registry[cls].dispatch.register(source_dtype, _method)

    def _register_equivalents(
        cls, pandera_dtype_cls: type[DataType], *source_dtypes: Any
    ) -> None:
        pandera_dtype = pandera_dtype_cls()  # type: ignore
        for source_dtype in source_dtypes:
            if isinstance(source_dtype, StrictEquivalent):
                cls._check_source_dtype(source_dtype.dtype)
                cls._registry[cls].strict_equivalents[
                    (source_dtype.dtype, type(source_dtype.dtype))
                ] = pandera_dtype
            else:
                cls._check_source_dtype(source_dtype)
                cls._registry[cls].equivalents[source_dtype] = pandera_dtype

    def register_dtype(
        cls: "Engine",
        pandera_dtype_cls: type[_DataType] | None = None,
        *,
        equivalents: list[Any] | None = None,
    ) -> Callable:
        """Register a Pandera :class:`~pandera.dtypes.DataType` with the engine,
        as class decorator.

        :param pandera_dtype: The DataType to register.
        :param equivalents: Equivalent scalar data type classes or
            non-parametrized data type instances.

        .. note::
            The classmethod ``from_parametrized_dtype`` will also be
            registered. See :ref:`here<dtypes>` for more usage details.

        :example:

        >>> import pandera.pandas as pa
        >>>
        >>> class MyDataType(pa.DataType):
        ...     pass
        >>>
        >>> class MyEngine(
        ...     metaclass=pa.engines.engine.Engine,
        ...     base_pandera_dtypes=MyDataType,
        ... ):
        ...     pass
        >>>
        >>> @MyEngine.register_dtype(equivalents=[bool])
        ... class MyBool(MyDataType):
        ...     pass

        """

        def _wrapper(pandera_dtype_cls: type[_DataType]) -> type[_DataType]:
            if not inspect.isclass(pandera_dtype_cls):
                raise ValueError(
                    f"{cls.__name__}.register_dtype can only decorate a class,"
                    f" got {pandera_dtype_cls}"
                )

            if equivalents:
                # Todo - Need changes to this function to support uninitialised object
                cls._register_equivalents(pandera_dtype_cls, *equivalents)

            if "from_parametrized_dtype" in pandera_dtype_cls.__dict__:
                cls._register_from_parametrized_dtype(pandera_dtype_cls)

            cls._registered_dtypes.add(pandera_dtype_cls)
            return pandera_dtype_cls

        if pandera_dtype_cls:
            return _wrapper(pandera_dtype_cls)

        return _wrapper

    def dtype(cls: "Engine", data_type: Any) -> DataType:
        """Convert input into a Pandera :class:`DataType` object."""
        if isinstance(data_type, cls._base_pandera_dtypes):
            return data_type

        if (
            inspect.isclass(data_type)
            and not hasattr(data_type, "__origin__")
            and issubclass(data_type, cls._base_pandera_dtypes)
        ):
            try:
                # Todo  - check if we can move to validate without initialization
                return data_type()
            except (TypeError, AttributeError) as err:
                raise TypeError(
                    f"DataType '{data_type.__name__}' cannot be instantiated: "
                    f"{err}\n "
                    + "Usage Tip: Use an instance or a string representation."
                ) from err

        registry = cls._registry[cls]

        # handle python generic types, e.g. typing.Dict[str, str]
        datatype_origin = get_origin(data_type)
        if datatype_origin is not None:
            equivalent_data_type = registry.get_equivalent(datatype_origin)
            return type(equivalent_data_type)(data_type)  # type: ignore

        # handle python's special declared type constructs like NamedTuple and
        # TypedDict
        datatype_generic_bases = (
            # handle python < 3.9 cases, where TypedDict/NameDtuple isn't part
            # of the generic base classes returned by
            # typing_inspect.get_generic_bases
            ((TypedDict,) if _is_typeddict(data_type) else ())
            or ((NamedTuple,) if _is_namedtuple(data_type) else ())
            or typing_inspect.get_generic_bases(data_type)
        )
        if datatype_generic_bases and inspect.getmodule(
            base := datatype_generic_bases[0]
        ).__name__ in {  # type: ignore[union-attr]
            *sys.stdlib_module_names,
            "typing_extensions",
        }:
            equivalent_data_type = registry.get_equivalent(base)
            if equivalent_data_type is None:
                raise TypeError(
                    f"Type '{data_type}' not understood by {cls.__name__}."
                )
            return type(equivalent_data_type)(data_type)  # type: ignore

        equivalent_data_type = registry.get_equivalent(data_type)
        if equivalent_data_type is not None:
            return equivalent_data_type
        elif isinstance(data_type, DataType):
            # in the case where data_type is a parameterized dtypes.DataType instance that isn't
            # in the equivalents registry, use its type to get the equivalent, and feed
            # the parameters into the recognized data type class.
            equivalent_data_type = registry.get_equivalent(type(data_type))
            if equivalent_data_type is not None:
                return type(equivalent_data_type)(**data_type.__dict__)

        try:
            return registry.dispatch(data_type)
        except (KeyError, ValueError):
            raise TypeError(
                f"Data type '{data_type}' not understood by {cls.__name__}."
            ) from None

    def get_registered_dtypes(
        cls,
    ) -> list[type[DataType]]:
        r"""Return the :class:`pandera.dtypes.DataType`\s registered
        with this engine."""
        return list(cls._registered_dtypes)
