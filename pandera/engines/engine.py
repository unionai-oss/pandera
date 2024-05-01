"""Data types engine interface."""

# https://github.com/PyCQA/pylint/issues/3268
# pylint:disable=no-value-for-parameter
import functools
import inspect
import sys
from abc import ABCMeta
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
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
_EngineType = Type[_Engine]


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


def _is_typeddict(x: Type) -> bool:
    return x.__class__.__name__ == "_TypedDictMeta"


def _is_namedtuple(x: Type) -> bool:
    return tuple in getattr(x, "__bases__", ()) and hasattr(
        x, "__annotations__"
    )


@dataclass
class _DtypeRegistry:
    dispatch: Dispatch
    equivalents: Dict[Any, DataType]


class Engine(ABCMeta):
    """Base Engine metaclass.

    Keep a registry of concrete Engines.
    """

    _registry: Dict["Engine", _DtypeRegistry] = {}
    _registered_dtypes: Set[Type[DataType]]
    _base_pandera_dtypes: Tuple[Type[DataType]]

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

        mcs._registry[engine] = _DtypeRegistry(dispatch=dtype, equivalents={})
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
        pandera_dtype_cls: Type[DataType],
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
        dtypes = typing_inspect.get_args(dtype) or [dtype]

        def _method(*args, **kwargs):
            return func(pandera_dtype_cls, *args, **kwargs)

        for source_dtype in dtypes:
            cls._check_source_dtype(source_dtype)
            cls._registry[cls].dispatch.register(source_dtype, _method)

    def _register_equivalents(
        cls, pandera_dtype_cls: Type[DataType], *source_dtypes: Any
    ) -> None:
        pandera_dtype = pandera_dtype_cls()  # type: ignore
        for source_dtype in source_dtypes:
            cls._check_source_dtype(source_dtype)
            cls._registry[cls].equivalents[source_dtype] = pandera_dtype

    def register_dtype(
        cls: _EngineType,
        pandera_dtype_cls: Optional[Type[_DataType]] = None,
        *,
        equivalents: Optional[List[Any]] = None,
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

        >>> import pandera as pa
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

        def _wrapper(pandera_dtype_cls: Type[_DataType]) -> Type[_DataType]:
            if not inspect.isclass(pandera_dtype_cls):
                raise ValueError(
                    f"{cls.__name__}.register_dtype can only decorate a class,"
                    f" got {pandera_dtype_cls}"
                )

            if equivalents:
                # pylint: disable=fixme
                # Todo - Need changes to this function to support uninitialised object
                cls._register_equivalents(pandera_dtype_cls, *equivalents)

            if "from_parametrized_dtype" in pandera_dtype_cls.__dict__:
                cls._register_from_parametrized_dtype(pandera_dtype_cls)

            cls._registered_dtypes.add(pandera_dtype_cls)
            return pandera_dtype_cls

        if pandera_dtype_cls:
            return _wrapper(pandera_dtype_cls)

        return _wrapper

    def dtype(cls: _EngineType, data_type: Any) -> DataType:
        """Convert input into a Pandera :class:`DataType` object."""
        # pylint: disable=too-many-return-statements
        if isinstance(data_type, cls._base_pandera_dtypes):
            return data_type

        if (
            inspect.isclass(data_type)
            and not hasattr(data_type, "__origin__")
            and issubclass(data_type, cls._base_pandera_dtypes)
        ):
            try:
                # pylint: disable=fixme
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
        datatype_origin = typing_inspect.get_origin(data_type)
        if datatype_origin is not None:
            equivalent_data_type = registry.equivalents.get(datatype_origin)
            return type(equivalent_data_type)(data_type)

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
        if datatype_generic_bases:
            equivalent_data_type = None
            for base in datatype_generic_bases:
                equivalent_data_type = registry.equivalents.get(base)
                break
            if equivalent_data_type is None:
                raise TypeError(
                    f"Type '{data_type}' not understood by {cls.__name__}."
                )
            return type(equivalent_data_type)(data_type)

        equivalent_data_type = registry.equivalents.get(data_type)
        if equivalent_data_type is not None:
            return equivalent_data_type
        elif isinstance(data_type, DataType):
            # in the case where data_type is a parameterized dtypes.DataType instance that isn't
            # in the equivalents registry, use its type to get the equivalent, and feed
            # the parameters into the recognized data type class.
            equivalent_data_type = registry.equivalents.get(type(data_type))
            if equivalent_data_type is not None:
                return type(equivalent_data_type)(**data_type.__dict__)

        try:
            return registry.dispatch(data_type)
        except (KeyError, ValueError):
            raise TypeError(
                f"Data type '{data_type}' not understood by {cls.__name__}."
            ) from None

    def get_registered_dtypes(  # pylint:disable=W1401
        cls,
    ) -> List[Type[DataType]]:
        r"""Return the :class:`pandera.dtypes.DataType`\s registered
        with this engine."""
        return list(cls._registered_dtypes)
