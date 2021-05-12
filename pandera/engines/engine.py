import functools
import inspect
import warnings
from abc import ABCMeta
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Type, Union, get_type_hints

import typing_inspect

from pandera.dtypes_ import DataType


@dataclass
class _DtypeRegistry:
    dispatch: Callable[[Any], DataType]
    equivalents: Dict[Any, Type[DataType]]


class Engine(ABCMeta):
    """Base Engine.

    Keep a registry of concrete Engines (currently, only pandas).
    """

    _registry: Dict["Engine", _DtypeRegistry] = {}
    _base_datatype: Type[DataType]

    def __new__(mcs, name, bases, namespace, **kwargs):
        base_datatype = kwargs.pop("base_datatype")
        try:  # allow multiple base datatypes
            base_datatype = tuple(base_datatype)
        except TypeError:
            pass
        namespace["_base_datatype"] = base_datatype
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        @functools.singledispatch
        def dtype(obj: Any) -> DataType:
            raise ValueError(f"Data type '{obj}' not understood")

        mcs._registry[cls] = _DtypeRegistry(dispatch=dtype, equivalents={})
        return cls

    def _check_source_dtype(cls, obj: Any) -> None:
        if isinstance(obj, cls._base_datatype) or (
            inspect.isclass(obj) and issubclass(obj, cls._base_datatype)
        ):
            raise ValueError(
                f"{cls._base_datatype.__name__} subclasses cannot be registered"
                f" with {cls.__name__}."
            )

    def _register_from_parametrized_dtype(
        cls, target_dtype: Union[DataType, Type[DataType]], method: classmethod
    ) -> None:
        func = method.__func__
        annotations = get_type_hints(func).values()
        dtype = next(iter(annotations))  # get 1st annotation
        # parse typing.Union
        dtypes = typing_inspect.get_args(dtype) or [dtype]

        def _method(*args, **kwargs):
            return func(target_dtype, *args, **kwargs)

        for source_dtype in dtypes:
            cls._check_source_dtype(source_dtype)
            cls._registry[cls].dispatch.register(source_dtype, _method)

    def _register_equivalents(
        cls,
        target_dtype: Union[DataType, Type[DataType]],
        *source_dtypes: Any,
    ) -> None:
        value = target_dtype()
        for source_dtype in source_dtypes:
            cls._check_source_dtype(source_dtype)
            cls._registry[cls].equivalents[source_dtype] = value

    def register_dtype(
        cls,
        dtype: Type[DataType] = None,
        *,
        equivalents: List[DataType] = None,
    ):
        """Register a DataType

        :param dtype: The DataType to register.
        :param equivalents: Equivalent scalar dtype class or
            non-parametrized dtype instance.

        .. note::
            Register the classmethod ``from_parametrized_dtype`` if present.
        """

        def _wrapper(dtype: Union[DataType, Type[DataType]]):
            if not inspect.isclass(dtype):
                raise ValueError(
                    f"{cls.__name__}.register_dtype can only decorate a class, "
                    + f"got {dtype}"
                )

            if equivalents:
                cls._register_equivalents(dtype, *equivalents)

            from_parametrized_dtype = dtype.__dict__.get(
                "from_parametrized_dtype"
            )
            if from_parametrized_dtype:
                if not isinstance(from_parametrized_dtype, classmethod):
                    raise ValueError(
                        f"{dtype.__name__}.from_parametrized_dtype "
                        + "must be a classmethod."
                    )
                cls._register_from_parametrized_dtype(
                    dtype,
                    from_parametrized_dtype,
                )
            elif not equivalents:
                warnings.warn(
                    f"register_dtype({dtype}) on a class without a "
                    + "'from_parametrized_dtype' classmethod has no effect."
                )

            return dtype

        if dtype:
            return _wrapper(dtype)

        return _wrapper

    def dtype(cls, obj: Any) -> DataType:
        """Convert input into a DataType object."""
        if isinstance(obj, cls._base_datatype):
            return obj

        if inspect.isclass(obj) and issubclass(obj, cls._base_datatype):
            try:
                return obj()
            except (TypeError, AttributeError) as err:
                raise TypeError(
                    f"DataType '{obj.__name__}' cannot be instantiated: "
                    f"{err}\n "
                    + "Usage Tip: Use an instance or a string representation."
                ) from err

        registry = cls._registry[cls]

        data_type = registry.equivalents.get(obj)
        if data_type is not None:
            return data_type

        try:
            return registry.dispatch(obj)
        except (KeyError, ValueError):
            raise TypeError(
                f"Data type '{obj}' not understood by {cls.__name__}."
            ) from None
