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
    _base_pandera_dtypes: Type[DataType]

    def __new__(mcs, name, bases, namespace, **kwargs):
        base_pandera_dtypes = kwargs.pop("base_pandera_dtypes")
        try:  # allow multiple base datatypes
            base_pandera_dtypes = tuple(base_pandera_dtypes)
        except TypeError:
            pass
        namespace["_base_pandera_dtypes"] = base_pandera_dtypes
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        @functools.singledispatch
        def dtype(data_type: Any) -> DataType:
            raise ValueError(f"Data type '{data_type}' not understood")

        mcs._registry[cls] = _DtypeRegistry(dispatch=dtype, equivalents={})
        return cls

    def _check_source_dtype(cls, data_type: Any) -> None:
        if isinstance(data_type, cls._base_pandera_dtypes) or (
            inspect.isclass(data_type)
            and issubclass(data_type, cls._base_pandera_dtypes)
        ):
            raise ValueError(
                f"{cls._base_pandera_dtypes.__name__} subclasses cannot be registered"
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
        pandera_dtype: Type[DataType] = None,
        *,
        equivalents: List[DataType] = None,
    ):
        """Register a Pandera :class:`DataType`.

        :param pandera_dtype: The DataType to register.
        :param equivalents: Equivalent scalar data type class or
            non-parametrized data type instance.

        .. note::
            The classmethod ``from_parametrized_dtype`` will also be registered.
        """

        def _wrapper(pandera_dtype: Union[DataType, Type[DataType]]):
            if not inspect.isclass(pandera_dtype):
                raise ValueError(
                    f"{cls.__name__}.register_dtype can only decorate a class, "
                    + f"got {pandera_dtype}"
                )

            if equivalents:
                cls._register_equivalents(pandera_dtype, *equivalents)

            from_parametrized_dtype = pandera_dtype.__dict__.get(
                "from_parametrized_dtype"
            )
            if from_parametrized_dtype:
                if not isinstance(from_parametrized_dtype, classmethod):
                    raise ValueError(
                        f"{pandera_dtype.__name__}.from_parametrized_dtype "
                        + "must be a classmethod."
                    )
                cls._register_from_parametrized_dtype(
                    pandera_dtype,
                    from_parametrized_dtype,
                )
            elif not equivalents:
                warnings.warn(
                    f"register_dtype({pandera_dtype}) on a class without a "
                    + "'from_parametrized_dtype' classmethod has no effect."
                )

            return pandera_dtype

        if pandera_dtype:
            return _wrapper(pandera_dtype)

        return _wrapper

    def dtype(cls, data_type: Any) -> DataType:
        """Convert input into a Pandera :class:`DataType` object."""
        if isinstance(data_type, cls._base_pandera_dtypes):
            return data_type

        if inspect.isclass(data_type) and issubclass(
            data_type, cls._base_pandera_dtypes
        ):
            try:
                return data_type()
            except (TypeError, AttributeError) as err:
                raise TypeError(
                    f"DataType '{data_type.__name__}' cannot be instantiated: "
                    f"{err}\n "
                    + "Usage Tip: Use an instance or a string representation."
                ) from err

        registry = cls._registry[cls]

        equivalent_data_type = registry.equivalents.get(data_type)
        if equivalent_data_type is not None:
            return equivalent_data_type

        try:
            return registry.dispatch(data_type)
        except (KeyError, ValueError):
            raise TypeError(
                f"Data type '{data_type}' not understood by {cls.__name__}."
            ) from None
