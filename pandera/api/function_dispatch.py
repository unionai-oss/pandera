"""Multidispatcher implementation."""

from inspect import signature
from typing import Callable, Dict, Tuple, Type, Union
import typing_inspect


class Dispatcher:
    """Dispatch implementation."""

    def __init__(self):
        self._function_registry: Dict[Type, Callable] = {}
        self._name = None

    def register(self, fn):
        # Get function signature
        self._name = fn.__name__
        data_types = get_first_arg_type(fn)
        for data_type in data_types:
            self._function_registry[data_type] = fn

    def __call__(self, *args, **kwargs):
        input_data_type = type(args[0])
        fn = self._function_registry[input_data_type]
        return fn(*args, **kwargs)

    @property
    def co_code(self):
        """Method for getting bytecode of all the registered functions."""
        _code = b""
        for fn in self._function_registry.values():
            _code += fn.__code__.co_code
        return _code

    @property
    def __name__(self):
        return f"{self._name}"

    def __str__(self):
        return f"{self._name}"

    def __repr__(self):
        return f"{self._name}"


def get_first_arg_type(fn):
    fn_sig = signature(fn)

    # register the check strategy for this particular check, identified
    # by the check `name`, and the data type of the check function. This
    # supports Union types. Also assume that the data type of the data
    # object to validate is the first argument.
    data_type = [*fn_sig.parameters.values()][0].annotation

    if typing_inspect.get_origin(data_type) in (tuple, Tuple):
        data_type, *_ = typing_inspect.get_args(data_type)

    if typing_inspect.get_origin(data_type) is Union:
        data_types = typing_inspect.get_args(data_type)
    else:
        data_types = (data_type,)

    return data_types
