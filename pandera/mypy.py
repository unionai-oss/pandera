"""Pandera mypy plugin."""

from typing import Any, Iterable
from mypy.plugin import Plugin
from mypy.plugin import MethodContext, FunctionContext, FunctionSigContext
from mypy.types import AnyType, TypeOfAny, CallableType, UnionType, Instance
from mypy.nodes import ARG_POS, ARG_OPT


PANDAS_DATAFRAME_FULLNAME = "pandera.typing.pandas.DataFrame"
PANDAS_CONCAT = "pandas.concat"
PANDERA_CHECK_TYPES_FULLNAME = "pandera.decorators.check_types"
BUILTIN_SLICE = "builtins.slice"


def plugin(version: str):
    # ignore version argument if the plugin works with all mypy versions.
    return PanderaPlugin


def is_pandas_module(fullname: str) -> bool:
    return fullname.startswith("pandas.")


class PanderaPluginConfig:

    def __init__(self, options):
        pass


class PanderaPlugin(Plugin):

    def __init__(self, options) -> None:
        self.plugin_config = PanderaPluginConfig(options)
        super().__init__(options)

    def get_function_signature_hook(self, fullname: str):
        if fullname == PANDAS_CONCAT or (is_pandas_module(fullname) and "concat" in fullname):
            return pandas_concat_callback
        elif is_pandas_module(fullname):
            return disable_pandas_function_callback

    def get_method_signature_hook(self, fullname: str):
        if is_pandas_module(fullname):
            return disable_pandas_function_callback


def pandas_concat_callback(ctx: FunctionSigContext) -> CallableType:
    import ipdb; ipdb.set_trace()
    union_type: UnionType = ctx.default_signature.arg_types[0]
    union_type.items[0]
    data_type = ctx.default_signature.arg_types[0]
    Iterable()
    return ctx.default_signature.copy_modified(
        arg_types=[
            AnyType(TypeOfAny.explicit)
            for _ in ctx.default_signature.arg_types
        ],
        arg_kinds=[
            ARG_OPT if x == ARG_POS else x
            for x in ctx.default_signature.arg_kinds
        ],
    )


def disable_pandas_function_callback(ctx: FunctionSigContext) -> CallableType:
    return ctx.default_signature.copy_modified(
        arg_types=[
            AnyType(TypeOfAny.explicit)
            for _ in ctx.default_signature.arg_types
        ],
        arg_kinds=[
            ARG_OPT if x == ARG_POS else x
            for x in ctx.default_signature.arg_kinds
        ],
    )
