"""Pandera mypy plugin."""

from mypy.plugin import Plugin
from mypy.plugin import MethodContext, FunctionContext, FunctionSigContext
from mypy.types import AnyType, TypeOfAny, CallableType


PANDAS_DATAFRAME_FULLNAME = "pandera.typing.pandas.DataFrame"
PANDERA_CHECK_TYPES_FULLNAME = "pandera.decorators.check_types"


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
        if is_pandas_module(fullname):
            return disable_pandas_function_callback

    def get_method_signature_hook(self, fullname: str):
        if is_pandas_module(fullname):
            return disable_pandas_method_callback

    def get_function_hook(self, fullname: str):
        if fullname == PANDERA_CHECK_TYPES_FULLNAME:
            return pandera_check_types_callback


def disable_pandas_function_callback(ctx: FunctionSigContext) -> CallableType:
    return ctx.default_signature.copy_modified(
        arg_types=[
            AnyType(TypeOfAny.explicit)
            for _ in ctx.default_signature.arg_types
        ],
    )


def disable_pandas_method_callback(ctx: MethodContext) -> CallableType:
    return ctx.default_signature.copy_modified(
        arg_types=[
            t if i == 0 else AnyType(TypeOfAny.explicit)
            for i, t in enumerate(ctx.default_signature.arg_types)
        ],
    )


def pandera_check_types_callback(ctx: FunctionContext):
    # NOTE: this doesn't work because the callback only changes the return
    # type of the result of decorating a function with `check_types`, but
    # it doesn't change the behavior of type checking within the function body
    #
    # @pa.check_types
    # def fn_cast_dataframe_invalid(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    #     return df  # mypy error still happens here
    #
    # reveal_type(fn_cast_dataframe_invalid(df))  # Any

    return ctx.default_return_type.copy_modified(
        ret_type=AnyType(TypeOfAny.explicit)
    )
