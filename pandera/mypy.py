"""Pandera mypy plugin."""

from typing import Union, cast

from mypy.nodes import ARG_OPT, ARG_POS, TypeInfo
from mypy.plugin import FunctionSigContext, MethodSigContext, Plugin
from mypy.types import AnyType, CallableType, Instance, TypeOfAny, UnionType

PANDAS_DATAFRAME_FULLNAME = "pandera.typing.pandas.DataFrame"
PANDAS_CONCAT = "pandas.core.reshape.concat.concat"
PANDERA_CHECK_TYPES_FULLNAME = "pandera.decorators.check_types"


# pylint: disable=unused-argument
def plugin(version: str):
    """Mypy plugin entrypoint."""
    return PanderaPlugin


def is_pandas_module(fullname: str) -> bool:
    """Check if a fully qualified name is from the pandas module"""
    return fullname.startswith("pandas.")


# pylint: disable=too-few-public-methods
class PanderaPluginConfig:
    """Pandera mypy plugin config"""

    def __init__(self, options):
        """Configuration options (config options are still TBD)."""


class PanderaPlugin(Plugin):
    """Pandera mypy plugin.

    Since pandera uses the pandas-stubs library:
    https://github.com/VirtusLab/pandas-stubs

    We need to patch all of the function/method signatures in the library
    which turn out to yield many false positives with respect to regular
    pandas usage. Currently this is mostly what this plugin does, though the
    future plan for this plugin is to improve and enable users to customize
    the static typing experience for both pandas and pandera.

    Once the pandas library officially supports type annotations via
    `py.typed`, we'll remove the dependency on pandas-stubs:
    https://github.com/pandas-dev/pandas/issues/28142#issuecomment-991967009
    """

    def __init__(self, options) -> None:
        self.plugin_config = PanderaPluginConfig(options)
        super().__init__(options)

    def get_function_signature_hook(self, fullname: str):
        """Adjust the function signatures of pandas functions."""
        if fullname == PANDAS_CONCAT:
            return self.pandas_concat_callback
        elif is_pandas_module(fullname):
            return self.disable_pandas_function_callback

    def get_method_signature_hook(self, fullname: str):
        """Adjust the function signatures of pandas methods."""
        if is_pandas_module(fullname):
            return self.disable_pandas_function_callback

    def pandas_concat_callback(
        self, ctx: Union[FunctionSigContext, MethodSigContext]
    ) -> CallableType:
        """Adjusts the signature pandas.concat to allow generator inputs."""
        iterable = self.lookup_fully_qualified("typing.Iterable")
        if iterable is not None:
            iterable_node = cast(TypeInfo, iterable.node)
        else:
            raise ValueError("typing.Iterable node not found")

        union_type = cast(UnionType, ctx.default_signature.arg_types[0])

        pandas_data_type = ctx.default_signature.ret_type
        arg_types = [
            UnionType(
                [
                    Instance(iterable_node, [pandas_data_type]),
                    *union_type.items,
                ]
            ),
            *ctx.default_signature.arg_types[1:],
        ]
        return ctx.default_signature.copy_modified(arg_types=arg_types)

    # pylint: disable=no-self-use
    def disable_pandas_function_callback(
        self, ctx: FunctionSigContext
    ) -> CallableType:
        """Makes all pandas function/method inputs Optional[Any]."""
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
