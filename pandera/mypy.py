"""Pandera mypy plugin."""

from collections.abc import Callable
from typing import Optional, Union, cast

from mypy.nodes import (
    AssignmentStmt,
    FuncBase,
    NameExpr,
    SymbolNode,
    TypeInfo,
    Var,
)
from mypy.plugin import (
    AttributeContext,
    ClassDefContext,
    FunctionSigContext,
    MethodSigContext,
    Plugin,
)
from mypy.types import CallableType, Instance, UnionType

DATAFRAMEMODEL_FULLNAMES = {
    "pandera.api.dataframe.model.DataFrameModel",
    "pandera.api.pandas.model.DataFrameModel",
    "pandera.api.geopandas.GeoDataFrameModel",
    "pandera.api.geopandas.model.GeoDataFrameModel",
    "pandera.pandas.DataFrameModel",
    "pandera.geopandas.GeoDataFrameModel",
    "pandera._pandas_deprecated.DataFrameModel",
}
PANDERA_PANDAS_DATAFRAME_FULLNAME = "pandera.typing.pandas.DataFrame"
PANDERA_PANDAS_SERIES_FULLNAME = "pandera.typing.pandas.Series"
PANDERA_PANDAS_INDEX_FULLNAME = "pandera.typing.pandas.Index"
PANDERA_MODIN_SERIES_FULLNAME = "pandera.typing.modin.Series"
PANDERA_MODIN_INDEX_FULLNAME = "pandera.typing.modin.Index"
PANDERA_DASK_SERIES_FULLNAME = "pandera.typing.dask.Series"
PANDERA_DASK_INDEX_FULLNAME = "pandera.typing.dask.Index"
PANDERA_PYSPARK_SERIES_FULLNAME = "pandera.typing.pyspark.Series"
PANDERA_PYSPARK_INDEX_FULLNAME = "pandera.typing.pyspark.Index"
PANDERA_GEOPANDAS_SERIES_FULLNAME = "pandera.typing.geopandas.GeoSeries"
PANDAS_CONCAT = "pandas.core.reshape.concat.concat"

FIELD_GENERICS_FULLNAMES = {
    PANDERA_PANDAS_SERIES_FULLNAME,
    PANDERA_PANDAS_INDEX_FULLNAME,
    PANDERA_MODIN_SERIES_FULLNAME,
    PANDERA_MODIN_INDEX_FULLNAME,
    PANDERA_DASK_SERIES_FULLNAME,
    PANDERA_DASK_INDEX_FULLNAME,
    PANDERA_PYSPARK_SERIES_FULLNAME,
    PANDERA_PYSPARK_INDEX_FULLNAME,
    PANDERA_GEOPANDAS_SERIES_FULLNAME,
}


def plugin(version: str):
    """Mypy plugin entrypoint."""
    return PanderaPlugin


def is_pandas_module(fullname: str) -> bool:
    """Check if a fully qualified name is from the pandas module"""
    return fullname.startswith("pandas.")


class PanderaPlugin(Plugin):
    """Pandera mypy plugin.

    Since pandera uses the pandas-stubs library:
    https://github.com/pandas-dev/pandas-stubs

    We need to patch all of the function/method signatures in the library
    which turn out to yield many false positives with respect to regular
    pandas usage. Currently this is what this plugin does, though the
    future plan for this plugin is to improve and enable users to customize
    the static typing experience for both pandas and pandera.
    """

    def __init__(self, options) -> None:
        self.plugin_config = PanderaPluginConfig(options)
        super().__init__(options)

    def get_base_class_hook(
        self, fullname: str
    ) -> "Callable[[ClassDefContext], None] | None":
        if fullname in DATAFRAMEMODEL_FULLNAMES:
            return self._pandera_model_class_maker_callback

        sym = self.lookup_fully_qualified(fullname)
        if sym and isinstance(sym.node, TypeInfo):  # pragma: no branch
            if any(
                get_fullname(base) in DATAFRAMEMODEL_FULLNAMES
                for base in sym.node.mro
            ):
                return self._pandera_model_class_maker_callback
        return None

    def _pandera_model_class_maker_callback(
        self, ctx: ClassDefContext
    ) -> None:
        transformer = DataFrameModelTransformer(ctx, self.plugin_config)
        transformer.transform()

    def get_class_attribute_hook(
        self, fullname: str
    ) -> "Callable[[AttributeContext], Instance] | None":
        if self._is_dataframe_model_field_attribute(fullname):
            return self._dataframe_model_class_attr_callback
        return None

    def get_attribute_hook(
        self, fullname: str
    ) -> "Callable[[AttributeContext], Instance] | None":
        if self._is_dataframe_model_field_attribute(fullname):
            return self._dataframe_model_class_attr_callback
        return None

    def _is_dataframe_model_field_attribute(self, fullname: str) -> bool:
        class_fullname, _, attr_name = fullname.rpartition(".")
        if not class_fullname or not attr_name:
            return False

        sym = self.lookup_fully_qualified(class_fullname)
        if not sym or not isinstance(sym.node, TypeInfo):
            return False

        class_info = sym.node
        if not any(
            get_fullname(base) in DATAFRAMEMODEL_FULLNAMES
            for base in class_info.mro
        ):
            return False

        if attr_name.startswith("_") or attr_name == "Config":
            return False

        attr_sym = class_info.names.get(attr_name)
        return bool(attr_sym and isinstance(attr_sym.node, Var))

    @staticmethod
    def _dataframe_model_class_attr_callback(
        ctx: AttributeContext,
    ) -> Instance:
        return ctx.api.named_generic_type("builtins.str", [])


class DataFrameModelTransformer:
    def __init__(self, ctx: ClassDefContext, plugin_config):
        self.ctx = ctx

    def transform(self) -> None:
        self.erase_field_type_arg()
        self.set_field_type_to_str()

    def _get_field_assignments(self):
        """Get DataFrameModel field assignment statements."""
        for def_ in self.ctx.cls.defs.body:
            if not isinstance(def_, AssignmentStmt):
                continue
            if len(def_.lvalues) != 1:
                continue
            field_name_expr = def_.lvalues[0]
            if not isinstance(field_name_expr, NameExpr):
                continue
            field_name = field_name_expr.name
            if field_name.startswith("_") or field_name == "Config":
                continue
            if def_.type is None:
                continue
            symbol_node = field_name_expr.node
            var = symbol_node if isinstance(symbol_node, Var) else None
            yield def_, var

    def erase_field_type_arg(self):
        """Erase type information of DataFrameModel fields.

        This allows for overriding types when subclassing DataFrameModels. For
        example:

        class BaseSchema(pa.DataFrameModel):
            x: pa.typing.Series[int]

        class Schema(BaseSchema):
            x: pa.typing.Series[str]  # mypy assignment error, cannot override types
        """
        for def_, var in self._get_field_assignments():
            type_ = def_.type
            if (
                # e.g. UnionType does not have module_name or name
                not hasattr(type_, "module_name") or not hasattr(type_, "name")
            ):
                continue
            if str(type_) in FIELD_GENERICS_FULLNAMES:
                type_.args = ()  # erase generic type arg
                if var is not None and hasattr(var.type, "args"):
                    var.type.args = ()

    def set_field_type_to_str(self) -> None:
        """Type DataFrameModel field class attributes as column names."""
        str_type = self.ctx.api.named_type("builtins.str")
        for def_, var in self._get_field_assignments():
            def_.type = str_type
            if var is not None:
                var.type = str_type


class PanderaPluginConfig:
    """Pandera mypy plugin config"""

    def __init__(self, options):
        """Configuration options (config options are still TBD)."""
        self.options = options


def get_fullname(x: Union[FuncBase, SymbolNode]) -> str:
    fn = x.fullname
    if callable(fn):  # pragma: no cover
        return fn()
    return fn
