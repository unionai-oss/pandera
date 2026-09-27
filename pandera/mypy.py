"""Pandera mypy plugin."""

from collections.abc import Callable
from typing import Optional, Union, cast

from mypy.nodes import (
    AssignmentStmt,
    FuncBase,
    IndexExpr,
    NameExpr,
    StrExpr,
    SymbolNode,
    TypeInfo,
    Var,
)
from mypy.plugin import (
    AttributeContext,
    ClassDefContext,
    FunctionSigContext,
    MethodContext,
    MethodSigContext,
    Plugin,
)
from mypy.types import (
    AnyType,
    CallableType,
    Instance,
    LiteralType,
    Type,
    TypeOfAny,
    get_proper_type,
)

DATAFRAMEMODEL_FULLNAMES = {
    "pandera.api.dataframe.model.DataFrameModel",
    "pandera.api.pandas.model.DataFrameModel",
    "pandera.api.polars.model.DataFrameModel",
    "pandera.api.geopandas.GeoDataFrameModel",
    "pandera.api.geopandas.model.GeoDataFrameModel",
    "pandera.pandas.DataFrameModel",
    "pandera.polars.DataFrameModel",
    "pandera.geopandas.GeoDataFrameModel",
    "pandera._pandas_deprecated.DataFrameModel",
}
PANDERA_PANDAS_DATAFRAME_FULLNAME = "pandera.typing.pandas.DataFrame"
PANDERA_PANDAS_SERIES_FULLNAME = "pandera.typing.pandas.Series"
PANDERA_PANDAS_INDEX_FULLNAME = "pandera.typing.pandas.Index"
PANDERA_POLARS_DATAFRAME_FULLNAME = "pandera.typing.polars.DataFrame"
PANDERA_POLARS_SERIES_FULLNAME = "pandera.typing.polars.Series"
PANDERA_MODIN_SERIES_FULLNAME = "pandera.typing.modin.Series"
PANDERA_MODIN_INDEX_FULLNAME = "pandera.typing.modin.Index"
PANDERA_DASK_SERIES_FULLNAME = "pandera.typing.dask.Series"
PANDERA_DASK_INDEX_FULLNAME = "pandera.typing.dask.Index"
PANDERA_PYSPARK_SERIES_FULLNAME = "pandera.typing.pyspark.Series"
PANDERA_PYSPARK_INDEX_FULLNAME = "pandera.typing.pyspark.Index"
PANDERA_GEOPANDAS_SERIES_FULLNAME = "pandera.typing.geopandas.GeoSeries"
PANDAS_CONCAT = "pandas.core.reshape.concat.concat"
PANDAS_DATAFRAME_GETITEM = "pandas.core.frame.DataFrame.__getitem__"
PANDAS_NDFRAME_GETITEM = "pandas.core.generic.NDFrame.__getitem__"
PANDAS_NDFRAME = "pandas.core.generic.NDFrame"
PANDERA_DATAFRAME_ATTR_CLASSES = frozenset(
    {
        PANDAS_NDFRAME,
        "pandas.core.frame.DataFrame",
        PANDERA_PANDAS_DATAFRAME_FULLNAME,
    }
)
PANDERA_DATAFRAME_GETITEM_CLASSES = frozenset(
    {
        PANDERA_PANDAS_DATAFRAME_FULLNAME,
        PANDERA_POLARS_DATAFRAME_FULLNAME,
        "pandas.core.frame.DataFrame",
        PANDAS_NDFRAME,
    }
)
PANDERA_DATAFRAME_SERIES_FULLNAMES = {
    PANDERA_PANDAS_DATAFRAME_FULLNAME: PANDERA_PANDAS_SERIES_FULLNAME,
    PANDERA_POLARS_DATAFRAME_FULLNAME: PANDERA_POLARS_SERIES_FULLNAME,
}
FIELD_TYPE_METADATA_KEY = "pandera_field_types"

FIELD_GENERICS_FULLNAMES = {
    PANDERA_PANDAS_SERIES_FULLNAME,
    PANDERA_PANDAS_INDEX_FULLNAME,
    PANDERA_POLARS_SERIES_FULLNAME,
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
    ) -> "Callable[[AttributeContext], Type] | None":
        if self._is_dataframe_model_field_attribute(fullname):
            return self._dataframe_model_class_attr_callback

        class_fullname, _, attr_name = fullname.rpartition(".")
        if (
            class_fullname in PANDERA_DATAFRAME_ATTR_CLASSES
            and attr_name
            and not attr_name.startswith("_")
        ):
            return self._make_dataframe_attribute_callback(attr_name)
        return None

    def get_method_signature_hook(
        self, fullname: str
    ) -> "Callable[[MethodSigContext], CallableType] | None":
        if _is_dataframe_getitem_hook(fullname):
            return self._dataframe_getitem_signature_hook
        return None

    def get_method_hook(
        self, fullname: str
    ) -> "Callable[[MethodContext], Type] | None":
        if _is_dataframe_getitem_hook(fullname):
            return self._dataframe_getitem_hook
        return None

    def _resolve_dataframe_getitem_type(
        self,
        self_type: Type,
        key_type: Type | None,
        api,
        *,
        context=None,
        key_exprs=None,
    ) -> Type | None:
        column_name = _extract_string_column_key(key_type, context, key_exprs)
        if column_name is None:
            return None

        schema_type, series_fullname = _extract_pandera_dataframe_schema(
            self_type
        )
        if schema_type is None or series_fullname is None:
            return None

        field_type = _get_model_field_type(schema_type, column_name)
        if field_type is None:
            return AnyType(TypeOfAny.implementation_artifact)

        return _to_series_type(field_type, series_fullname, api)

    def _dataframe_getitem_hook(self, ctx: MethodContext) -> Type:
        key_type = None
        key_exprs = None
        if len(ctx.arg_types) >= 2 and ctx.arg_types[1]:
            key_type = get_proper_type(ctx.arg_types[1][0])
        if len(ctx.args) >= 2 and ctx.args[1]:
            key_exprs = ctx.args[1]

        ret_type = self._resolve_dataframe_getitem_type(
            get_proper_type(ctx.type),
            key_type,
            ctx.api,
            context=ctx.context,
            key_exprs=key_exprs,
        )
        return ret_type if ret_type is not None else ctx.default_return_type

    def _make_dataframe_attribute_callback(
        self, attr_name: str
    ) -> "Callable[[AttributeContext], Type]":
        def _callback(ctx: AttributeContext) -> Type:
            return self._resolve_dataframe_column_type(ctx, attr_name)

        return _callback

    def _resolve_dataframe_column_type(
        self, ctx: AttributeContext, column_name: str
    ) -> Type:
        schema_type, series_fullname = _extract_pandera_dataframe_schema(
            get_proper_type(ctx.type)
        )
        if schema_type is None or series_fullname is None:
            return ctx.default_attr_type

        field_type = _get_model_field_type(schema_type, column_name)
        if field_type is None:
            return AnyType(TypeOfAny.implementation_artifact)

        ret_type = _to_series_type(field_type, series_fullname, ctx.api)
        return ret_type if ret_type is not None else ctx.default_attr_type

    def _dataframe_getitem_signature_hook(
        self, ctx: MethodSigContext
    ) -> CallableType:
        signature = ctx.default_signature
        key_exprs = ctx.args[1] if len(ctx.args) >= 2 else None

        ret_type = self._resolve_dataframe_getitem_type(
            get_proper_type(ctx.type),
            None,
            ctx.api,
            context=ctx.context,
            key_exprs=key_exprs,
        )
        if ret_type is None:
            return signature

        return signature.copy_modified(ret_type=ret_type)

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
        self.store_field_types()
        self.erase_field_type_arg()
        self.set_field_type_to_str()

    def store_field_types(self) -> None:
        """Store original field types on the model TypeInfo for column lookup."""
        field_types: dict[str, Type] = {}
        for base in reversed(self.ctx.cls.info.mro):
            if isinstance(base, TypeInfo):
                field_types.update(
                    base.metadata.get(FIELD_TYPE_METADATA_KEY, {})
                )

        for _def, var in self._iter_model_field_vars():
            if _def is not None and _def.type is not None:
                field_types[_def.lvalues[0].name] = self._normalize_field_type(
                    _def.type
                )
            elif var is not None and var.type is not None:
                field_types[var.name] = self._normalize_field_type(var.type)

        if field_types:
            self.ctx.cls.info.metadata[FIELD_TYPE_METADATA_KEY] = field_types

    def _normalize_field_type(self, type_: Type) -> Type:
        """Normalize model field annotations to Series[..., ...] form."""
        type_ = get_proper_type(type_)
        if isinstance(type_, Instance):
            if type_.type.fullname in FIELD_GENERICS_FULLNAMES:
                return type_
            series_node = self.ctx.api.lookup_fully_qualified(
                PANDERA_PANDAS_SERIES_FULLNAME
            )
            if series_node is not None and isinstance(
                series_node.node, TypeInfo
            ):
                return Instance(series_node.node, [type_])
        return type_

    def _iter_model_field_vars(self):
        """Yield model field assignment statements and class attribute vars."""
        seen: set[str] = set()
        for def_ in self.ctx.cls.defs.body:
            if not isinstance(def_, AssignmentStmt):
                continue
            if len(def_.lvalues) != 1:
                continue
            field_name_expr = def_.lvalues[0]
            if not isinstance(field_name_expr, NameExpr):
                continue
            field_name = field_name_expr.name
            if not _is_model_field_name(field_name):
                continue
            if def_.type is None:
                continue
            seen.add(field_name)
            symbol_node = field_name_expr.node
            var = symbol_node if isinstance(symbol_node, Var) else None
            yield def_, var

        for name, sym in self.ctx.cls.info.names.items():
            if not _is_model_field_name(name) or name in seen:
                continue
            if isinstance(sym.node, Var) and sym.type is not None:
                yield None, sym.node

    def _get_field_assignments(self):
        """Get DataFrameModel field assignment statements."""
        for def_, var in self._iter_model_field_vars():
            if def_ is not None:
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
        for def_, var in self._iter_model_field_vars():
            if def_ is None:
                continue
            type_ = def_.type
            if (
                # e.g. UnionType does not have module_name or name
                not hasattr(type_, "module_name") or not hasattr(type_, "name")
            ):
                continue
            if (
                isinstance(type_, Instance)
                and type_.type.fullname in FIELD_GENERICS_FULLNAMES
            ):
                type_.args = ()  # erase generic type arg
                if var is not None and hasattr(var.type, "args"):
                    var.type.args = ()

    def set_field_type_to_str(self) -> None:
        """Type DataFrameModel field class attributes as column names."""
        str_type = self.ctx.api.named_type("builtins.str")
        for def_, var in self._iter_model_field_vars():
            if def_ is not None:
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


def _is_model_field_name(name: str) -> bool:
    return not name.startswith("_") and name != "Config"


def _is_dataframe_getitem_hook(fullname: str) -> bool:
    if not fullname.endswith(".__getitem__"):
        return False
    class_fullname = fullname.rpartition(".")[0]
    return class_fullname in PANDERA_DATAFRAME_GETITEM_CLASSES


def _extract_string_column_key(
    key_type: Type | None, context, key_exprs=None
) -> str | None:
    """Extract a string column key from a literal type or index expression."""
    if key_type is not None:
        key_type = get_proper_type(key_type)
        if isinstance(key_type, LiteralType) and isinstance(
            key_type.value, str
        ):
            return key_type.value

    if key_exprs:
        for expr in key_exprs:
            if isinstance(expr, StrExpr):
                return expr.value

    if isinstance(context, IndexExpr) and isinstance(context.index, StrExpr):
        return context.index.value

    return None


def _extract_pandera_dataframe_schema(
    instance_type: Type,
) -> tuple[Instance | None, str | None]:
    """Return (schema type, series fullname) for a pandera dataframe instance."""
    instance_type = get_proper_type(instance_type)
    if not isinstance(instance_type, Instance):
        return None, None
    if not instance_type.args:
        return None, None

    dataframe_fullname = instance_type.type.fullname
    if dataframe_fullname not in PANDERA_DATAFRAME_SERIES_FULLNAMES:
        return None, None

    schema_type = get_proper_type(instance_type.args[0])
    if not isinstance(schema_type, Instance):
        return None, None

    return schema_type, PANDERA_DATAFRAME_SERIES_FULLNAMES[dataframe_fullname]


def _get_model_field_type(
    schema_type: Instance, column_name: str
) -> Type | None:
    field_types = schema_type.type.metadata.get(FIELD_TYPE_METADATA_KEY)
    if not field_types:
        return None
    return field_types.get(column_name)


def _to_series_type(
    field_type: Type, series_fullname: str, api
) -> Type | None:
    """Map a model field annotation to a pandera Series generic type."""
    field_type = get_proper_type(field_type)
    if isinstance(field_type, Instance):
        if field_type.type.fullname in FIELD_GENERICS_FULLNAMES:
            return field_type
    return field_type
