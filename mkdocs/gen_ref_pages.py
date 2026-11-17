"""Generate the per-object pandera API reference pages.

When the MkDocs build runs, the ``gen-files`` plugin executes this script to
create one Markdown page per public API object at
``reference/generated/<fully.qualified.name>.md``. Material for MkDocs +
mkdocstrings renders each page from the object's numpy-style docstring.

The object lists here are transcribed from the historical Sphinx
``autosummary`` blocks that used to live in ``docs/source/reference/*.rst``
(the section pages that used to host them are the ``reference/*.md`` files in
this directory). Keeping the same fully-qualified names preserves the exact
``/en/latest/reference/generated/<name>.html`` URLs of the old site, so
existing links and readthedocs redirects keep working.

Each section maps to a list of fully-qualified object names. `gen-files` does
not need a navigation entry: the ``reference/*.md`` section pages link to
these detail pages directly.
"""

from __future__ import annotations

import mkdocs_gen_files


# object name -> short human title used as the page heading
def _objects(*names: str) -> list[str]:
    return list(names)


GROUPS = {
    "core": _objects(
        # Schemas
        "pandera.api.pandas.container.DataFrameSchema",
        "pandera.api.geopandas.container.GeoDataFrameSchema",
        "pandera.api.pandas.array.SeriesSchema",
        "pandera.api.polars.container.DataFrameSchema",
        "pandera.api.pyspark.container.DataFrameSchema",
        "pandera.api.dataframe.container.DataFrameSchema",
        # Schema components
        "pandera.api.pandas.components.Column",
        "pandera.api.pandas.components.Index",
        "pandera.api.pandas.components.MultiIndex",
        "pandera.api.polars.components.Column",
        "pandera.api.pyspark.components.Column",
        "pandera.api.dataframe.components.ComponentSchema",
        # Checks
        "pandera.api.checks.Check",
        "pandera.api.hypotheses.Hypothesis",
        # Data objects
        "pandera.api.polars.types.PolarsData",
        "pandera.api.pyspark.types.PysparkDataframeColumnObject",
        # Configuration
        "pandera.config.PanderaConfig",
        "pandera.config.ValidationDepth",
        "pandera.config.ValidationScope",
        "pandera.config.config_context",
        "pandera.config.get_config_context",
        "pandera.config.get_config_global",
        "pandera.config.reset_config_context",
        "pandera.config.set_config",
        "pandera.set_config",
    ),
    "dataframe_models": _objects(
        "pandera.api.pandas.model.DataFrameModel",
        "pandera.api.polars.model.DataFrameModel",
        "pandera.api.pyspark.model.DataFrameModel",
        "pandera.api.dataframe.model.DataFrameModel",
        "pandera.api.dataframe.model_components.Field",
        "pandera.api.dataframe.model_components.check",
        "pandera.api.dataframe.model_components.dataframe_check",
        "pandera.api.dataframe.model_components.parser",
        "pandera.api.dataframe.model_components.dataframe_parser",
        "pandera.api.pandas.model_config.BaseConfig",
        "pandera.api.polars.model_config.BaseConfig",
        "pandera.api.pyspark.model_config.BaseConfig",
        "pandera.typing.DataFrame",
        "pandera.typing.Series",
        "pandera.typing.Index",
        "pandera.typing.FieldType",
        "pandera.typing.geopandas.GeoDataFrame",
        "pandera.typing.geopandas.GeoSeries",
        "pandera.typing.dask.DataFrame",
        "pandera.typing.dask.Series",
        "pandera.typing.dask.Index",
        "pandera.typing.pyspark.DataFrame",
        "pandera.typing.pyspark.Series",
        "pandera.typing.pyspark.Index",
        "pandera.typing.modin.DataFrame",
        "pandera.typing.modin.Series",
        "pandera.typing.modin.Index",
        "pandera.typing.fastapi.UploadFile",
        "pandera.typing.formats.Formats",
    ),
    "decisions": _objects(
        "pandera.decisions.Choice",
        "pandera.decisions.Score",
        "pandera.decisions.Noul",
        "pandera.decisions.Confidence",
        "pandera.decisions.DecisionProvider",
        "pandera.decisions.MockProvider",
        "pandera.decisions.RecordingProvider",
        "pandera.decisions.ReplayProvider",
        "pandera.decisions.set_provider",
        "pandera.decisions.get_provider",
        "pandera.decisions.provider",
        "pandera.decisions.register_provider",
        "pandera.decisions.capabilities_of",
        "pandera.decisions.enabled",
        "pandera.decisions.Holds",
        "pandera.decisions.holds",
        "pandera.decisions.MemoryCache",
        "pandera.decisions.SQLiteCache",
        "pandera.decisions.Plan",
        "pandera.decisions.questions",
        "pandera.decisions.plan",
        "pandera.decisions.stats",
        "pandera.decisions.Decision",
        "pandera.decisions.ProviderCapabilities",
        "pandera.decisions.ProviderLimits",
        "pandera.decisions.DecisionsConfigError",
        "pandera.decisions.ProviderCapabilityError",
    ),
    "decorators": _objects(
        "pandera.decorators.check_input",
        "pandera.decorators.check_output",
        "pandera.decorators.check_io",
        "pandera.decorators.check_types",
    ),
    "dtypes": _objects(
        # Library-agnostic dtypes
        "pandera.dtypes.DataType",
        "pandera.dtypes.Bool",
        "pandera.dtypes.Timestamp",
        "pandera.dtypes.DateTime",
        "pandera.dtypes.Timedelta",
        "pandera.dtypes.Category",
        "pandera.dtypes.Float",
        "pandera.dtypes.Float16",
        "pandera.dtypes.Float32",
        "pandera.dtypes.Float64",
        "pandera.dtypes.Float128",
        "pandera.dtypes.Int",
        "pandera.dtypes.Int8",
        "pandera.dtypes.Int16",
        "pandera.dtypes.Int32",
        "pandera.dtypes.Int64",
        "pandera.dtypes.UInt",
        "pandera.dtypes.UInt8",
        "pandera.dtypes.UInt16",
        "pandera.dtypes.UInt32",
        "pandera.dtypes.UInt64",
        "pandera.dtypes.Complex",
        "pandera.dtypes.Complex64",
        "pandera.dtypes.Complex128",
        "pandera.dtypes.Complex256",
        "pandera.dtypes.Decimal",
        "pandera.dtypes.String",
        # Pandas dtypes
        "pandera.engines.pandas_engine.BOOL",
        "pandera.engines.pandas_engine.INT8",
        "pandera.engines.pandas_engine.INT16",
        "pandera.engines.pandas_engine.INT32",
        "pandera.engines.pandas_engine.INT64",
        "pandera.engines.pandas_engine.UINT8",
        "pandera.engines.pandas_engine.UINT16",
        "pandera.engines.pandas_engine.UINT32",
        "pandera.engines.pandas_engine.UINT64",
        "pandera.engines.pandas_engine.STRING",
        "pandera.engines.numpy_engine.Object",
        "pandera.engines.pandas_engine.Decimal",
        "pandera.engines.pandas_engine.Category",
        "pandera.engines.pandas_engine.NpString",
        "pandera.engines.pandas_engine.DateTime",
        "pandera.engines.pandas_engine.Date",
        "pandera.engines.pandas_engine.Period",
        "pandera.engines.pandas_engine.Sparse",
        "pandera.engines.pandas_engine.Interval",
        "pandera.engines.pandas_engine.PydanticModel",
        "pandera.engines.pandas_engine.PythonDict",
        "pandera.engines.pandas_engine.PythonList",
        "pandera.engines.pandas_engine.PythonTuple",
        "pandera.engines.pandas_engine.PythonTypedDict",
        "pandera.engines.pandas_engine.PythonNamedTuple",
        # PyArrow dtypes
        "pandera.engines.pandas_engine.ArrowBool",
        "pandera.engines.pandas_engine.ArrowInt64",
        "pandera.engines.pandas_engine.ArrowInt32",
        "pandera.engines.pandas_engine.ArrowInt16",
        "pandera.engines.pandas_engine.ArrowInt8",
        "pandera.engines.pandas_engine.ArrowString",
        "pandera.engines.pandas_engine.ArrowUInt64",
        "pandera.engines.pandas_engine.ArrowUInt32",
        "pandera.engines.pandas_engine.ArrowUInt16",
        "pandera.engines.pandas_engine.ArrowUInt8",
        "pandera.engines.pandas_engine.ArrowFloat64",
        "pandera.engines.pandas_engine.ArrowFloat32",
        "pandera.engines.pandas_engine.ArrowFloat16",
        "pandera.engines.pandas_engine.ArrowDecimal128",
        "pandera.engines.pandas_engine.ArrowTimestamp",
        "pandera.engines.pandas_engine.ArrowDictionary",
        "pandera.engines.pandas_engine.ArrowList",
        "pandera.engines.pandas_engine.ArrowStruct",
        "pandera.engines.pandas_engine.ArrowNull",
        "pandera.engines.pandas_engine.ArrowDate32",
        "pandera.engines.pandas_engine.ArrowDate64",
        "pandera.engines.pandas_engine.ArrowDuration",
        "pandera.engines.pandas_engine.ArrowTime32",
        "pandera.engines.pandas_engine.ArrowTime64",
        "pandera.engines.pandas_engine.ArrowBinary",
        "pandera.engines.pandas_engine.ArrowLargeBinary",
        "pandera.engines.pandas_engine.ArrowLargeString",
        # GeoPandas dtypes
        "pandera.engines.geopandas_engine.Geometry",
        # Polars dtypes
        "pandera.engines.polars_engine.Int8",
        "pandera.engines.polars_engine.Int16",
        "pandera.engines.polars_engine.Int32",
        "pandera.engines.polars_engine.Int64",
        "pandera.engines.polars_engine.UInt8",
        "pandera.engines.polars_engine.UInt16",
        "pandera.engines.polars_engine.UInt32",
        "pandera.engines.polars_engine.UInt64",
        "pandera.engines.polars_engine.Float32",
        "pandera.engines.polars_engine.Float64",
        "pandera.engines.polars_engine.Decimal",
        "pandera.engines.polars_engine.Date",
        "pandera.engines.polars_engine.DateTime",
        "pandera.engines.polars_engine.Time",
        "pandera.engines.polars_engine.Timedelta",
        "pandera.engines.polars_engine.Array",
        "pandera.engines.polars_engine.List",
        "pandera.engines.polars_engine.Struct",
        "pandera.engines.polars_engine.Bool",
        "pandera.engines.polars_engine.String",
        "pandera.engines.polars_engine.Enum",
        "pandera.engines.polars_engine.Categorical",
        "pandera.engines.polars_engine.Category",
        "pandera.engines.polars_engine.Binary",
        "pandera.engines.polars_engine.Null",
        "pandera.engines.polars_engine.Object",
        # Utility functions
        "pandera.dtypes.is_subdtype",
        "pandera.dtypes.is_float",
        "pandera.dtypes.is_int",
        "pandera.dtypes.is_uint",
        "pandera.dtypes.is_complex",
        "pandera.dtypes.is_numeric",
        "pandera.dtypes.is_bool",
        "pandera.dtypes.is_string",
        "pandera.dtypes.is_datetime",
        "pandera.dtypes.is_timedelta",
        "pandera.dtypes.immutable",
        # Xarray dtypes
        "pandera.engines.xarray_engine.DataType",
        # Engines
        "pandera.engines.engine.Engine",
        "pandera.engines.numpy_engine.Engine",
        "pandera.engines.pandas_engine.Engine",
        "pandera.engines.polars_engine.Engine",
        "pandera.engines.pyspark_engine.Engine",
        "pandera.engines.xarray_engine.Engine",
    ),
    "errors": _objects(
        "pandera.errors.SchemaError",
        "pandera.errors.SchemaErrors",
        "pandera.errors.SchemaInitError",
        "pandera.errors.SchemaDefinitionError",
    ),
    "extensions": _objects(
        "pandera.extensions",
    ),
    "geopandas": _objects(
        "pandera.geopandas.GeoDataFrameModel",
        "pandera.geopandas.GeoDataFrameSchema",
    ),
    "io": _objects(
        "pandera.io.pandas_io.from_yaml",
        "pandera.io.pandas_io.to_yaml",
        "pandera.io.pandas_io.to_script",
        "pandera.io.pandas_io.from_json",
        "pandera.io.pandas_io.to_json",
        "pandera.io.polars_io.from_yaml",
        "pandera.io.polars_io.to_yaml",
        "pandera.io.polars_io.from_json",
        "pandera.io.polars_io.to_json",
        "pandera.io.pyspark_sql_io.from_yaml",
        "pandera.io.pyspark_sql_io.to_yaml",
        "pandera.io.pyspark_sql_io.from_json",
        "pandera.io.pyspark_sql_io.to_json",
        "pandera.io.ibis_io.from_yaml",
        "pandera.io.ibis_io.to_yaml",
        "pandera.io.ibis_io.from_json",
        "pandera.io.ibis_io.to_json",
        "pandera.io.xarray_io.to_yaml",
        "pandera.io.xarray_io.from_yaml",
        "pandera.io.xarray_io.to_json",
        "pandera.io.xarray_io.from_json",
    ),
    "narwhals": _objects(
        "pandera.api.narwhals.types.NarwhalsData",
        "pandera.api.narwhals.types.NarwhalsCheckResult",
        "pandera.backends.narwhals.base.NarwhalsSchemaBackend",
        "pandera.backends.narwhals.container.DataFrameSchemaBackend",
        "pandera.backends.narwhals.components.ColumnBackend",
        "pandera.backends.narwhals.checks.NarwhalsCheckBackend",
        "pandera.engines.narwhals_engine.DataType",
        "pandera.engines.narwhals_engine.Int8",
        "pandera.engines.narwhals_engine.Int16",
        "pandera.engines.narwhals_engine.Int32",
        "pandera.engines.narwhals_engine.Int64",
        "pandera.engines.narwhals_engine.UInt8",
        "pandera.engines.narwhals_engine.UInt16",
        "pandera.engines.narwhals_engine.UInt32",
        "pandera.engines.narwhals_engine.UInt64",
        "pandera.engines.narwhals_engine.Float32",
        "pandera.engines.narwhals_engine.Float64",
        "pandera.engines.narwhals_engine.String",
        "pandera.engines.narwhals_engine.Bool",
        "pandera.engines.narwhals_engine.Date",
        "pandera.engines.narwhals_engine.DateTime",
        "pandera.engines.narwhals_engine.Duration",
        "pandera.engines.narwhals_engine.Categorical",
        "pandera.engines.narwhals_engine.List",
        "pandera.engines.narwhals_engine.Struct",
        "pandera.engines.narwhals_engine.Engine",
        "pandera.backends.narwhals.builtin_checks.equal_to",
        "pandera.backends.narwhals.builtin_checks.not_equal_to",
        "pandera.backends.narwhals.builtin_checks.greater_than",
        "pandera.backends.narwhals.builtin_checks.greater_than_or_equal_to",
        "pandera.backends.narwhals.builtin_checks.less_than",
        "pandera.backends.narwhals.builtin_checks.less_than_or_equal_to",
        "pandera.backends.narwhals.builtin_checks.in_range",
        "pandera.backends.narwhals.builtin_checks.isin",
        "pandera.backends.narwhals.builtin_checks.notin",
        "pandera.backends.narwhals.builtin_checks.str_matches",
        "pandera.backends.narwhals.builtin_checks.str_contains",
        "pandera.backends.narwhals.builtin_checks.str_startswith",
        "pandera.backends.narwhals.builtin_checks.str_endswith",
        "pandera.backends.narwhals.builtin_checks.str_length",
    ),
    "parsers": _objects(
        "pandera.api.parsers.Parser",
        "pandera.api.parsers.ColumnParser",
        "pandera.api.parsers.ParseContext",
        "pandera.api.pandas.components.ParsedColumn",
        "pandera.api.dataframe.model_components.ParsedField",
        "pandera.api.dataframe.model_components.ParsedFieldInfo",
    ),
    "pytorch": _objects(
        "pandera.api.tensordict.container.TensorDictSchema",
        "pandera.api.tensordict.components.Tensor",
        "pandera.api.tensordict.model.TensorDictModel",
        "pandera.api.tensordict.model_components.Field",
        "pandera.config.ValidationDepth",
        "pandera.config.ValidationScope",
    ),
    "schema_inference": _objects(
        "pandera.schema_inference.pandas.infer_schema",
    ),
    "strategies": _objects(
        "pandera.strategies.pandas_strategies",
    ),
    "xarray": _objects(
        "pandera.api.xarray.container.DataArraySchema",
        "pandera.api.xarray.container.DatasetSchema",
        "pandera.api.xarray.container.DataTreeSchema",
        "pandera.api.xarray.components.DataVar",
        "pandera.api.xarray.components.Coordinate",
        "pandera.api.xarray.model.DataArrayModel",
        "pandera.api.xarray.model.DatasetModel",
        "pandera.api.xarray.model.DataTreeModel",
        "pandera.api.xarray.model_components.Field",
        "pandera.typing.xarray.Coordinate",
        "pandera.typing.xarray.DataArray",
        "pandera.typing.xarray.Dataset",
        "pandera.typing.xarray.DataTree",
        "pandera.typing.xarray.XarrayAnnotationBase",
        "pandera.api.xarray.model_config.DataArrayConfig",
        "pandera.api.xarray.model_config.DatasetConfig",
        "pandera.api.xarray.model_config.DataTreeConfig",
        "pandera.api.xarray.base.BaseDataArraySchema",
        "pandera.api.xarray.base.BaseDatasetSchema",
        "pandera.api.xarray.base.BaseDataTreeSchema",
        "pandera.api.xarray.types.XarrayData",
        "pandera.api.xarray.types.XARRAY_CHECK_OBJECT_TYPES",
        "pandera.api.xarray.types.XarrayCheckObjects",
        "pandera.api.xarray.utils.get_validation_depth",
        "pandera.schema_inference.xarray.infer_schema",
        # xarray hypothesis strategies (also referenced from strategies page)
        "pandera.strategies.xarray_strategies.xarray_dtype_strategy",
        "pandera.strategies.xarray_strategies.data_array_strategy",
        "pandera.strategies.xarray_strategies.dataset_strategy",
        "pandera.strategies.xarray_strategies.data_array_schema_strategy",
        "pandera.strategies.xarray_strategies.dataset_schema_strategy",
    ),
}


PAGE_TITLES = {
    "core": "Core",
    "dataframe_models": "DataFrame Models",
    "decisions": "Decision Parsing",
    "decorators": "Decorators",
    "dtypes": "Data Types",
    "errors": "Errors",
    "extensions": "Extensions",
    "geopandas": "GeoPandas",
    "io": "IO Utilities",
    "narwhals": "Narwhals Backend",
    "parsers": "Parsers and Derived Columns",
    "pytorch": "PyTorch",
    "schema_inference": "Schema Inference",
    "strategies": "Data Synthesis Strategies",
    "xarray": "Xarray",
}

# Objects rendered as bare module pages rather than single-object pages.
_MODULES = {
    "pandera.extensions",
    "pandera.strategies.pandas_strategies",
}

# Base page (section) each generated stub belongs to, used for cross-linking.
_BASE = {}
for _section, _objs in GROUPS.items():
    for _o in _objs:
        _BASE.setdefault(_o, _section)


def generate() -> None:
    for _section, names in GROUPS.items():
        for fullname in names:
            short = fullname.rsplit(".", 1)[-1]
            _render(fullname, short, module=_MODULES.__contains__(fullname))


def _render(fullname: str, short: str, module: bool) -> None:
    path = f"reference/generated/{fullname}.md"
    lines = [
        f"# {fullname}",
        "",
    ]
    if module:
        lines.append(f"::: {fullname}")
    else:
        lines.append(f"::: {fullname}")
    lines.append("")
    with mkdocs_gen_files.open(path, "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    generate()
