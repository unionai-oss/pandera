# Codebase Structure

**Analysis Date:** 2026-03-08

## Directory Layout

```
pandera/                          # Root of repository
├── pandera/                      # Main Python package
│   ├── __init__.py               # Top-level namespace (pandas-first, conditional)
│   ├── pandas.py                 # Full pandas API entry point
│   ├── polars.py                 # Full polars API entry point
│   ├── ibis.py                   # Full ibis API entry point
│   ├── pyspark.py                # Full pyspark API entry point
│   ├── dtypes.py                 # Framework-agnostic DataType base classes
│   ├── errors.py                 # All exception classes
│   ├── config.py                 # PanderaConfig, ValidationDepth, env var loading
│   ├── decorators.py             # @check_types, @check_input, @check_output, @check_io
│   ├── extensions.py             # Backwards-compat shim for pandera.api.extensions
│   ├── utils.py                  # Shared utility functions
│   ├── validation_depth.py       # Validation scope helpers
│   ├── inspection_utils.py       # Reflection helpers for decorators
│   ├── import_utils.py           # Optional dependency import helpers
│   ├── mypy.py                   # mypy plugin entrypoint
│   ├── external_config.py        # External configuration loading
│   ├── system.py                 # System info utilities
│   ├── api/                      # Framework-agnostic public API definitions
│   │   ├── base/                 # Abstract base classes
│   │   │   ├── checks.py         # BaseCheck, MetaCheck, CheckResult
│   │   │   ├── schema.py         # BaseSchema (BACKEND_REGISTRY, get_backend)
│   │   │   ├── model.py          # BaseModel, MetaModel
│   │   │   ├── model_components.py  # BaseFieldInfo
│   │   │   ├── model_config.py   # BaseModelConfig
│   │   │   ├── error_handler.py  # ErrorHandler, ErrorCategory
│   │   │   ├── parsers.py        # BaseParser
│   │   │   └── types.py          # CheckList, ParserList, StrictType
│   │   ├── dataframe/            # Generic dataframe schema (shared across pandas/polars/ibis)
│   │   │   ├── container.py      # DataFrameSchema[TDataObject] (1396 lines)
│   │   │   ├── components.py     # ComponentSchema base
│   │   │   ├── model.py          # DataFrameModel, MODEL_CACHE, GENERIC_SCHEMA_CACHE
│   │   │   ├── model_components.py  # Field, FieldInfo, CheckInfo, check, parser decorators
│   │   │   └── model_config.py   # BaseConfig for DataFrameModel
│   │   ├── pandas/               # Pandas-specific API
│   │   │   ├── array.py          # SeriesSchema
│   │   │   ├── components.py     # Column, Index, MultiIndex
│   │   │   ├── container.py      # DataFrameSchema[pd.DataFrame]
│   │   │   ├── model.py          # DataFrameModel (pandas)
│   │   │   ├── model_config.py   # pandas ModelConfig
│   │   │   └── types.py          # PandasDtypeInputTypes, is_table, is_field
│   │   ├── polars/               # Polars-specific API
│   │   │   ├── components.py     # Column (polars)
│   │   │   ├── container.py      # DataFrameSchema (polars)
│   │   │   ├── model.py          # DataFrameModel (polars)
│   │   │   ├── model_config.py   # polars ModelConfig
│   │   │   ├── types.py          # PolarsData, type helpers
│   │   │   └── utils.py          # Polars-specific utilities
│   │   ├── ibis/                 # Ibis-specific API
│   │   │   ├── components.py     # Column (ibis)
│   │   │   ├── container.py      # DataFrameSchema (ibis)
│   │   │   ├── error_handler.py  # Ibis error handling
│   │   │   ├── model.py          # DataFrameModel (ibis)
│   │   │   └── types.py          # Ibis type helpers
│   │   ├── pyspark/              # PySpark-specific API
│   │   │   ├── column_schema.py  # PySpark ColumnSchema
│   │   │   ├── components.py     # PySpark components
│   │   │   ├── container.py      # DataFrameSchema (pyspark)
│   │   │   ├── model.py          # DataFrameModel (pyspark)
│   │   │   ├── model_components.py
│   │   │   ├── model_config.py
│   │   │   └── types.py
│   │   ├── checks.py             # Check class (wraps BaseCheck)
│   │   ├── hypotheses.py         # Hypothesis class
│   │   ├── parsers.py            # Parser class
│   │   ├── extensions.py         # register_builtin_check, register_check_method
│   │   └── function_dispatch.py  # Dispatcher for multi-type check functions
│   ├── backends/                 # Validation execution backends (one per framework)
│   │   ├── base/
│   │   │   ├── __init__.py       # BaseSchemaBackend, BaseCheckBackend, BaseParserBackend, CoreCheckResult
│   │   │   ├── builtin_checks.py # Framework-agnostic built-in check implementations
│   │   │   └── builtin_hypotheses.py
│   │   ├── pandas/
│   │   │   ├── array.py          # SeriesSchemaBackend
│   │   │   ├── base.py           # PandasSchemaBackend (shared pandas logic)
│   │   │   ├── builtin_checks.py # Pandas built-in check implementations
│   │   │   ├── builtin_hypotheses.py
│   │   │   ├── checks.py         # PandasCheckBackend
│   │   │   ├── components.py     # ColumnBackend, IndexBackend, MultiIndexBackend
│   │   │   ├── container.py      # DataFrameSchemaBackend (850 lines)
│   │   │   ├── error_formatters.py  # reshape_failure_cases
│   │   │   ├── hypotheses.py     # PandasHypothesisBackend
│   │   │   ├── parsers.py        # PandasParserBackend
│   │   │   └── register.py       # register_pandas_backends() (lru_cache)
│   │   ├── polars/
│   │   │   ├── base.py
│   │   │   ├── builtin_checks.py
│   │   │   ├── checks.py
│   │   │   ├── components.py
│   │   │   ├── container.py
│   │   │   ├── error_formatters.py
│   │   │   └── register.py       # register_polars_backends()
│   │   ├── ibis/
│   │   │   ├── base.py
│   │   │   ├── builtin_checks.py
│   │   │   ├── checks.py
│   │   │   ├── components.py
│   │   │   ├── constants.py
│   │   │   ├── container.py
│   │   │   └── register.py
│   │   └── pyspark/
│   │       ├── base.py
│   │       ├── builtin_checks.py
│   │       ├── checks.py
│   │       ├── column.py
│   │       ├── components.py
│   │       ├── container.py
│   │       ├── decorators.py
│   │       ├── error_formatters.py
│   │       └── register.py
│   ├── engines/                  # Dtype registry and coercion per framework
│   │   ├── engine.py             # Engine metaclass, _DtypeRegistry, StrictEquivalent
│   │   ├── numpy_engine.py
│   │   ├── pandas_engine.py
│   │   ├── polars_engine.py
│   │   ├── ibis_engine.py
│   │   ├── pyarrow_engine.py
│   │   ├── pyspark_engine.py
│   │   ├── geopandas_engine.py
│   │   ├── type_aliases.py
│   │   └── utils.py
│   ├── typing/                   # Type annotation helpers for DataFrameModel fields
│   │   ├── common.py             # DataFrameBase, dtype type aliases (Int64, String, etc.)
│   │   ├── pandas.py             # DataFrame[Schema], Series[dtype] annotations
│   │   ├── polars.py             # DataFrame[Schema] for polars
│   │   ├── dask.py
│   │   ├── modin.py
│   │   ├── fastapi.py
│   │   ├── formats.py
│   │   ├── geopandas.py
│   │   ├── ibis.py
│   │   ├── pyspark.py
│   │   └── pyspark_sql.py
│   ├── accessors/                # Framework-native accessor extensions (df.pandera.schema)
│   │   ├── pandas_accessor.py    # PanderaDataFrameAccessor, PanderaSeriesAccessor
│   │   ├── polars_accessor.py
│   │   ├── dask_accessor.py
│   │   ├── modin_accessor.py
│   │   ├── pyspark_accessor.py
│   │   └── pyspark_sql_accessor.py
│   ├── io/                       # Schema serialization/deserialization
│   │   └── pandas_io.py          # YAML/JSON/Frictionless IO for pandas schemas
│   ├── schema_inference/         # Infer schema from existing data
│   │   └── pandas.py
│   ├── schema_statistics/        # Compute schema statistics from data
│   │   └── (pandas statistics helpers)
│   └── strategies/               # Hypothesis-based data generation strategies
│       ├── base_strategies.py    # STRATEGY_DISPATCHER, base strategy utilities
│       └── pandas_strategies.py  # Pandas-specific data generation strategies
├── tests/                        # Test suite, mirroring framework breakdown
│   ├── base/                     # Tests for base API layer
│   ├── pandas/                   # Pandas-specific tests
│   │   └── modules/              # Test helper modules
│   ├── polars/                   # Polars-specific tests
│   ├── ibis/                     # Ibis-specific tests
│   ├── pyspark/                  # PySpark-specific tests
│   ├── dask/                     # Dask-specific tests
│   ├── modin/                    # Modin-specific tests
│   ├── geopandas/                # GeoPandas-specific tests
│   ├── fastapi/                  # FastAPI integration tests
│   ├── hypotheses/               # Hypothesis strategy tests
│   ├── io/                       # Schema IO tests
│   ├── mypy/                     # mypy plugin tests
│   │   ├── config/               # mypy configs per test scenario
│   │   └── pandas_modules/       # Python modules used as mypy test inputs
│   └── strategies/               # Data synthesis strategy tests
├── docs/                         # Documentation source
├── .github/
│   └── workflows/                # CI/CD GitHub Actions workflows
└── pyproject.toml                # Package metadata, dependencies, tool config
```

## Directory Purposes

**`pandera/api/`:**
- Purpose: All user-facing schema specification classes; zero execution logic
- Contains: Abstract base classes (`base/`), generic dataframe schema (`dataframe/`), framework-specific schema classes (`pandas/`, `polars/`, `ibis/`, `pyspark/`)
- Key files: `pandera/api/base/schema.py`, `pandera/api/dataframe/container.py`, `pandera/api/dataframe/model.py`, `pandera/api/checks.py`

**`pandera/backends/`:**
- Purpose: All validation execution logic; one subdirectory per supported framework
- Contains: `DataFrameSchemaBackend`, `ColumnBackend`, check backends, parser backends, error formatters, and a `register.py` per framework
- Key files: `pandera/backends/pandas/container.py`, `pandera/backends/pandas/register.py`, `pandera/backends/base/__init__.py`

**`pandera/engines/`:**
- Purpose: Map native framework dtypes to pandera's `DataType` abstraction; handle coercion
- Contains: One engine module per framework; `engine.py` provides the abstract `Engine` metaclass
- Key files: `pandera/engines/engine.py`, `pandera/engines/pandas_engine.py`, `pandera/engines/polars_engine.py`

**`pandera/typing/`:**
- Purpose: Generic type aliases used in `DataFrameModel` field annotations and for mypy/Pydantic compatibility
- Contains: `DataFrame[Schema]` and `Series[dtype]` generic aliases per framework
- Key files: `pandera/typing/common.py`, `pandera/typing/pandas.py`

**`pandera/accessors/`:**
- Purpose: Register `pandera` as a native accessor namespace on data objects (e.g. `df.pandera.schema`)
- Contains: One accessor file per framework; uses framework-native accessor registration APIs

**`pandera/io/`:**
- Purpose: Schema serialization to YAML/JSON; schema deserialization from YAML/JSON/Frictionless
- Key files: `pandera/io/pandas_io.py`

**`pandera/schema_inference/`:**
- Purpose: Automatically infer a `DataFrameSchema` from an existing DataFrame
- Key files: `pandera/schema_inference/pandas.py`

**`pandera/strategies/`:**
- Purpose: Generate synthetic data satisfying a schema using the `hypothesis` library
- Key files: `pandera/strategies/base_strategies.py`, `pandera/strategies/pandas_strategies.py`

**`tests/`:**
- Purpose: All tests, organized by framework to match the package structure
- Contains: `pytest`-based test modules; mypy test inputs under `tests/mypy/`

## Key File Locations

**Entry Points:**
- `pandera/__init__.py`: Top-level import namespace (conditionally pandas-first)
- `pandera/pandas.py`: Full pandas API; triggers backend registration on import
- `pandera/polars.py`: Full polars API; calls `register_polars_backends()` eagerly
- `pandera/ibis.py`: Full ibis API
- `pandera/pyspark.py`: Full pyspark API
- `pandera/decorators.py`: `@check_types`, `@check_input`, `@check_output`, `@check_io`

**Configuration:**
- `pandera/config.py`: `PanderaConfig`, `ValidationDepth`, `ValidationScope`, `get_config_context()`
- `pyproject.toml`: All package metadata, optional dependency groups, tool settings

**Core Schema Logic:**
- `pandera/api/base/schema.py`: `BaseSchema` with `BACKEND_REGISTRY` and `get_backend()`
- `pandera/api/dataframe/container.py`: Generic `DataFrameSchema[TDataObject]`
- `pandera/api/dataframe/model.py`: `DataFrameModel`, `MODEL_CACHE`, `to_schema()`
- `pandera/api/checks.py`: `Check` class
- `pandera/dtypes.py`: `DataType` abstract base

**Backend Registration:**
- `pandera/backends/pandas/register.py`: `register_pandas_backends()` — maps schema+data types to backend classes
- `pandera/backends/polars/register.py`: `register_polars_backends()`
- `pandera/backends/ibis/register.py`: `register_ibis_backends()`
- `pandera/backends/pyspark/register.py`: `register_pyspark_backends()`

**Validation Execution:**
- `pandera/backends/pandas/container.py`: `DataFrameSchemaBackend.validate()` — main pandas validation pipeline
- `pandera/backends/polars/container.py`: Polars equivalent
- `pandera/api/base/error_handler.py`: `ErrorHandler` — error accumulation and categorization

**Testing:**
- `tests/pandas/`: Core pandas validation tests
- `tests/polars/`: Polars validation tests
- `tests/ibis/`: Ibis validation tests
- `tests/mypy/`: Static type checking tests via mypy plugin

## Naming Conventions

**Files:**
- `container.py`: The `DataFrameSchema` class for a given framework (api or backend)
- `components.py`: `Column`, `Index`, `MultiIndex` classes for a given framework
- `array.py`: `SeriesSchema` class (pandas only)
- `register.py`: Backend registration function (e.g. `register_pandas_backends()`)
- `builtin_checks.py`: Built-in check function implementations for a framework
- `error_formatters.py`: Failure case reshaping utilities for a framework
- `*_engine.py`: Engine subclass for a specific framework in `pandera/engines/`
- `*_accessor.py`: Accessor class for a specific framework in `pandera/accessors/`
- `*_strategies.py`: Data synthesis strategies for a specific framework

**Classes:**
- Schema specs: `DataFrameSchema`, `SeriesSchema`, `Column`, `Index`, `MultiIndex`
- Backend implementations: `DataFrameSchemaBackend`, `ColumnBackend`, `SeriesSchemaBackend`, `PandasCheckBackend`
- Engine classes: `Engine` (per-framework singleton via metaclass)
- Model classes: `DataFrameModel` (per framework, all named identically)

**Directories:**
- Each framework gets a matching subdirectory in both `pandera/api/` and `pandera/backends/`
- Framework names used consistently: `pandas`, `polars`, `ibis`, `pyspark`

## Where to Add New Code

**New validation check (built-in):**
- Register the check function: `pandera/backends/<framework>/builtin_checks.py` using `@register_builtin_check`
- Add data synthesis strategy: `pandera/strategies/<framework>_strategies.py`
- Tests: `tests/<framework>/test_checks.py` or existing check test file

**New schema component (e.g. new index type):**
- API class: `pandera/api/<framework>/components.py`
- Backend class: `pandera/backends/<framework>/components.py`
- Register backend in: `pandera/backends/<framework>/register.py`

**New supported framework:**
- API directory: `pandera/api/<framework>/` with `container.py`, `components.py`, `model.py`, `types.py`
- Backend directory: `pandera/backends/<framework>/` with `container.py`, `components.py`, `checks.py`, `register.py`
- Engine: `pandera/engines/<framework>_engine.py`
- Typing: `pandera/typing/<framework>.py`
- Accessor: `pandera/accessors/<framework>_accessor.py`
- Entry module: `pandera/<framework>.py`
- Tests: `tests/<framework>/`

**New dtype:**
- Add to `pandera/dtypes.py` (abstract definition)
- Register in appropriate engine: `pandera/engines/<framework>_engine.py`

**New schema serialization format:**
- Add to `pandera/io/pandas_io.py` (currently pandas only) or create `pandera/io/<framework>_io.py`

**Shared utility functions:**
- Pure utilities with no framework dependency: `pandera/utils.py`
- Import helpers for optional dependencies: `pandera/import_utils.py`

## Special Directories

**`pandera/backends/base/`:**
- Purpose: Abstract backend contracts (`BaseSchemaBackend`, `BaseCheckBackend`, `BaseParserBackend`) and framework-agnostic built-in check/hypothesis implementations
- Generated: No
- Committed: Yes

**`tests/mypy/`:**
- Purpose: Static analysis tests; contains Python source files used as mypy inputs and separate mypy config files per scenario
- Generated: No
- Committed: Yes

**`.hypothesis/`:**
- Purpose: Hypothesis library database of previously found failing examples; speeds up re-runs
- Generated: Yes (by Hypothesis during test runs)
- Committed: No (typically gitignored)

**`.planning/`:**
- Purpose: GSD planning documents for AI-assisted development workflow
- Generated: Yes (by GSD tooling)
- Committed: Yes

---

*Structure analysis: 2026-03-08*
