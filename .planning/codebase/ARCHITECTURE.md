# Architecture

**Analysis Date:** 2026-03-08

## Pattern Overview

**Overall:** Plugin-based multi-backend validation library with a strict API/Backend separation

**Key Characteristics:**
- The `api/` layer defines framework-agnostic schema specifications as pure Python classes; the `backends/` layer implements all execution logic per data framework
- Backends are registered lazily at validation time via a registry pattern keyed on `(schema_class, data_object_type)`, enabling zero-import cost for unused backends
- A shared `api/dataframe/` layer provides a generic intermediate that pandas, polars, and ibis specific APIs all extend, reducing code duplication across backends
- Two user-facing APIs coexist: object-based (`DataFrameSchema`, `Column`, `Check`) and class-based (`DataFrameModel` with typed `Field` annotations)
- Built-in checks and hypotheses are registered as named functions on the `Check` class via a `MetaCheck` metaclass; custom checks extend this via `register_check_method`

## Layers

**Public API Layer:**
- Purpose: Expose user-facing schema definitions; no validation logic lives here
- Location: `pandera/api/`
- Contains: `BaseSchema`, `DataFrameSchema`, `Column`, `Check`, `Parser`, `Hypothesis`, `DataFrameModel`, `Field`
- Depends on: `pandera/dtypes.py`, `pandera/engines/`, `pandera/errors.py`
- Used by: End users, decorators, `pandera/pandas.py`, `pandera/polars.py`, `pandera/ibis.py`, `pandera/pyspark.py`

**Backend Layer:**
- Purpose: Implement all parsing, validation, and error reporting for each data framework
- Location: `pandera/backends/`
- Contains: `DataFrameSchemaBackend`, `ColumnBackend`, `PandasCheckBackend`, `SeriesSchemaBackend`, and equivalents for polars, ibis, pyspark
- Depends on: `pandera/api/`, `pandera/errors.py`, `pandera/config.py`
- Used by: Called dynamically from `BaseSchema.get_backend()` at validation time

**Engines Layer:**
- Purpose: Abstract dtype systems across frameworks into a unified pandera `DataType` hierarchy
- Location: `pandera/engines/`
- Contains: `engine.py` (abstract `Engine` metaclass and `DataType` registry), `pandas_engine.py`, `polars_engine.py`, `ibis_engine.py`, `numpy_engine.py`, `pyarrow_engine.py`, `pyspark_engine.py`, `geopandas_engine.py`
- Depends on: `pandera/dtypes.py`
- Used by: `pandera/api/` (dtype coercion), `pandera/backends/` (type checking)

**Typing Layer:**
- Purpose: Provide type annotations for schema models and Pydantic/mypy integration
- Location: `pandera/typing/`
- Contains: `common.py` (dtype aliases, `DataFrameBase`), `pandas.py`, `polars.py`, `dask.py`, `modin.py`, `fastapi.py`, `geopandas.py`, `ibis.py`, `pyspark.py`, `pyspark_sql.py`
- Depends on: `pandera/dtypes.py`
- Used by: `DataFrameModel` field annotations, `@check_types` decorator

**Framework Entry Modules:**
- Purpose: Convenience namespaces that bundle the full public API for each framework and trigger backend registration
- Location: `pandera/pandas.py`, `pandera/polars.py`, `pandera/ibis.py`, `pandera/pyspark.py`
- Contains: Re-exports of all symbols users need
- Depends on: All of `api/`, `backends/`, `engines/`, `typing/`
- Used by: End users via `import pandera.pandas as pa`

**Accessors Layer:**
- Purpose: Attach `pandera` namespace to native data objects (e.g. `df.pandera.schema`) using framework accessor APIs
- Location: `pandera/accessors/`
- Contains: `pandas_accessor.py`, `polars_accessor.py`, `dask_accessor.py`, `modin_accessor.py`, `pyspark_accessor.py`, `pyspark_sql_accessor.py`
- Depends on: `pandera/api/`
- Used by: Backend `validate()` methods to annotate data objects with their schema

**IO Layer:**
- Purpose: Serialize/deserialize schemas to/from YAML, JSON, and Frictionless formats
- Location: `pandera/io/`, `pandera/io/pandas_io.py`
- Contains: YAML/JSON schema serialization and deserialization for pandas schemas
- Depends on: `pandera/api/pandas/`, optional `pyyaml`, `frictionless`
- Used by: `DataFrameSchema.to_yaml()`, `DataFrameSchema.from_yaml()`

**Schema Inference Layer:**
- Purpose: Infer a schema from an existing DataFrame
- Location: `pandera/schema_inference/`
- Contains: `pandas.py`
- Depends on: `pandera/api/pandas/`, `pandera/schema_statistics/`
- Used by: `pandera.infer_schema()`

**Strategies Layer:**
- Purpose: Data synthesis strategies using the Hypothesis library for property-based testing
- Location: `pandera/strategies/`
- Contains: `base_strategies.py`, `pandas_strategies.py`
- Depends on: `pandera/api/`, `hypothesis`
- Used by: `schema.strategy()`, `schema.example()`, `Check.strategy`

## Data Flow

**Schema Definition (Object API):**

1. User creates `DataFrameSchema(columns={"col": Column(int, Check.gt(0))})`
2. `DataFrameSchema.__init__()` stores column specs — defined in `pandera/api/dataframe/container.py`
3. No backend or engine code is executed at definition time

**Schema Definition (Model API):**

1. User subclasses `DataFrameModel` with `Field` annotations — defined in `pandera/api/dataframe/model.py`
2. `DataFrameModel.to_schema()` translates type hints and `Field()` metadata into a `DataFrameSchema` object
3. `MODEL_CACHE` (module-level dict) caches the resulting schema per `(model_class, thread_id)`

**Validation Flow:**

1. User calls `schema.validate(df)` on `pandera/api/pandas/container.py:DataFrameSchema`
2. `validate()` calls `self.get_backend(df)` which looks up `BACKEND_REGISTRY[(DataFrameSchema, pd.DataFrame)]`
3. If not registered, `register_default_backends()` lazily calls `register_pandas_backends()` from `pandera/backends/pandas/register.py`
4. `DataFrameSchemaBackend.validate()` in `pandera/backends/pandas/container.py` runs:
   - `preprocess()` — copy or coerce in-place
   - `run_parsers()` — apply schema-level parsers
   - `collect_column_info()` — resolve column names (including regex patterns)
   - Core schema checks: column name uniqueness, column presence, value uniqueness
   - `run_schema_component_checks()` — validates each `Column`, `Index`, `MultiIndex`
   - `run_checks()` — runs dataframe-level `Check` objects
5. Errors collected into `ErrorHandler`; raises `SchemaError` (eager) or `SchemaErrors` (lazy)

**Decorator Flow:**

1. User annotates function with `@check_types` or `@check_input`/`@check_output` from `pandera/decorators.py`
2. Decorator inspects argument type hints at call time
3. If argument type is a `DataFrameModel` subclass, its schema is resolved and `validate()` is called
4. Validation errors propagate as `SchemaError`/`SchemaErrors`

**State Management:**
- Schema definitions are stateless Python objects (no mutable shared state)
- `MODEL_CACHE` and `GENERIC_SCHEMA_CACHE` are module-level dicts in `pandera/api/dataframe/model.py`, keyed by `(model_class, thread_id)` for thread safety
- `BACKEND_REGISTRY` is a class variable on `BaseSchema` and `BaseCheck` (defined in `pandera/api/base/schema.py` and `pandera/api/base/checks.py`), populated once per process and never mutated after registration

## Key Abstractions

**BaseSchema (`pandera/api/base/schema.py`):**
- Purpose: Abstract base for all schema types across all frameworks
- Examples: `pandera/api/dataframe/container.py:DataFrameSchema`, `pandera/api/pandas/array.py:SeriesSchema`
- Pattern: Template method — subclasses implement `validate()` by delegating to `get_backend()`

**DataFrameSchema (`pandera/api/dataframe/container.py`):**
- Purpose: Generic, framework-agnostic dataframe schema container parameterized by `TDataObject`
- Examples: `pandera/api/pandas/container.py:DataFrameSchema[pd.DataFrame]`, `pandera/api/polars/container.py:DataFrameSchema`
- Pattern: Generic class with type parameter; framework-specific subclasses add dtype-setting logic

**DataFrameModel (`pandera/api/dataframe/model.py`):**
- Purpose: Declarative, class-based schema definition using Python type annotations
- Examples: `pandera/api/pandas/model.py:DataFrameModel`, `pandera/api/polars/model.py:DataFrameModel`
- Pattern: Metaclass-driven (`MetaModel`); `to_schema()` performs reflection to build a `DataFrameSchema`

**BaseSchemaBackend (`pandera/backends/base/__init__.py`):**
- Purpose: Abstract contract for all validation execution logic
- Examples: `pandera/backends/pandas/container.py:DataFrameSchemaBackend`, `pandera/backends/polars/container.py:DataFrameSchemaBackend`
- Pattern: Strategy pattern — registered per `(schema_type, data_type)` pair; invoked by `schema.get_backend()`

**DataType (`pandera/dtypes.py`):**
- Purpose: Framework-agnostic data type base class with `coerce()`, `check()`, and `try_coerce()` methods
- Examples: Engine-specific subclasses in `pandera/engines/pandas_engine.py` (`INT64`, `STRING`, `DATETIME`, etc.)
- Pattern: Abstract class; engine-specific subclasses registered via `Engine` metaclass dispatch

**Engine (`pandera/engines/engine.py`):**
- Purpose: Registry and dispatch system mapping native dtype representations to pandera `DataType` instances
- Examples: `pandas_engine.Engine`, `polars_engine.Engine`, `ibis_engine.Engine`
- Pattern: Metaclass-generated singleton with `dtype()` factory method using `functools.singledispatch`

**Check (`pandera/api/checks.py`):**
- Purpose: Wraps a validation function; dispatches execution to the appropriate `BaseCheckBackend`
- Pattern: `MetaCheck` metaclass provides attribute access to `CHECK_FUNCTION_REGISTRY`; built-in checks registered via `@register_builtin_check` in `pandera/api/extensions.py`

## Entry Points

**`pandera/pandas.py`:**
- Location: `pandera/pandas.py`
- Triggers: `import pandera.pandas as pa`
- Responsibilities: Triggers `register_pandas_backends()`, re-exports full pandas API including `DataFrameSchema`, `Column`, `Check`, `DataFrameModel`, decorators, dtypes

**`pandera/polars.py`:**
- Location: `pandera/polars.py`
- Triggers: `import pandera.polars as pa`
- Responsibilities: Calls `register_polars_backends()` immediately on import, re-exports polars API

**`pandera/ibis.py`:**
- Location: `pandera/ibis.py`
- Triggers: `import pandera.ibis as pa`
- Responsibilities: Re-exports ibis API

**`pandera/pyspark.py`:**
- Location: `pandera/pyspark.py`
- Triggers: `import pandera.pyspark as pa`
- Responsibilities: Re-exports pyspark API

**`pandera/__init__.py`:**
- Location: `pandera/__init__.py`
- Triggers: `import pandera`
- Responsibilities: Conditionally imports pandas API if pandas/numpy are available; always exports `dtypes`, `typing`, `Check`, `Field`

**`pandera/decorators.py`:**
- Location: `pandera/decorators.py`
- Triggers: `@check_types`, `@check_input`, `@check_output`, `@check_io`
- Responsibilities: Pipeline integration — wraps functions to validate inputs/outputs at call time using type hints to locate the schema

## Error Handling

**Strategy:** Dual-mode — eager (raises `SchemaError` immediately on first failure) or lazy (collects all failures, raises `SchemaErrors`)

**Patterns:**
- `ErrorHandler` in `pandera/api/base/error_handler.py` accumulates `SchemaError` instances during lazy validation; categorizes them as `DATA`, `SCHEMA`, or `DTYPE_COERCION`
- `SchemaError` (single failure) and `SchemaErrors` (multiple failures from lazy mode) defined in `pandera/errors.py`
- `ReducedPickleExceptionBase` in `pandera/errors.py` ensures exceptions survive multiprocessing serialization by converting non-picklable attributes to strings
- `BackendNotFoundError` raised when no backend is registered for a `(schema_class, data_type)` pair
- `ParserError` raised when dtype coercion fails

## Cross-Cutting Concerns

**Logging:** Not used — errors surface exclusively via exceptions
**Validation Depth:** Schema-level (`ValidationScope.SCHEMA`) vs data-level (`ValidationScope.DATA`) controlled via `PanderaConfig.validation_depth` or `PANDERA_VALIDATION_DEPTH` env var; config lives in `pandera/config.py`
**Authentication:** Not applicable
**Pydantic Integration:** `DataFrameModel` supports both Pydantic v1 and v2; gated by `PYDANTIC_V2` flag in `pandera/engines/__init__.py`; `@check_types` uses `pydantic.validate_arguments` for argument coercion
**Thread Safety:** `MODEL_CACHE` uses `(model_class, thread_id)` as cache key in `pandera/api/dataframe/model.py`; backend registry is class-level and write-once

---

*Architecture analysis: 2026-03-08*
