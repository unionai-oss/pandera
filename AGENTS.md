# AGENTS.md

## Project Overview

Pandera is a data validation library for Python that provides a flexible,
expressive API for validating dataframes and series across multiple backends.
It supports pandas, polars, pyspark, ibis, dask, modin, and geopandas.

- **License:** MIT
- **Python:** >= 3.10 (tested on 3.10-3.14)
- **Docs:** https://pandera.readthedocs.io
- **Repo:** https://github.com/pandera-dev/pandera

## Repository Structure

```
pandera/                  # Main package
├── api/                  # Public validation API (backend-specific schemas/models)
│   ├── base/             # Abstract base classes (BaseSchema, BaseCheck, etc.)
│   ├── pandas/           # Pandas DataFrameSchema, Column, Index, DataFrameModel
│   ├── geopandas/        # GeoDataFrameSchema, GeoDataFrameModel (pandas-backed)
│   ├── polars/           # Polars schema and model implementations
│   ├── pyspark/          # PySpark schema and model implementations
│   ├── ibis/             # Ibis schema and model implementations
│   ├── dataframe/        # Shared dataframe components
│   ├── checks.py         # Check class (ge, le, isin, str_matches, etc.)
│   └── hypotheses.py     # Hypothesis-based statistical testing
├── backends/             # Validation logic implementations per backend
│   ├── base/             # Abstract backend + builtin check/hypothesis impls
│   ├── pandas/           # Pandas backend (checks, components, container, etc.)
│   ├── polars/           # Polars backend
│   ├── pyspark/          # PySpark backend
│   └── ibis/             # Ibis backend
├── engines/              # Type system and dtype handling per backend
│   ├── engine.py         # Engine metaclass and type registry
│   ├── pandas_engine.py  # Pandas dtype definitions
│   ├── numpy_engine.py   # NumPy dtype definitions
│   ├── polars_engine.py  # Polars dtype definitions
│   ├── pyspark_engine.py # PySpark dtype definitions
│   ├── ibis_engine.py    # Ibis dtype definitions
│   ├── pyarrow_engine.py # PyArrow dtype definitions
│   └── geopandas_engine.py
├── typing/               # Type stubs for mypy integration
├── strategies/           # Hypothesis strategies: import ``pandera.strategies.pandas_strategies`` etc. (``strategies`` package does not import backends)
├── schema_inference/     # Infer schemas from data
├── schema_statistics/    # Statistical validation helpers
├── io/                   # Serialization: ``pandera.io.pandas_io``, ``polars_io``, ``pyspark_sql_io``, ``ibis_io``, ``xarray_io`` (``pandera.io`` package has no imports)
├── accessors/            # Pandas/PySpark accessor extensions (.pandera)
├── config.py             # PanderaConfig, ValidationDepth, ValidationScope
├── decorators.py         # @check_input, @check_output, @check_io, @check_types
├── dtypes.py             # Abstract data type definitions
├── errors.py             # SchemaError, SchemaInitError, ParserError, etc.
├── extensions.py         # Custom check/parser extension mechanism
├── pandas.py             # Pandas entry point: `import pandera.pandas as pa`
├── geopandas.py          # GeoPandas entry (`pg`): pandas API + GeoDataFrameSchema/Model
├── polars.py             # Polars entry point: `import pandera.polars as pa`
├── pyspark.py            # PySpark entry point: `import pandera.pyspark as pa`
└── ibis.py               # Ibis entry point: `import pandera.ibis as pa`

tests/                    # Test suite (mirrors backend structure)
├── base/                 # Core tests (no backend-specific deps)
├── pandas/               # Pandas backend tests (~38 files)
├── polars/               # Polars backend tests
├── pyspark/              # PySpark backend tests
├── ibis/                 # Ibis backend tests
├── dask/                 # Dask integration tests
├── modin/                # Modin integration tests
├── geopandas/            # GeoPandas tests
├── strategies/           # Hypothesis strategy tests
├── hypotheses/           # Statistical hypothesis tests
├── io/                   # Serialization tests
├── fastapi/              # FastAPI integration tests
├── mypy/                 # MyPy type-checking tests
└── conftest.py           # Shared pytest fixtures

docs/source/              # Sphinx documentation (MyST markdown + RST)
```

## Architecture

### Layered Design

1. **API layer** (`pandera/api/`): Defines schemas, models, checks, and
   parsers. Each backend has its own subpackage inheriting from `base/`.
2. **Backend layer** (`pandera/backends/`): Implements actual validation logic.
   Backends are registered via `BaseSchema.BACKEND_REGISTRY` and discovered at
   runtime.
3. **Engine layer** (`pandera/engines/`): Manages dtype registration and
   coercion. Uses an `Engine` metaclass pattern for pluggable type systems.

### Key Patterns

- **Backend registry:** Backends register themselves at import time. The schema
  objects delegate validation to the registered backend.
- **Engine metaclass:** Each engine (pandas, polars, etc.) uses a metaclass that
  maintains a dtype registry. Types are registered with `@Engine.register_dtype`.
- **DataFrameModel:** Pydantic-style class-based schema definitions using type
  annotations and `Field()` descriptors.
- **Lazy validation:** Pass `lazy=True` to `schema.validate()` to collect all
  errors instead of failing on the first one.

### Entry Points

Users import backend-specific modules:
```python
import pandera.pandas as pa    # Pandas
import pandera.polars as pa    # Polars
import pandera.pyspark as pa   # PySpark
import pandera.ibis as pa      # Ibis
```

The top-level `import pandera` falls back to the pandas API for backward
compatibility.

## Development Setup

```bash
# Install uv and sync all extras
make setup

# macOS (uses polars-lts-cpu)
make setup-macos
```

This runs `uv sync --all-extras` which installs all optional dependencies and
dev/testing groups.

## Running Tests

Tests are organized by backend. Each backend's tests live in `tests/<backend>/`.

```bash
# Run core + pandas tests
pytest tests/core tests/pandas

# Run a specific backend's tests
pytest tests/polars/
pytest tests/pyspark/
pytest tests/ibis/

# Run all tests with coverage
pytest --cov=pandera --cov-report=term-missing tests/

# Run via nox (parameterized across Python/pandas/pydantic/polars versions)
nox -db uv -s tests

# Run a specific nox session
nox -db uv -s "tests(extra='polars', pandas=None, pydantic=None, polars='1.33.1')"
```

The nox `tests` session maps extras to test directories: `extra=None` runs
`tests/base/`, `extra='pandas'` runs `tests/pandas/`, etc.

### Test Matrix (CI)

- Python: 3.10, 3.11, 3.12, 3.13, 3.14
- Pandas: 2.1.1, 2.3.3
- Pydantic: 1.10.11, 2.12.3
- Polars: 0.20.0, 1.33.1

## Code Quality

### Linting and Formatting

- **Ruff:** Linting (`I`, `UP` rules) and formatting. Line length: 79.
- **isort:** Import sorting (line length 79).
- **mypy:** Static type checking (v1.10.0). Config in `mypy.ini`.
- **pyupgrade:** Python 3.9+ syntax upgrades.
- **flynt:** f-string conversion.
- **codespell:** Spell checking.

`prek` hooks enforce all of the above. Run manually:
```bash
prek run --all-files
```

### Style Guidelines

- Line length: **79 characters**
- Target Python version: **3.10+**
- Use f-strings (enforced by flynt)
- Use modern Python syntax: `X | Y` unions, etc. (enforced by pyupgrade/ruff)
- Ruff ignores `UP007` (X | Y in annotations — kept for runtime typing compat)

## Dependencies

### Core (always required)
- `packaging`, `pydantic`, `typeguard`, `typing_extensions`, `typing_inspect`

### Optional (by backend/feature)
| Extra        | Key packages                          |
|--------------|---------------------------------------|
| `pandas`     | numpy, pandas >= 2.1.1                |
| `polars`     | polars >= 0.20.0                      |
| `pyspark`    | pyspark[connect] >= 3.2.0             |
| `ibis`       | ibis-framework >= 9.0.0               |
| `dask`       | dask[dataframe], distributed          |
| `modin`      | modin, ray, dask                      |
| `geopandas`  | geopandas, shapely                    |
| `strategies` | hypothesis >= 6.92.7                  |
| `hypotheses` | scipy                                 |
| `io`         | pyyaml                                |
| `fastapi`    | fastapi                               |
| `mypy`       | pandas-stubs, scipy-stubs             |
| `all`        | Everything above                      |

## Building Documentation

```bash
# Full build with doctests (cleans first)
make docs

# Quick build (no clean, no -W flag)
make quick-docs

# Via nox
nox -db uv -s docs
```

Documentation uses Sphinx with MyST (markdown) and RST. Source is in
`docs/source/`. API reference is auto-generated into
`docs/source/reference/generated/`.

## Error Hierarchy

- `SchemaError` — Raised when data fails validation
- `SchemaErrors` — Container for multiple errors (lazy validation)
- `SchemaInitError` — Raised when a schema is defined incorrectly
- `ParserError` — Raised when data parsing/coercion fails
- `BackendNotFoundError` — Raised when a required backend is not installed

## Adding a New Check

1. For builtin checks, add to `pandera/backends/base/builtin_checks.py` and
   register in each backend's `checks.py`.
2. For custom checks via extensions, use `pandera.extensions.register_check_method`.
3. Add corresponding tests in the relevant `tests/<backend>/` directory.

## Adding a New Backend

1. Create `pandera/api/<backend>/` with schema, model, and component classes
   inheriting from `pandera/api/base/`.
2. Create `pandera/backends/<backend>/` with validation implementations
   inheriting from `pandera/backends/base/`.
3. Create `pandera/engines/<backend>_engine.py` with dtype registrations.
4. Create a top-level entry point `pandera/<backend>.py`.
5. Add tests in `tests/<backend>/`.
6. Register the backend in the appropriate `register.py` file.

## CI/CD

- **GitHub Actions:** `.github/workflows/ci-tests.yml` runs linting, unit tests
  across all backends/platforms, coverage, and mypy.
- **Publishing:** `.github/workflows/publish.yml` handles PyPI releases.
- **Versioning:** Automatic via `setuptools_scm` from git tags. Version file at
  `pandera/_version.py`.
