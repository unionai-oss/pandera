# Technology Stack

**Analysis Date:** 2026-03-08

## Languages

**Primary:**
- Python 3.10+ - All library source code, tests, tooling scripts

**Secondary:**
- None (pure Python project)

## Runtime

**Environment:**
- CPython 3.10, 3.11, 3.12, 3.13, 3.14 (all actively tested in CI)
- Minimum required: Python 3.10

**Package Manager:**
- `uv` (primary, used as venv backend in nox and for lock file)
- `pip` (fallback)
- Lockfile: `uv.lock` (present); `pixi.lock` (also present for pixi-based dev environment)
- `pixi` supported as an alternative dev environment manager via `pixi.toml`

## Frameworks

**Core:**
- `pydantic` (>=1.10.11 or >=2.12.3, both supported) - Schema model validation and type coercion; Pydantic v1 and v2 compatibility maintained
- `typeguard` - Runtime type checking for decorator-based validation
- `packaging` (>=20.0) - Version parsing utilities
- `typing_extensions` - Backports of typing features
- `typing_inspect` (>=0.6.0) - Runtime inspection of generic types

**Testing:**
- `pytest` (>=8.x) - Test runner
- `pytest-cov` - Coverage reporting
- `pytest-xdist` - Parallel test execution (used for `strategies` tests)
- `pytest-asyncio` - Async test support (for FastAPI integration tests)
- `hypothesis` (>=6.92.7) - Property-based testing and data synthesis strategies
- `xdoctest` - Doctest runner (run as part of the docs nox session)

**Build/Dev:**
- `nox` - Test session automation, multi-version matrix testing
- `setuptools` + `setuptools_scm` - Package build; version derived from git tags
- `ruff` - Linting (imports, style upgrades) and formatting
- `mypy` (1.10.0 pinned in CI) - Static type checking
- `pre-commit` - Git hooks for mypy
- `black` (optional) - Schema script formatting (used when serializing schema to Python script)
- `sphinx` (<9) + extensions - Documentation build

## Key Dependencies

**Critical (always installed):**
- `packaging >= 20.0` - Version comparison at runtime
- `pydantic` - Core to schema model system
- `typeguard` - Runtime type enforcement
- `typing_extensions` - Used throughout for `Annotated`, `get_type_hints`, etc.
- `typing_inspect >= 0.6.0` - Generic introspection

**Dataframe Backends (optional extras):**
- `pandas >= 2.1.1` + `numpy >= 1.24.4` — `pandera[pandas]`; the primary supported backend; see `pandera/backends/pandas/`
- `polars >= 0.20.0` — `pandera[polars]`; see `pandera/backends/polars/` and `pandera/engines/polars_engine.py`
- `pyspark[connect] >= 3.2.0` — `pandera[pyspark]`; see `pandera/backends/pyspark/` and `pandera/engines/pyspark_engine.py`
- `modin` + `ray`/`dask[dataframe]` — `pandera[modin]`, `pandera[modin-ray]`, `pandera[modin-dask]`
- `dask[dataframe]` + `distributed` — `pandera[dask]`; see `pandera/backends/pandas/` (modin/dask share pandas backend)
- `ibis-framework >= 9.0.0` — `pandera[ibis]`; see `pandera/backends/ibis/` and `pandera/engines/ibis_engine.py`
- `geopandas < 1.1.0` + `shapely` — `pandera[geopandas]`; see `pandera/engines/geopandas_engine.py`

**Data Format/IO:**
- `pyyaml >= 5.1` — `pandera[io]`; YAML schema serialization/deserialization via `pandera/io/pandas_io.py`
- `frictionless <= 4.40.8` — `pandera[frictionless]`; Frictionless Data schema interoperability
- `pyarrow >= 13` — PyArrow type support via `pandera/engines/pyarrow_engine.py`
- `fastapi` — `pandera[fastapi]`; UploadFile integration via `pandera/typing/fastapi.py`

**Hypothesis/Statistics:**
- `hypothesis >= 6.92.7` — `pandera[strategies]`; data synthesis via `pandera/strategies/`
- `scipy` — `pandera[hypotheses]`; statistical hypothesis tests

**Type stubs:**
- `pandas-stubs` — `pandera[mypy]`
- `scipy-stubs` (Python >= 3.10) — `pandera[mypy]`

**DuckDB (dev/test only):**
- `duckdb` — Used as default backend for ibis tests

## Configuration

**Environment:**
- Pandera behavior is configured entirely through environment variables (no `.env` file):
  - `PANDERA_VALIDATION_ENABLED` — Enable/disable validation (`True`/`1`)
  - `PANDERA_VALIDATION_DEPTH` — `SCHEMA_ONLY`, `DATA_ONLY`, or `SCHEMA_AND_DATA`
  - `PANDERA_CACHE_DATAFRAME` — Cache validated dataframe (`True`/`1`)
  - `PANDERA_KEEP_CACHED_DATAFRAME` — Retain cache after validation (`True`/`1`)
  - `SPARK_LOCAL_IP` — Set automatically to `127.0.0.1` if not defined (PySpark)
  - `PYARROW_IGNORE_TIMEZONE` — Set automatically to `1` if not defined (PySpark)
  - `CI_MODIN_ENGINES` — `ray` or `dask` for modin test sessions
  - `CI` — Set to `"true"` on CI runs; changes nox behavior
- Config object: `pandera/config.py` — `PanderaConfig` dataclass, global `CONFIG` instance

**Build:**
- `pyproject.toml` — Project metadata, dependencies, optional extras, tool config (ruff, pytest, pyright, codespell)
- `setup.py` — Minimal stub delegating to setuptools
- `noxfile.py` — Full multi-version test matrix, parametrized by Python, pandas, pydantic, and polars versions
- `mypy.ini` — Mypy configuration; excludes pyspark and docs from type checking
- `.github/workflows/ci-tests.yml` — CI matrix (lint + unit tests across OS, Python, pydantic, pandas, polars versions)
- `.github/workflows/publish.yml` — PyPI release workflow (trusted publishing)

## Platform Requirements

**Development:**
- Python 3.10+
- uv or pixi for environment management
- nox for running test matrix locally

**Production:**
- Pure library: no server, no database, no persistent state
- Published to PyPI; users install with `pip install pandera[<extras>]`
- Documentation hosted on ReadTheDocs at https://pandera.readthedocs.io

---

*Stack analysis: 2026-03-08*
