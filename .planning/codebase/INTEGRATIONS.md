# External Integrations

**Analysis Date:** 2026-03-08

## APIs & External Services

Pandera is a pure Python library. It has no runtime calls to external APIs or cloud services. All integrations below are optional dataframe library backends and developer tooling.

**Dataframe Library Integrations (optional, import-time):**
- `pandas` — Primary dataframe backend. Engines: `pandera/engines/pandas_engine.py`. Backend: `pandera/backends/pandas/`.
- `polars` — Polars dataframe backend. Engine: `pandera/engines/polars_engine.py`. Backend: `pandera/backends/polars/`.
- `pyspark` (via `pyspark[connect]`) — Apache Spark backend. Engine: `pandera/engines/pyspark_engine.py`. Backend: `pandera/backends/pyspark/`.
- `modin` — Drop-in pandas replacement using Ray or Dask. Shares the pandas backend (`pandera/backends/pandas/`). Engine detection via `pandera/engines/pandas_engine.py`.
- `dask[dataframe]` + `distributed` — Dask dataframe backend. Shares the pandas backend (`pandera/backends/pandas/`).
- `ibis-framework >= 9.0.0` — Ibis query framework; engine `pandera/engines/ibis_engine.py`; backend `pandera/backends/ibis/`. Ibis itself supports 20+ backends (DuckDB, BigQuery, Snowflake, etc.) transparently.
- `geopandas` + `shapely` — Geospatial dataframe extension. Engine: `pandera/engines/geopandas_engine.py`. Also uses `pyproj`.

## Data Storage

**Databases:**
- None at runtime. No persistent storage.
- DuckDB is used as the ibis backend in tests and dev environments (`duckdb` in `[dependency-groups].dev`). Connection is in-memory, managed by ibis.

**File Storage:**
- Local filesystem only. Schema I/O reads/writes YAML files via `pandera/io/pandas_io.py` using `pyyaml`.
- Frictionless Data schema files (JSON/YAML) supported via the `frictionless` optional dependency.

**Caching:**
- No external cache. In-process dataframe caching controlled by `PANDERA_CACHE_DATAFRAME` env var (`pandera/config.py`).

## Authentication & Identity

**Auth Provider:**
- None. Pandera is a library with no user accounts, sessions, or auth flows.

## Monitoring & Observability

**Error Tracking:**
- None at runtime.

**Coverage:**
- Codecov — CI uploads coverage XML reports via `codecov/codecov-action@v4` in `.github/workflows/ci-tests.yml`.
- Token: `PANDERA_CODECOV_TOKEN` (GitHub Actions secret).

**Logs:**
- Python standard `logging` module used internally. Log level configured via pytest `log_cli_level = 20` (INFO) in `pyproject.toml`.

## CI/CD & Deployment

**Hosting:**
- PyPI — Package published via `.github/workflows/publish.yml` using `pypa/gh-action-pypi-publish@release/v1` with OIDC trusted publishing (no stored API token).
- ReadTheDocs — Documentation at https://pandera.readthedocs.io (configured via the `docs` nox session and Sphinx build).

**CI Pipeline:**
- GitHub Actions — Defined in `.github/workflows/ci-tests.yml`.
  - Lint job: ruff (lint + format check) via `astral-sh/ruff-action@v3`, mypy via pre-commit.
  - Test jobs: unit-tests-base, unit-tests-pandas, unit-tests-supplemental-extras, unit-tests-dataframe-extras.
  - Matrix: Python 3.10–3.14, pandas 2.1.1/2.3.3, pydantic 1.10.11/2.12.3, polars 0.20.0/1.33.1.
  - Runners: ubuntu-latest, windows-latest, macos-latest (for base tests).
- `nox` orchestrates all sessions locally and on CI (`nox -db uv`).
- Benchmarking: `asv` (Airspeed Velocity) configured in `asv_bench/` directory.

## Environment Configuration

**Required env vars (runtime):**
- None required. All env vars are optional with safe defaults (`pandera/config.py`).

**Optional env vars:**
- `PANDERA_VALIDATION_ENABLED` — Default `True`
- `PANDERA_VALIDATION_DEPTH` — Default `SCHEMA_AND_DATA`
- `PANDERA_CACHE_DATAFRAME` — Default `False`
- `PANDERA_KEEP_CACHED_DATAFRAME` — Default `False`
- `SPARK_LOCAL_IP` — Auto-set to `127.0.0.1` for PySpark (`pandera/external_config.py`)
- `PYARROW_IGNORE_TIMEZONE` — Auto-set to `1` for PySpark (`pandera/external_config.py`)

**Secrets (CI only):**
- `PANDERA_CODECOV_TOKEN` — Coverage upload to Codecov.
- PyPI publishing uses OIDC trusted publishing (no token stored).

## Webhooks & Callbacks

**Incoming:**
- None. Pandera is a library, not a service.

**Outgoing:**
- None at runtime.

## FastAPI Integration

**Integration Type:** Optional library extension, not a deployed service.
- File: `pandera/typing/fastapi.py`
- Provides `pandera.typing.fastapi.UploadFile` — a subclass of `fastapi.UploadFile` that validates uploaded CSV/dataframe files against a `DataFrameModel` schema automatically.
- Requires `fastapi` and `starlette` to be installed (`pandera[fastapi]`).
- Pydantic v1 and v2 validation paths both implemented.

## Schema Format Integrations

**YAML Schema I/O:**
- Reads/writes pandera schemas to YAML format.
- Implementation: `pandera/io/pandas_io.py`
- Requires: `pyyaml >= 5.1` (`pandera[io]`)

**Frictionless Data:**
- Parses Frictionless Data descriptors into pandera schemas.
- Implementation: `pandera/io/pandas_io.py` (conditional import of `frictionless`)
- Requires: `frictionless <= 4.40.8` (`pandera[frictionless]`)

**PyArrow:**
- PyArrow dtype support in the pandas engine.
- Engine: `pandera/engines/pyarrow_engine.py`
- Requires: `pyarrow >= 13`

---

*Integration audit: 2026-03-08*
