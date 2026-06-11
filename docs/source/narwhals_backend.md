(narwhals-backend)=

# Narwhals

As of *0.32.0*, Pandera ships an optional
[Narwhals](https://narwhals-dev.github.io/narwhals/)-based validation
backend that powers the {ref}`Polars <polars>`, {ref}`Ibis <ibis>`, and
{ref}`PySpark SQL <native-pyspark>` integrations behind a single unified code
path. The Narwhals backend is **opt-in**: by default Pandera continues to use
the native Polars, Ibis, and PySpark backends. The public API
(`import pandera.polars as pa`, `import pandera.ibis as pa`,
`import pandera.pyspark as pa`) is unchanged regardless of which backend is
active.

## Enabling the Narwhals backend

To switch the Polars, Ibis, and PySpark SQL integrations onto the
Narwhals-powered backend, install the `narwhals` extra and set the
`PANDERA_USE_NARWHALS_BACKEND` environment variable to `True` before
importing `pandera.polars`, `pandera.ibis`, or `pandera.pyspark`:

```bash
pip install 'pandera[narwhals]'
export PANDERA_USE_NARWHALS_BACKEND=True
```

You can also enable it programmatically by setting
{py:attr}`pandera.config.CONFIG.use_narwhals_backend` to `True` before any
`pandera.polars` / `pandera.ibis` / `pandera.pyspark` schema is constructed:

```python
import pandera.config

pandera.config.CONFIG.use_narwhals_backend = True

import pandera.polars as pa  # narwhals backend now registered
```

The backend choice is locked in the first time a Polars or Ibis schema is
created (the registration step is `lru_cache`-d). To switch backends in the
same process, clear the cache and re-register:

```python
from pandera.backends.polars.register import register_polars_backends
from pandera.backends.ibis.register import register_ibis_backends
from pandera.backends.pyspark.register import register_pyspark_backends

register_polars_backends.cache_clear()
register_ibis_backends.cache_clear()
register_pyspark_backends.cache_clear()
```

If `PANDERA_USE_NARWHALS_BACKEND=True` but `narwhals` is not installed,
schema construction raises an `ImportError` directing you to install
`pandera[narwhals]`.

## What it is

Narwhals is a lightweight compatibility layer that provides a subset of the
Polars expression API on top of multiple underlying DataFrame libraries
(Polars, pandas, PyArrow, Modin, cuDF, Dask, Ibis, DuckDB, PySpark, etc.).
Pandera uses it to express validation logic &mdash; column selection, type
coercion, check evaluation, failure-case collection &mdash; once, and have
it executed natively by each supported engine.

## What it changes for you

* **Unified checks across Polars, Ibis, and PySpark SQL.** Built-in checks
  (`isin`, `in_range`, `str_matches`, etc.) are implemented as Narwhals
  expressions and run unchanged on Polars LazyFrames, Ibis tables, and
  PySpark SQL DataFrames when the Narwhals backend is enabled. PySpark SQL
  is a SQL-lazy backend: element-wise checks are not supported, and row
  sampling (`sample=` / `tail=` parameters) is not supported.
* **Lazy validation stays lazy.** For Polars LazyFrames, Ibis tables, and
  PySpark SQL DataFrames, Pandera threads validation through the native lazy
  API: no full-frame `.collect()` / `.execute()` is triggered during
  validation. Only the bounded `failure_cases` frame is materialized, and
  only on error.
* **Custom checks become portable.** A check written against
  `pandera.polars` typically works against `pandera.ibis` (and vice versa)
  as long as it uses Narwhals expressions. The `native` parameter on `Check`
  controls which frame type the check function receives: `native=True`
  (the **default**) passes the native backend frame (e.g. `pl.DataFrame`,
  `ibis.Table`) so the check is backend-specific; setting `native=False`
  passes a Narwhals-wrapped frame so the check can run unchanged across all
  supported backends using only the Narwhals expression API.

(narwhals-pyspark-differences)=

## PySpark SQL: differences from the native backend

Because the Narwhals backend for PySpark shares its check implementations with
the Polars and Ibis backends, several behaviours differ from the native PySpark
backend:

- **SQL-lazy execution.** No element-wise checks (no `map_batches` on SQL-lazy
  frames), and no row sampling via `sample=` / `tail=` parameters.
- **`coerce=True` is a no-op.** The Narwhals `ColumnBackend` has no coercion
  step. Setting `coerce=True` on a `Field` or `Column` performs no coercion;
  Pandera emits a ``SchemaWarning`` per column to make the subsequent
  ``WRONG_DATATYPE`` error understandable rather than silent. Setting
  ``coerce=True`` at the `Config` level (row-wise `auto_coerce` dtype) is
  handled and does not warn.
  If you rely on `coerce=True` to convert column dtypes, use the native PySpark
  backend (`PANDERA_USE_NARWHALS_BACKEND=False`).
- **Custom checks using `PysparkDataframeColumnObject` are incompatible.**
  Custom checks registered via `@register_check_method` that expect a
  `pyspark_obj: PysparkDataframeColumnObject` argument will not work under the
  Narwhals backend. The Narwhals backend passes a `NarwhalsData(frame, key)`
  named tuple to check functions instead, so the custom check signature and
  body must be rewritten against the Narwhals frame API (or kept on the
  native backend).
- **`failure_cases` rows may be omitted for scalar Polars errors.** Schema-level
  failure cases produced as scalar Polars frames (e.g. from a wrong-dtype check)
  are still reported in the ``errors`` dict but their rows are omitted from the
  aggregated ``failure_cases`` frame. See the
  {ref}`Known gaps <narwhals-known-gaps>` section for details.
- **Unified `SchemaErrors` contract.** Like the Polars and Ibis Narwhals
  backends, the PySpark Narwhals backend raises `pandera.errors.SchemaErrors`
  on validation failure (or `SchemaError` for the first error when
  `lazy=False`). This differs from the native PySpark backend, which attaches
  errors to `dataframe.pandera.errors`. If you depend on the
  `dataframe.pandera.errors` accessor, use the native PySpark backend
  (`PANDERA_USE_NARWHALS_BACKEND=False`).

## Opting out

The Narwhals backend is **off by default**, so no action is needed to
continue using the native Polars and Ibis backends. If you previously
opted in and want to switch back, unset the environment variable (or set
it to `False`):

```bash
unset PANDERA_USE_NARWHALS_BACKEND
# or
export PANDERA_USE_NARWHALS_BACKEND=False
```

The native paths remain fully supported alongside the Narwhals path.

(narwhals-known-gaps)=

## Known gaps

A small number of features are currently not wired through the Narwhals
backend. Follow-up milestones track each of the gaps below:

* Under the PySpark Narwhals backend, schema-level ``failure_cases`` produced as
  scalar Polars frames (e.g. from a wrong-dtype error) are still reported in the
  ``errors`` dict but their rows are omitted from the aggregated ``failure_cases``
  frame. This is because scalar Polars frames cannot be converted to PySpark
  without a live ``SparkSession`` at the error-collection site; this gap is
  tracked for a future release.
* Column-level `coerce=True` is currently a no-op for **all** Narwhals backends
  (Polars, Ibis, PySpark SQL). Pandera emits a one-time ``SchemaWarning`` per
  column so the subsequent ``WRONG_DATATYPE`` error is understandable rather than
  silent. Full column-level coercion support is tracked as a follow-up.
* `coerce` for the Ibis backend (deferred; `Ibis` coerces eagerly today)
* `add_missing_columns` parser and `set_default` for `Column` fields
* `group_by`-based checks beyond element-wise and column-wise expressions
* Element-wise checks for SQL-lazy backends (Ibis and PySpark SQL). As a consequence,
  the shared built-in check suite in ``tests/common/`` does not run for the PySpark
  Narwhals backend (all shared checks are element-wise; running them would produce only
  skips with no useful coverage signal).
* Schema IO (YAML/JSON) for Narwhals-backed schemas
* Hypothesis data-synthesis strategies
* `sample=` / `tail=` row sampling for SQL-lazy backends (Ibis and PySpark SQL)
* `check_unique` (column-level uniqueness) does not produce a per-row boolean
  `check_output`, so `drop_invalid_rows=True` cannot filter rows that fail a
  uniqueness constraint — those rows remain in the output. This gap is tracked
  for a future release.

See the {ref}`Supported DataFrame Libraries <supported-dataframe-libraries>`
page for the user-facing integrations; the Narwhals layer is an
implementation detail that keeps them consistent.
