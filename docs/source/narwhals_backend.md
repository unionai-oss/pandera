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

The Narwhals backend is **opt-in**. Install the `narwhals` extra alongside the
backend(s) you use:

```bash
pip install 'pandera[narwhals,polars]'   # Polars
pip install 'pandera[narwhals,ibis]'     # Ibis
pip install 'pandera[narwhals,pyspark]'  # PySpark SQL
```

Then enable it using **either** of the following options.

### Environment variable (process start)

Set `PANDERA_USE_NARWHALS_BACKEND` to `True` before starting Python:

```bash
export PANDERA_USE_NARWHALS_BACKEND=True
python your_script.py
```

This value is read when `pandera.config` is first imported.

### Programmatic configuration

Call {func}`~pandera.set_config` at any point — before or after importing
`pandera.polars`, `pandera.ibis`, or `pandera.pyspark`:

```python
import pandera

pandera.set_config(use_narwhals_backend=True)

import pandera.polars as pa
```

See {ref}`Backend registration <narwhals-backend-registration>` for details on
when backends are registered and how runtime toggling works.

### Advanced: manual re-registration

Prefer {func}`~pandera.set_config` to toggle backends within a process. For
low-level control (for example, in tests), clear the registration caches and
call the register functions with the desired flag:

```python
from pandera.backends.polars.register import register_polars_backends

register_polars_backends.cache_clear()
register_polars_backends(use_narwhals_backend=True)
```

The same pattern applies to `register_ibis_backends` and
`register_pyspark_backends`.

If `PANDERA_USE_NARWHALS_BACKEND=True` but `narwhals` is not installed,
schema construction raises an `ImportError` directing you to install
`pandera[narwhals]`.

(narwhals-backend-registration)=

## Backend registration

Pandera chooses between the native and Narwhals validation backends through a
**registration** step that maps each schema class (for example,
{py:class}`~pandera.api.polars.container.DataFrameSchema`) to a concrete
backend implementation for a given frame type (for example, `polars.DataFrame`).

Two behaviours govern how that mapping is established and updated at runtime.

### Lazy registration

Validation backends for Polars, Ibis, and PySpark SQL are registered
**lazily** — not when you import a pandera backend module, but the first time
a schema needs a backend. Concretely, registration runs when you:

- construct a {py:class}`~pandera.api.polars.container.DataFrameSchema`,
  {py:class}`~pandera.api.ibis.container.DataFrameSchema`, or
  {py:class}`~pandera.api.pyspark.container.DataFrameSchema`, or
- call `validate()` on a column or schema component that triggers backend
  lookup.

Until one of those happens, importing ``pandera.polars``, ``pandera.ibis``, or
``pandera.pyspark`` has no effect on which validation backend is active:

```python
import pandera.polars as pa

# CONFIG.use_narwhals_backend is read here — not at import time above
pa.config.set_config(use_narwhals_backend=True)

schema = pa.DataFrameSchema({"name": pa.Column(str)})  # narwhals backends registered
schema.validate(df)
```

At registration time, pandera reads the current value of
``CONFIG.use_narwhals_backend`` (from the environment variable or a prior
{func}`~pandera.set_config` call) and registers either the native or Narwhals
backend implementations. The register functions are cached with
``@lru_cache``; the ``use_narwhals_backend`` flag is part of the cache key, so
native and Narwhals registrations do not collide.

:::{tip}
Because registration is lazy, you can call {func}`~pandera.set_config` **after**
importing a backend module and **before** constructing your first schema — no
manual cache clearing is required in that case.
:::

### Runtime re-registration

If you change ``use_narwhals_backend`` with {func}`~pandera.set_config` **after**
backends have already been registered, pandera **re-registers** them
automatically:

1. The global ``CONFIG.use_narwhals_backend`` value is updated.
2. Pandera detects which of the Polars / Ibis / PySpark register functions had
   already run.
3. Registration caches are cleared and existing registry entries for those
   backends are removed.
4. Only the backends that were previously registered are registered again, now
   using the new flag value.
5. A ``UserWarning`` is emitted to make the swap visible.

```python
import pandera.polars as pa

schema = pa.DataFrameSchema({"age": pa.Column(int)})
schema.validate(df)  # uses native Polars backend (default)

pa.config.set_config(use_narwhals_backend=True)
# UserWarning: Re-registered pandera backends after use_narwhals_backend changed.

schema.validate(df)  # same schema object, now validated by the Narwhals backend
```

Existing schema objects **continue to work** after re-registration. Schemas
do not store a backend reference at construction time; they look up the
registered backend from the global registry on each ``validate()`` call.

Re-registration applies only to backends that had already been registered in
the current process. If you call ``set_config(use_narwhals_backend=True)``
before constructing any Polars/Ibis/PySpark schema, no re-registration occurs
— the first lazy registration picks up the updated config silently.

:::{note}
Runtime re-registration is triggered by {func}`~pandera.set_config`, which
updates the **global** ``CONFIG``. The ``config_context`` manager overrides
settings for validation behaviour (for example, ``validation_depth``) but does
**not** change which validation backend is registered. Use
{func}`~pandera.set_config` (or the environment variable) to switch between
native and Narwhals backends.
:::

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
  backend (see {ref}`Opting out <narwhals-opting-out>`).
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
  (see {ref}`Opting out <narwhals-opting-out>`).

(narwhals-opting-out)=

## Opting out

The Narwhals backend is **off by default**, so no action is needed to
continue using the native Polars, Ibis, and PySpark backends. If you
previously opted in and want to switch back, unset the environment variable
(or set it to `False`):

```bash
unset PANDERA_USE_NARWHALS_BACKEND
# or
export PANDERA_USE_NARWHALS_BACKEND=False
```

Or call {func}`~pandera.set_config` programmatically:

```python
import pandera

pandera.set_config(use_narwhals_backend=False)
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
