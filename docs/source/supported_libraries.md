(supported-dataframe-libraries)=

# Supported DataFrame Libraries

Pandera started out as a pandas-specific dataframe validation library, and
moving forward its core functionality will continue to support pandas. However,
pandera's adoption has resulted in the realization that it can be a much more
powerful tool by supporting other dataframe-like formats.

(dataframe-libraries)=

## DataFrame Library Support

Pandera supports validation of the following DataFrame libraries:

:::{list-table}
:widths: 25 75

* - {ref}`Pandas <dataframeschemas>`
  - Validate pandas dataframes. This is the original datafra me library supported
    by pandera.
* - {ref}`Polars <polars>`
  - Validate Polars dataframes. Polars is a blazingly fast dataframe library.
* - {ref}`Ibis <ibis>`
  - Validate Ibis tables. Ibis is the portable Python dataframe library.
* - {ref}`Pyspark SQL <native-pyspark>`
  - A data processing library for large-scale data.
:::

:::{note}
*new in 0.26.0* &mdash; Pandera ships an optional
[Narwhals](https://narwhals-dev.github.io/narwhals/)-powered backend that
unifies the Polars and Ibis validation paths behind a single implementation.
It is **opt-in**: set the `PANDERA_USE_NARWHALS_BACKEND=True` environment
variable (or `pandera.config.CONFIG.use_narwhals_backend = True`) and install
the `narwhals` extra. The user-facing API
(`import pandera.polars as pa` / `import pandera.ibis as pa`) is unchanged
regardless of which backend is active. See the
{ref}`Narwhals-powered backends <narwhals-backends>` section below for
details.
:::

```{toctree}
:hidden: true
:maxdepth: 1

Polars <polars>
Ibis <ibis>
Pyspark SQL <pyspark_sql>
```

## Validating Pandas-like DataFrames

Pandera provides multiple ways of scaling up data validation of pandas-like
dataframes that don't fit into memory. Fortunately, pandera doesn't have to
re-invent the wheel. Standing on shoulders of giants, it integrates with the
existing ecosystem of libraries that allow you to perform validations on
out-of-memory pandas-like dataframes. The following libraries are supported
via pandera's pandas validation backend:

:::{list-table}
:widths: 25 75

* - {ref}`Dask <scaling-dask>`
  - Apply pandera schemas to Dask dataframe partitions.
* - {ref}`Modin <scaling-modin>`
  - A pandas drop-in replacement, distributed using a Ray or Dask backend.
* - {ref}`Pyspark Pandas <scaling-pyspark>`
  - The pandas-like interface exposed by pyspark.
:::

```{toctree}
:hidden: true
:maxdepth: 1

Dask <dask>
Modin <modin>
Pyspark Pandas <pyspark>
```

## Domain-specific Data Validation

The pandas ecosystem provides support for
[domain-specific data manipulation](https://pandas.pydata.org/community/ecosystem.html),
and by extension pandera can provide access to data types, methods, and data
container types specific to these libraries.

:::{list-table}
:widths: 25 75

* - {ref}`GeoPandas <supported-lib-geopandas>`
  - An extension of pandas that adds geospatial data processing capabilities.
:::

```{toctree}
:hidden: true
:maxdepth: 1

GeoPandas <geopandas>
```

## Alternative Acceleration Frameworks

Pandera works with other dataframe-agnostic libraries that allow for distributed
dataframe validation:

:::{list-table}
:widths: 25 75

* - {ref}`Fugue <scaling-fugue>`
  - Apply pandera schemas to distributed dataframe partitions with Fugue.
:::

```{toctree}
:hidden: true
:maxdepth: 1

Fugue <fugue>
```

(narwhals-backends)=

## Narwhals-powered backends

As of *0.26.0*, Pandera ships an optional
[Narwhals](https://narwhals-dev.github.io/narwhals/)-based validation
backend that powers both the {ref}`Polars <polars>` and {ref}`Ibis <ibis>`
integrations behind a single unified code path. The Narwhals backend is
**opt-in**: by default Pandera continues to use the native Polars and Ibis
backends. The public API (`import pandera.polars as pa`,
`import pandera.ibis as pa`) is unchanged regardless of which backend is
active.

### Enabling the Narwhals backend

To switch the Polars and Ibis integrations onto the Narwhals-powered
backend, install the `narwhals` extra and set the
`PANDERA_USE_NARWHALS_BACKEND` environment variable to `True` before
importing `pandera.polars` or `pandera.ibis`:

```bash
pip install 'pandera[narwhals]'
export PANDERA_USE_NARWHALS_BACKEND=True
```

You can also enable it programmatically by setting
{py:attr}`pandera.config.CONFIG.use_narwhals_backend` to `True` before any
`pandera.polars` / `pandera.ibis` schema is constructed:

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

register_polars_backends.cache_clear()
register_ibis_backends.cache_clear()
```

If `PANDERA_USE_NARWHALS_BACKEND=True` but `narwhals` is not installed,
schema construction raises an `ImportError` directing you to install
`pandera[narwhals]`.

### What it is

Narwhals is a lightweight compatibility layer that provides a subset of the
Polars expression API on top of multiple underlying DataFrame libraries
(Polars, pandas, PyArrow, Modin, cuDF, Dask, Ibis, DuckDB, PySpark, etc.).
Pandera uses it to express validation logic &mdash; column selection, type
coercion, check evaluation, failure-case collection &mdash; once, and have
it executed natively by each supported engine.

### What it changes for you

* **Unified checks across Polars and Ibis.** Built-in checks
  (`isin`, `in_range`, `str_matches`, etc.) are implemented as Narwhals
  expressions and run unchanged on both Polars LazyFrames and Ibis tables
  when the Narwhals backend is enabled.
* **Lazy validation stays lazy.** For Polars LazyFrames and Ibis tables,
  Pandera threads validation through the native lazy API: no full-frame
  `.collect()` / `.execute()` is triggered during validation. Only the
  bounded `failure_cases` frame is materialized, and only on error.
* **Custom checks become portable.** A check written against
  `pandera.polars` typically works against `pandera.ibis` (and vice versa)
  as long as it uses Narwhals expressions. Backend-native checks
  (pure `polars.Expr` or pure `ibis` expressions) are still supported via
  the `native=True` flag on `Check`.

### Opting out

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

### Known gaps

A small number of features are currently not wired through the Narwhals
backend. Follow-up milestones track each of the gaps below:

* `coerce` for the Ibis backend (deferred; `Ibis` coerces eagerly today)
* `add_missing_columns` parser and `set_default` for `Column` fields
* `group_by`-based checks beyond element-wise and column-wise expressions
* Schema IO (YAML/JSON) for Narwhals-backed schemas
* Hypothesis data-synthesis strategies
* `sample=` subsampling (only `head=` / `tail=` are supported today)

See the {ref}`Supported DataFrame Libraries <supported-dataframe-libraries>`
section above for the user-facing integrations; the Narwhals layer is an
implementation detail that keeps them consistent.

:::{note}
Don't see a library that you want supported? Check out the
[github issues](https://github.com/pandera-dev/pandera/issues) to see if
that library is in the roadmap. If it isn't, open up a
[new issue](https://github.com/pandera-dev/pandera/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=)
to add support for it!
:::
