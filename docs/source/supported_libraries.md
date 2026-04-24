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
*new in 0.26.0* &mdash; Pandera's Polars and Ibis backends are powered by
[Narwhals](https://narwhals-dev.github.io/narwhals/), a lightweight
compatibility layer between DataFrame libraries. The Narwhals-based backends
are enabled automatically when `narwhals` is installed; the user-facing API
(`import pandera.polars as pa` / `import pandera.ibis as pa`) is unchanged.
See the {ref}`Narwhals-powered backends <narwhals-backends>` section below
for details.
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

As of *0.26.0*, Pandera ships an internal
[Narwhals](https://narwhals-dev.github.io/narwhals/)-based validation
backend that powers both the {ref}`Polars <polars>` and {ref}`Ibis <ibis>`
integrations behind a single unified code path. The backend is enabled
automatically whenever `narwhals` is importable &mdash; you do not need to
install any additional extras, and the public API (`import pandera.polars
as pa`, `import pandera.ibis as pa`) is unchanged.

### What it is

Narwhals is a lightweight compatibility layer that provides a subset of the
Polars expression API on top of multiple underlying DataFrame libraries
(Polars, pandas, PyArrow, Modin, cuDF, Dask, Ibis, DuckDB, PySpark, etc.).
Pandera uses it to express validation logic &mdash; column selection, type
coercion, check evaluation, failure-case collection &mdash; once, and have
it executed natively by each supported engine.

### What it changes for you

* **Unified checks across Polars and Ibis.** Built-in checks
  (`isin`, `in_range`, `str_matches`, etc.) are now implemented as Narwhals
  expressions and run unchanged on both Polars LazyFrames and Ibis tables.
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

If you need the previous non-Narwhals Polars backend for compatibility
reasons, uninstall `narwhals`:

```bash
pip uninstall narwhals
```

Pandera will then fall back to the legacy Polars backend. The legacy path
remains available but is in maintenance mode; new backend work targets the
Narwhals path.

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
