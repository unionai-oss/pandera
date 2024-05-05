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
  - Validate pandas dataframes. This is the original dataframe library supported
    by pandera.
* - {ref}`Polars <polars>`
  - Validate Polars dataframes, the blazingly fast dataframe library.
* - {ref}`Pyspark SQL <native-pyspark>`
  - A data processing library for large-scale data.
:::

```{toctree}
:hidden: true
:maxdepth: 1

Polars <polars>
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

:::{note}
Don't see a library that you want supported? Check out the
[github issues](https://github.com/pandera-dev/pandera/issues) to see if
that library is in the roadmap. If it isn't, open up a
[new issue](https://github.com/pandera-dev/pandera/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=)
to add support for it!
:::
