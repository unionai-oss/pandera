# Supported DataFrame Libraries

Pandera started out as a pandas-specific dataframe validation library, and
moving forward its core functionality will continue to support pandas. However,
pandera's adoption has resulted in the realization that it can be a much more
powerful tool by supporting other dataframe-like formats.

## DataFrame Library Support

Pandera supports validation of the following DataFrame libraries:

| Library | Description |
| :------ | :---------- |
| [Pandas](./dataframe_schemas.html) | Validate pandas dataframes. This is the original dataframe library supported by pandera. |
| [Polars](./polars.html) | Validate Polars dataframes. Polars is a blazingly fast dataframe library. |
| [Ibis](./ibis.html) | Validate Ibis tables. Ibis is the portable Python dataframe library. |
| [PyArrow](./pyarrow.html) | Validate PyArrow tables. Arrow is the in-memory columnar exchange format. |
| [DataFusion](./datafusion.html) | Validate DataFusion dataframes. DataFusion is an Arrow-native query engine. |
| [PySpark SQL](./pyspark_sql.html) | A data processing library for large-scale data. |

## Validating Pandas-like DataFrames

Pandera provides multiple ways of scaling up data validation of pandas-like
dataframes that don't fit into memory. Fortunately, pandera doesn't have to
re-invent the wheel. Standing on shoulders of giants, it integrates with the
existing ecosystem of libraries that allow you to perform validations on
out-of-memory pandas-like dataframes. The following libraries are supported
via pandera's pandas validation backend:

| Library | Description |
| :------ | :---------- |
| [Dask](./dask.html) | Apply pandera schemas to Dask dataframe partitions. |
| [Modin](./modin.html) | A pandas drop-in replacement, distributed using a Ray or Dask backend. |
| [Pyspark Pandas](./pyspark.html) | The pandas-like interface exposed by pyspark. |

## Domain-specific Data Validation

The pandas ecosystem provides support for
[domain-specific data manipulation](https://pandas.pydata.org/community/ecosystem.html),
and by extension pandera can provide access to data types, methods, and data
container types specific to these libraries.

| Library | Description |
| :------ | :---------- |
| [GeoPandas](./geopandas.html) | An extension of pandas that adds geospatial data processing capabilities. |

## Interoperable Dataframe Backends

*new in 0.32.0*

Pandera provides opt-in support for Narwhals-powered interoperable dataframe backends.
This allows Pandera to use a single backend to power the validation of different
dataframe libraries for the library-specific schema definitions, where checks
are expressed in the native library's API.

| Library | Description |
| :------ | :---------- |
| [Narwhals](./narwhals_backend.html) | Unified opt-in backend for Polars, Ibis, PySpark SQL, and pandas. Enable with `PANDERA_USE_NARWHALS_BACKEND=True` or [`set_config`](./reference/generated/pandera.set_config.html). |

## Alternative Acceleration Frameworks

Pandera works with other dataframe-agnostic libraries that allow for distributed
dataframe validation:

| Library | Description |
| :------ | :---------- |
| [Fugue](./fugue.html) | Apply pandera schemas to distributed dataframe partitions with Fugue. |

!!! note

    Don't see a library that you want supported? Check out the
    [github issues](https://github.com/pandera-dev/pandera/issues) to see if
    that library is in the roadmap. If it isn't, open up a
    [new issue](https://github.com/pandera-dev/pandera/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=)
    to add support for it!
