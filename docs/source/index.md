---
file_format: mystnb
---

% pandera documentation entrypoint

# The Open-source Framework for Validating DataFrame-like Objects

> *Data validation for scientists, engineers, and analysts seeking correctness.*

```{image} https://img.shields.io/github/actions/workflow/status/unionai-oss/pandera/ci-tests.yml?branch=main&label=tests&style=for-the-badge
:alt: CI Build
:target: https://github.com/unionai-oss/pandera/actions/workflows/ci-tests.yml?query=branch%3Amain
```

```{image} https://readthedocs.org/projects/pandera/badge/?version=stable&style=for-the-badge
:alt: Documentation Stable Status
:target: https://pandera.readthedocs.io/en/stable/?badge=stable
```

```{image} https://img.shields.io/pypi/v/pandera.svg?style=for-the-badge
:alt: pypi
:target: https://pypi.org/project/pandera/
```

```{image} https://img.shields.io/pypi/l/pandera.svg?style=for-the-badge
:alt: pypi versions
:target: https://pypi.python.org/pypi/
```

```{image} https://go.union.ai/pandera-pyopensci-badge
:alt: pyOpenSci Review
:target: https://github.com/pyOpenSci/software-review/issues/12
```

```{image} https://img.shields.io/badge/repo%20status-Active-Green?style=for-the-badge
:alt: "Project Status: Active \u2013 The project has reached a stable, usable state\
:  \ and is being actively developed."
:target: https://www.repostatus.org/#active
```

```{image} https://readthedocs.org/projects/pandera/badge/?version=latest&style=for-the-badge
:alt: Documentation Latest Status
:target: https://pandera.readthedocs.io/en/stable/?badge=latest
```

```{image} https://img.shields.io/codecov/c/github/unionai-oss/pandera?style=for-the-badge
:alt: Code Coverage
:target: https://codecov.io/gh/unionai-oss/pandera
```

```{image} https://img.shields.io/pypi/pyversions/pandera.svg?style=for-the-badge
:alt: PyPI pyversions
:target: https://pypi.python.org/pypi/pandera/
```

```{image} https://img.shields.io/badge/DOI-10.5281/zenodo.3385265-blue?style=for-the-badge
:alt: DOI
:target: https://doi.org/10.5281/zenodo.3385265
```

```{image} http://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=for-the-badge
:alt: asv
:target: https://pandera-dev.github.io/pandera-asv-logs/
```

```{image} https://img.shields.io/pypi/dm/pandera?style=for-the-badge&color=blue
:alt: Monthly Downloads
:target: https://pepy.tech/project/pandera
```

```{image} https://img.shields.io/pepy/dt/pandera?style=for-the-badge&color=blue
:alt: Total Downloads
:target: https://pepy.tech/badge/pandera
```

```{image} https://img.shields.io/conda/dn/conda-forge/pandera?style=for-the-badge
:alt: Conda Downloads
:target: https://anaconda.org/conda-forge/pandera
```

```{image} https://img.shields.io/badge/Slack-4A154B?logo=slack&logoColor=fff&style=for-the-badge
:alt: Slack Community
:target: https://flyte-org.slack.com/archives/C08FDTY2X3L
```

Pandera is a [Union.ai](https://union.ai/blog-post/pandera-joins-union-ai) open
source project that provides a flexible and expressive API for performing data
validation on dataframe-like objects. The goal of Pandera is to make data
processing pipelines more readable and robust with statistically typed
dataframes.

Dataframes contain information that `pandera` explicitly validates at runtime.
This is useful in production-critical data pipelines or reproducible research
settings. With `pandera`, you can:

1. Define a schema once and use it to validate {ref}`different dataframe types <supported-dataframe-libraries>`
   including [pandas](http://pandas.pydata.org), [polars](https://docs.pola.rs/), [dask](https://dask.org/),
   [modin](https://modin.readthedocs.io/), and
   [pyspark](https://spark.apache.org/docs/latest/api/python/index.html).
2. {ref}`Check<checks>` the types and properties of columns in a
   `pd.DataFrame` or values in a `pd.Series`.
3. Perform more complex statistical validation like
   {ref}`hypothesis testing<hypothesis>`.
4. {ref}`Parse<parsers>` data to standardize the preprocessing steps needed to
   produce valid data.
5. Seamlessly integrate with existing data analysis/processing pipelines
   via {ref}`function decorators<decorators>`.
6. Define dataframe models with the {ref}`class-based API <dataframe-models>` with
   pydantic-style syntax and validate dataframes using the typing syntax.
7. {ref}`Synthesize data <data-synthesis-strategies>` from schema objects for
   property-based testing with pandas data structures.
8. {ref}`Lazily Validate <lazy-validation>` dataframes so that all validation
   rules are executed before raising an error.
9. {ref}`Integrate <integrations>` with a rich ecosystem of python tools like
   [pydantic](https://pydantic-docs.helpmanual.io/),
   [fastapi](https://fastapi.tiangolo.com/) and [mypy](http://mypy-lang.org/).

(installation)=

## Install

Pandera supports [multiple dataframe libraries](https://pandera.readthedocs.io/en/stable/supported_libraries.html), including [pandas](http://pandas.pydata.org), [polars](https://docs.pola.rs/), [pyspark](https://spark.apache.org/docs/latest/api/python/index.html), and more.

Most of the documentation will use the `pandas` DataFrames, install Pandera with the `pandas` extra:

With `pip`:

```bash
pip install 'pandera[pandas]'
```

With `uv`:

```
uv pip install 'pandera[pandas]'
```

With `conda`:

```bash
conda install -c conda-forge pandera-pandas
```

### Extras

Installing additional functionality:

::::{tab-set}

:::{tab-item} pip
```{code} bash
pip install 'pandera[hypotheses]'  # hypothesis checks
pip install 'pandera[io]'          # yaml/script schema io utilities
pip install 'pandera[strategies]'  # data synthesis strategies
pip install 'pandera[mypy]'        # enable static type-linting of pandas
pip install 'pandera[fastapi]'     # fastapi integration
pip install 'pandera[dask]'        # validate dask dataframes
pip install 'pandera[pyspark]'     # validate pyspark dataframes
pip install 'pandera[modin]'       # validate modin dataframes
pip install 'pandera[modin-ray]'   # validate modin dataframes with ray
pip install 'pandera[modin-dask]'  # validate modin dataframes with dask
pip install 'pandera[geopandas]'   # validate geopandas geodataframes
pip install 'pandera[polars]'      # validate polars dataframes
```
:::

:::{tab-item} conda
```{code} bash
conda install -c conda-forge pandera-hypotheses  # hypothesis checks
conda install -c conda-forge pandera-io          # yaml/script schema io utilities
conda install -c conda-forge pandera-strategies  # data synthesis strategies
conda install -c conda-forge pandera-mypy        # enable static type-linting of pandas
conda install -c conda-forge pandera-fastapi     # fastapi integration
conda install -c conda-forge pandera-dask        # validate dask dataframes
conda install -c conda-forge pandera-pyspark     # validate pyspark dataframes
conda install -c conda-forge pandera-modin       # validate modin dataframes
conda install -c conda-forge pandera-modin-ray   # validate modin dataframes with ray
conda install -c conda-forge pandera-modin-dask  # validate modin dataframes with dask
conda install -c conda-forge pandera-geopandas   # validate geopandas geodataframes
conda install -c conda-forge pandera-polars      # validate polars dataframes
```
:::
::::

## Quick Start

```{code-cell} python
import pandas as pd
import pandera.pandas as pa

# data to validate
df = pd.DataFrame({
    "column1": [1, 2, 3],
    "column2": [1.1, 1.2, 1.3],
    "column3": ["a", "b", "c"],
})

schema = pa.DataFrameSchema({
    "column1": pa.Column(int, pa.Check.ge(0)),
    "column2": pa.Column(float, pa.Check.lt(10)),
    "column3": pa.Column(
        str,
        [
            pa.Check.isin([*"abc"]),
            pa.Check(lambda series: series.str.len() == 1),
        ]
    ),
}
)

validated_df = schema.validate(df)
print(validated_df)
```

## Dataframe Model

`pandera` also provides a class-based API for writing schemas inspired by
[dataclasses](https://docs.python.org/3/library/dataclasses.html) and
[pydantic](https://docs.pydantic.dev/latest/). The equivalent
{class}`~pandera.api.pandas.model.DataFrameModel` for the above
{class}`~pandera.pandas.DataFrameSchema` would be:

```{code-cell} python
# define a schema
class Schema(pa.DataFrameModel):
    column1: int = pa.Field(ge=0)
    column2: float = pa.Field(lt=10)
    column3: str = pa.Field(isin=[*"abc"])

    @pa.check("column3")
    def custom_check(cls, series: pd.Series) -> pd.Series:
        return series.str.len() == 1

Schema.validate(df)
```

:::{warning}
Pandera `v0.24.0` introduces the `pandera.pandas` module, which is now the
(highly) recommended way of defining `DataFrameSchema`s and `DataFrameModel`s
for `pandas` data structures like `DataFrame`s. Defining a dataframe schema from
the top-level `pandera` module will produce a `FutureWarning`:

```python
import pandera as pa

schema = pa.DataFrameSchema({"col": pa.Column(str)})
```

Update your import to:

```python
import pandera.pandas as pa
```

And all of the rest of your pandera code should work. Using the top-level
`pandera` module to access `DataFrameSchema` and the other pandera classes
or functions will be deprecated in a future version
:::

## Informative Errors

If the dataframe does not pass validation checks, `pandera` provides
useful error messages. An `error` argument can also be supplied to
`Check` for custom error messages.

In the case that a validation `Check` is violated:

```{code-cell} python
:tags: [raises-exception]

simple_schema = pa.DataFrameSchema({
    "column1": pa.Column(
        int,
        pa.Check(
            lambda x: 0 <= x <= 10,
            element_wise=True,
            error="range checker [0, 10]"
        )
    )
})

# validation rule violated
fail_check_df = pd.DataFrame({
    "column1": [-20, 5, 10, 30],
})

try:
    simple_schema(fail_check_df)
except pa.errors.SchemaError as exc:
    print(exc)
```

And in the case of a mis-specified column name:

```{code-cell} python
:tags: [raises-exception]

# column name mis-specified
wrong_column_df = pd.DataFrame({
    "foo": ["bar"] * 10,
    "baz": [1] * 10
})


try:
    simple_schema(wrong_column_df)
except pa.errors.SchemaError as exc:
    print(exc)
```

## Error Reports

If the dataframe is validated lazily with `lazy=True`, errors will be aggregated
into an error report. The error report groups `DATA` and `SCHEMA` errors to
to give an overview of error sources within a dataframe. Take the following schema
and dataframe:

```{code-cell} python
:tags: [raises-exception]

schema = pa.DataFrameSchema(
    {"id": pa.Column(int, pa.Check.lt(10))},
    name="MySchema",
    strict=True,
)

df = pd.DataFrame({"id": [1, None, 30], "extra_column": [1, 2, 3]})

try:
    schema.validate(df, lazy=True)
except pa.errors.SchemaErrors as exc:
    print(exc)
```

Validating the above dataframe will result in data level errors, namely the `id`
column having a value which fails a check, as well as schema level errors, such as the
extra column and the `None` value.


This error report can be useful for debugging, with each item in the various
lists corresponding to a `SchemaError`


(supported-features)=

## Supported Features by DataFrame Backend

Currently, pandera provides three validation backends: `pandas`, `pyspark`, and
`polars`. The table below shows which of pandera's features are available for the
{ref}`supported dataframe libraries <dataframe-libraries>`:

:::{table}
:widths: auto
:align: left

| feature | pandas | pyspark | polars |
| :------ | ------ | ------- | ------ |
| {ref}`DataFrameSchema validation <dataframeschemas>`                      | ✅ | ✅ | ✅ |
| {ref}`DataFrameModel validation <dataframe-models>`                       | ✅ | ✅ | ✅ |
| {ref}`SeriesSchema validation <seriesschemas>`                            | ✅ | 🚫 | ❌ |
| {ref}`Index/MultiIndex validation <index-validation>`                     | ✅ | 🚫 | 🚫 |
| {ref}`Built-in and custom Checks <checks>`                                | ✅ | ✅ | ✅ |
| {ref}`Groupby checks <column-check-groups>`                               | ✅ | ❌ | ❌ |
| {ref}`Custom check registration <extensions>`                             | ✅ | ✅ | ❌ |
| {ref}`Hypothesis testing <hypothesis>`                                    | ✅ | ❌ | ❌ |
| {ref}`Built-in <dtype-validation>` and {ref}`custom <dtypes>` `DataType`s | ✅ | ✅ | ✅ |
| {ref}`Preprocessing with Parsers <parsers>`                               | ✅ | ❌ | ❌ |
| {ref}`Data synthesis strategies <data-synthesis-strategies>`              | ✅ | ❌ | ❌ |
| {ref}`Validation decorators <decorators>`                                 | ✅ | ✅ | ✅ |
| {ref}`Lazy validation <lazy-validation>`                                  | ✅ | ✅ | ✅ |
| {ref}`Dropping invalid rows <drop-invalid-rows>`                          | ✅ | ❌ | ✅ |
| {ref}`Pandera configuration <configuration>`                              | ✅ | ✅ | ✅ |
| {ref}`Schema Inference <schema-inference>`                                | ✅ | ❌ | ❌ |
| {ref}`Schema persistence <schema-persistence>`                            | ✅ | ❌ | ❌ |
| {ref}`Data Format Conversion <data-format-conversion>`                    | ✅ | ❌ | ❌ |
| {ref}`Pydantic type support <pydantic-integration>`                       | ✅ | ❌ | ❌ |
| {ref}`FastAPI support <fastapi-integration>`                              | ✅ | ❌ | ❌ |

:::

:::{admonition} Legend
:class: important

- ✅: Supported
- ❌: Not supported
- 🚫: Not applicable
:::


:::{note}
The `dask`, `modin`, `geopandas`, and `pyspark.pandas` support in pandera all
leverage the pandas validation backend.
:::


## Contributing

All contributions, bug reports, bug fixes, documentation improvements,
enhancements and ideas are welcome.

A detailed overview on how to contribute can be found in the
[contributing
guide](https://github.com/pandera-dev/pandera/blob/main/.github/CONTRIBUTING.md)
on GitHub.

## Issues

Submit issues, feature requests or bugfixes on
[github](https://github.com/pandera-dev/pandera/issues).

## Need Help?

There are many ways of getting help with your questions. You can ask a question
on [Github Discussions](https://github.com/pandera-dev/pandera/discussions/categories/q-a)
page or reach out to the maintainers and pandera community on
[Slack](https://flyte-org.slack.com/archives/C08FDTY2X3L)

```{toctree}
:caption: Introduction
:hidden: true
:maxdepth: 6

Welcome to Pandera <self>
▶️ Try Pandera <https://colab.research.google.com/github/unionai-oss/pandera/blob/main/docs/source/notebooks/try_pandera.ipynb>
Official Website <https://union.ai/pandera>
```

```{toctree}
:caption: User Guide
:hidden: true
:maxdepth: 6

dataframe_schemas
dataframe_models
series_schemas
dtype_validation
checks
hypothesis
parsers
dtypes
decorators
drop_invalid_rows
schema_inference
lazy_validation
error_report
data_synthesis_strategies
extensions
data_format_conversion
supported_libraries
integrations
configuration
```

```{toctree}
:caption: Reference
:hidden: true
:maxdepth: 6

reference/index
```

```{toctree}
:caption: Community
:hidden: true
:maxdepth: 6

CONTRIBUTING
```

## How to Cite

If you use `pandera` in the context of academic or industry research, please
consider citing the paper and/or software package.

### [Paper](https://conference.scipy.org/proceedings/scipy2020/niels_bantilan.html)

```
@InProceedings{ niels_bantilan-proc-scipy-2020,
  author    = { {N}iels {B}antilan },
  title     = { pandera: {S}tatistical {D}ata {V}alidation of {P}andas {D}ataframes },
  booktitle = { {P}roceedings of the 19th {P}ython in {S}cience {C}onference },
  pages     = { 116 - 124 },
  year      = { 2020 },
  editor    = { {M}eghann {A}garwal and {C}hris {C}alloway and {D}illon {N}iederhut and {D}avid {S}hupe },
  doi       = { 10.25080/Majora-342d178e-010 }
}
```

### Software Package

```{image} https://img.shields.io/badge/DOI-10.5281/zenodo.3385265-blue?style=for-the-badge
:alt: DOI
:target: https://doi.org/10.5281/zenodo.3385265
```

## License and Credits

`pandera` is licensed under the [MIT license](https://github.com/pandera-dev/pandera/blob/main/LICENSE.txt)
and is written and maintained by Niels Bantilan (niels@union.ai)

# Indices and tables

- {ref}`genindex`
