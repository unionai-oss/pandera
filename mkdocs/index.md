# The Open-source Framework for Dataset Validation

> *Data validation for scientists, engineers, and analysts seeking correctness.*

[![CI Build](https://img.shields.io/github/actions/workflow/status/unionai-oss/pandera/ci-tests.yml?branch=main&label=tests&style=for-the-badge)](https://github.com/unionai-oss/pandera/actions/workflows/ci-tests.yml?query=branch%3Amain)
[![Documentation Stable](https://readthedocs.org/projects/pandera/badge/?version=stable&style=for-the-badge)](https://pandera.readthedocs.io/en/stable/?badge=stable)
[![PyPI](https://img.shields.io/pypi/v/pandera.svg?style=for-the-badge)](https://pypi.org/project/pandera/)
[![License](https://img.shields.io/pypi/l/pandera.svg?style=for-the-badge)](https://pypi.python.org/pypi/)
[![pyOpenSci Review](https://go.union.ai/pandera-pyopensci-badge)](https://github.com/pyOpenSci/software-review/issues/12)
[![Project Status: Active](https://img.shields.io/badge/repo%20status-Active-Green?style=for-the-badge)](https://www.repostatus.org/#active)
[![Documentation Latest](https://readthedocs.org/projects/pandera/badge/?version=latest&style=for-the-badge)](https://pandera.readthedocs.io/en/stable/?badge=latest)
[![Codecov](https://img.shields.io/codecov/c/github/unionai-oss/pandera?style=for-the-badge)](https://codecov.io/gh/unionai-oss/pandera)
[![Python versions](https://img.shields.io/pypi/pyversions/pandera.svg?style=for-the-badge)](https://pypi.python.org/pypi/pandera/)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.3385265-blue?style=for-the-badge)](https://doi.org/10.5281/zenodo.3385265)
[![benchmarked by asv](https://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=for-the-badge)](https://pandera-dev.github.io/pandera-asv-logs/)
[![Downloads](https://img.shields.io/pepy/dt/pandera?style=for-the-badge&color=blue)](https://pepy.tech/badge/pandera)
[![Conda](https://img.shields.io/conda/dn/conda-forge/pandera?style=for-the-badge)](https://anaconda.org/conda-forge/pandera)
[![Slack Community](https://img.shields.io/badge/Slack-4A154B?logo=slack&logoColor=fff&style=for-the-badge)](https://flyte-org.slack.com/archives/C08FDTY2X3L)

Pandera is a [Union.ai](https://union.ai/blog-post/pandera-joins-union-ai) open
source project that provides a flexible and expressive API for performing data
validation on dataframe-like objects. The goal of Pandera is to make data
processing pipelines more readable and robust with statistically typed
dataframes.

Dataframes contain information that `pandera` explicitly validates at runtime.
This is useful in production-critical data pipelines or reproducible research
settings. With `pandera`, you can:

1. Define a schema once and use it to validate [different dataframe types](./supported_libraries.html)
   including [pandas](http://pandas.pydata.org), [polars](https://docs.pola.rs/), [dask](https://dask.org/),
   [modin](https://modin.readthedocs.io/), [ibis](https://ibis-project.org/), and
   [pyspark](https://spark.apache.org/docs/latest/api/python/index.html). Use the [Narwhals-powered backend](./narwhals_backend.html) to keep validation fully lazy when possible.
1. [Check](./checks.html) the types and properties of columns in a
   `pd.DataFrame` or values in a `pd.Series`.
1. Perform more complex statistical validation like
   [hypothesis testing](./hypothesis.html).
1. [Parse](./parsers.html) data to standardize the preprocessing steps needed to
   produce valid data.
1. Seamlessly integrate with existing data analysis/processing pipelines
   via [function decorators](./decorators.html).
1. Define dataframe models with the [class-based API](./dataframe_models.html) with
   pydantic-style syntax and validate dataframes using the typing syntax.
1. [Synthesize data](./data_synthesis_strategies.html) from schema objects for
   property-based testing with pandas data structures.
1. [Lazily validate](./lazy_validation.html) dataframes so that all validation
   rules are executed before raising an error.
1. [Integrate](./integrations.html) with a rich ecosystem of python tools like
   [pydantic](https://pydantic-docs.helpmanual.io/),
   [fastapi](https://fastapi.tiangolo.com/) and [mypy](http://mypy-lang.org/).
1. [Derive](./parsers.html#derived-columns) column values from other columns with
   `ParsedField`/`ParsedColumn`, including asking a
   [decision model](./decisions.html) to parse unstructured text into typed,
   validated columns — see the [tutorial](./decision_parsing_tutorial.html).

## Install

Pandera supports [multiple dataframe libraries](./supported_libraries.html), including [pandas](http://pandas.pydata.org), [polars](https://docs.pola.rs/), [pyspark](https://spark.apache.org/docs/latest/api/python/index.html), and [ibis](https://ibis-project.org/).

Most of the documentation will use the `pandas` DataFrames; install Pandera with the `pandas` extra:

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

=== "pip"

    ```bash
    pip install 'pandera[hypotheses]'   # hypothesis checks
    pip install 'pandera[io]'           # yaml/script schema io utilities
    pip install 'pandera[frictionless]' # frictionless schema serialization/deserialization
    pip install 'pandera[strategies]'   # data synthesis strategies
    pip install 'pandera[mypy]'         # enable static type-linting of pandas
    pip install 'pandera[fastapi]'      # fastapi integration
    pip install 'pandera[dask]'         # validate dask dataframes
    pip install 'pandera[pyspark]'      # validate pyspark dataframes
    pip install 'pandera[modin]'        # validate modin dataframes
    pip install 'pandera[modin-ray]'    # validate modin dataframes with ray
    pip install 'pandera[modin-dask]'   # validate modin dataframes with dask
    pip install 'pandera[geopandas]'    # validate geopandas geodataframes
    pip install 'pandera[polars]'       # validate polars dataframes
    pip install 'pandera[ibis]'         # validate ibis tables
    pip install 'pandera[pyarrow]'      # validate pyarrow tables
    pip install 'pandera[datafusion]'   # validate datafusion dataframes
    pip install 'pandera[xarray]'       # validate xarray data structures
    pip install 'pandera[narwhals]'     # use the Narwhals-powered backend
    pip install 'pandera[cli]'          # command-line interface (Typer)
    ```

=== "conda"

    ```bash
    conda install -c conda-forge pandera-hypotheses   # hypothesis checks
    conda install -c conda-forge pandera-io           # yaml/script schema io utilities
    conda install -c conda-forge pandera-frictionless # frictionless schema serialization/deserialization
    conda install -c conda-forge pandera-strategies   # data synthesis strategies
    conda install -c conda-forge pandera-mypy         # enable static type-linting of pandas
    conda install -c conda-forge pandera-fastapi      # fastapi integration
    conda install -c conda-forge pandera-dask         # validate dask dataframes
    conda install -c conda-forge pandera-pyspark      # validate pyspark dataframes
    conda install -c conda-forge pandera-modin        # validate modin dataframes
    conda install -c conda-forge pandera-modin-ray    # validate modin dataframes with ray
    conda install -c conda-forge pandera-modin-dask   # validate modin dataframes with dask
    conda install -c conda-forge pandera-geopandas    # validate geopandas geodataframes
    conda install -c conda-forge pandera-polars       # validate polars dataframes
    conda install -c conda-forge pandera-ibis         # validate ibis tables
    conda install -c conda-forge pandera-narwhals     # use the Narwhals-powered backend
    ```

## Quick Start

```python
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
[`DataFrameModel`](./reference/generated/pandera.api.pandas.model.DataFrameModel.html) for the above
[`DataFrameSchema`](./reference/generated/pandera.api.pandas.container.DataFrameSchema.html) would be:

```python
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

!!! warning

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
    or functions will be deprecated in version `0.29.0`.

## Informative Errors

If the dataframe does not pass validation checks, `pandera` provides
useful error messages. An `error` argument can also be supplied to
`Check` for custom error messages.

In the case that a validation `Check` is violated:

```python
# validation rule violated
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

fail_check_df = pd.DataFrame({
    "column1": [-20, 5, 10, 30],
})

try:
    simple_schema(fail_check_df)
except pa.errors.SchemaError as exc:
    print(exc)
```

And in the case of a mis-specified column name:

```python
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
give an overview of error sources within a dataframe. Take the following schema
and dataframe:

```python
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
lists corresponding to a `SchemaError`.

## Supported Features by DataFrame Backend

Currently, pandera provides six validation backends: `pandas`, `pyspark`, `polars`,
`ibis`, `pyarrow`, and `datafusion`. The table below shows which of pandera's features
are available for the [supported dataframe libraries](./supported_libraries.html):

| feature | pandas | pyspark | polars | ibis | pyarrow | datafusion |
| :------ | ------ | ------- | ------ | ---- | ------- | ---------- |
| [DataFrameSchema validation](./dataframe_schemas.html)                      | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [DataFrameModel validation](./dataframe_models.html)                        | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [Nested DataFrameModels/DataFrameSchemas](./dataframe_models.html)          | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| [SeriesSchema validation](./series_schemas.html)                            | ✅ | 🚫 | ❌ | ❌ | ❌ | ❌ |
| [Index/MultiIndex validation](./dataframe_schemas.html)                     | ✅ | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 |
| [Built-in and custom Checks](./checks.html)                                 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [Groupby checks](./dataframe_schemas.html)                                  | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [Custom check registration](./extensions.html)                              | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| [Hypothesis testing](./hypothesis.html)                                     | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [Built-in](./dtype_validation.html) and [custom](./dtypes.html) `DataType`s | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [Preprocessing with Parsers](./parsers.html)                                | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [Derived columns (ParsedField/ParsedColumn)](./parsers.html#derived-columns) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [Decision parsing & semantic checks](./decisions.html)                      | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [Data synthesis strategies](./data_synthesis_strategies.html)               | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [Validation decorators](./decorators.html)                                  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [Lazy validation](./lazy_validation.html)                                   | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [Dropping invalid rows](./drop_invalid_rows.html)                           | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ |
| [Pandera configuration](./configuration.html)                               | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [Schema Inference](./schema_inference.html)                                 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [Schema persistence](./frictionless.html)                                   | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [Data Format Conversion](./data_format_conversion.html)                     | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |
| [Pydantic type support](./pydantic_integration.html)                        | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [FastAPI support](./fastapi_integration.html)                               | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

!!! important "Legend"

    - ✅: Supported
    - ❌: Not supported
    - 🚫: Not applicable

!!! note

    The `dask`, `modin`, `geopandas`, and `pyspark.pandas` support in pandera all
    leverage the pandas validation backend.

!!! note

    The `pyarrow` and `datafusion` backends are served entirely by the
    [Narwhals-powered backend](./narwhals_backend.html), so they need no opt-in
    configuration. These caveats apply to their columns above:

    - `coerce=True` is not yet implemented, so it emits a warning and reports a
      `WRONG_DATATYPE` error rather than casting.
    - [Data format conversion](./data_format_conversion.html) covers `dict`,
      `parquet` and `feather`, plus `csv` and `json` for `datafusion`.
    - A `datafusion.DataFrame` is a lazy query plan, so data-level checks only run
      when the validation depth is set to `SCHEMA_AND_DATA`.

    See [PyArrow](./pyarrow.html) and [DataFusion](./datafusion.html) for details.

!!! important

    *new in 0.32.0*; Pandera ships an optional
    [Narwhals-powered backend](./narwhals_backend.html) that keeps validation
    fully lazy when possible. Install the extra for each backend you use and
    enable it with either of the following:

    ```bash
    pip install 'pandera[narwhals,polars]'    # Polars
    pip install 'pandera[narwhals,ibis]'      # Ibis
    pip install 'pandera[narwhals,pyspark]'   # PySpark SQL

    export PANDERA_USE_NARWHALS_BACKEND=True
    ```

    ```python
    import pandera

    pandera.set_config(use_narwhals_backend=True)
    ```

    You can call [`pandera.set_config`](./reference/generated/pandera.set_config.html)
    before or after importing a pandera backend module. See
    [Backend registration](./narwhals_backend.html) for details on lazy registration
    and runtime re-registration.

## Contributing

All contributions, bug reports, bug fixes, documentation improvements,
enhancements and ideas are welcome.

A detailed overview on how to contribute can be found in the
[contributing guide](https://github.com/pandera-dev/pandera/blob/main/.github/CONTRIBUTING.md)
on GitHub.

## Issues

Submit issues, feature requests or bugfixes on
[github](https://github.com/pandera-dev/pandera/issues).

## Need Help?

There are many ways of getting help with your questions. You can ask a question
on [Github Discussions](https://github.com/pandera-dev/pandera/discussions/categories/q-a)
page or reach out to the maintainers and pandera community on
[Slack](https://flyte-org.slack.com/archives/C08FDTY2X3L).

## How to Cite

If you use `pandera` in the context of academic or industry research, please
consider citing the paper and/or software package.

### [Paper](https://conference.scipy.org/proceedings/scipy2020/niels_bantilan.html)

```bibtex
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

[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.3385265-blue?style=for-the-badge)](https://doi.org/10.5281/zenodo.3385265)

## License and Credits

`pandera` is licensed under the [MIT license](https://github.com/pandera-dev/pandera/blob/main/LICENSE.txt)
and is written and maintained by Niels Bantilan (niels@union.ai).
