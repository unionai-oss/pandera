<br>
<div align="center"><a href="https://www.union.ai/pandera"><img src="docs/source/_static/pandera-banner.png" width="400"></a></div>

<h1 align="center">
  The Open-source Framework for Precision Data Testing
</h1>

<p align="center">
  📊 🔎 ✅
</p>

<p align="center">
  <i>Data validation for scientists, engineers, and analysts seeking correctness.</i>
</p>

<br>


[![CI Build](https://img.shields.io/github/actions/workflow/status/unionai-oss/pandera/ci-tests.yml?branch=main&label=tests&style=for-the-badge)](https://github.com/unionai-oss/pandera/actions/workflows/ci-tests.yml?query=branch%3Amain)
[![Documentation Status](https://readthedocs.org/projects/pandera/badge/?version=stable&style=for-the-badge)](https://pandera.readthedocs.io/en/stable/?badge=stable)
[![PyPI version shields.io](https://img.shields.io/pypi/v/pandera.svg?style=for-the-badge)](https://pypi.org/project/pandera/)
[![PyPI license](https://img.shields.io/pypi/l/pandera.svg?style=for-the-badge)](https://pypi.python.org/pypi/)
[![pyOpenSci](https://go.union.ai/pandera-pyopensci-badge)](https://github.com/pyOpenSci/software-review/issues/12)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://img.shields.io/badge/repo%20status-Active-Green?style=for-the-badge)](https://www.repostatus.org/#active)
[![Documentation Status](https://readthedocs.org/projects/pandera/badge/?version=latest&style=for-the-badge)](https://pandera.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://img.shields.io/codecov/c/github/unionai-oss/pandera?style=for-the-badge)](https://codecov.io/gh/unionai-oss/pandera)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pandera.svg?style=for-the-badge)](https://pypi.python.org/pypi/pandera/)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.3385265-blue?style=for-the-badge)](https://doi.org/10.5281/zenodo.3385265)
[![asv](http://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=for-the-badge)](https://pandera-dev.github.io/pandera-asv-logs/)
[![Monthly Downloads](https://img.shields.io/pypi/dm/pandera?style=for-the-badge&color=blue)](https://pepy.tech/project/pandera)
[![Total Downloads](https://img.shields.io/pepy/dt/pandera?style=for-the-badge&color=blue)](https://pepy.tech/project/pandera)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/pandera?style=for-the-badge)](https://anaconda.org/conda-forge/pandera)
[![Discord](https://img.shields.io/badge/discord-chat-purple?color=%235765F2&label=discord&logo=discord&style=for-the-badge)](https://discord.gg/vyanhWuaKB)

`pandera` is a [Union.ai](https://union.ai/blog-post/pandera-joins-union-ai) open
source project that provides a flexible and expressive API for performing data
validation on dataframe-like objects to make data processing pipelines more readable and robust.

Dataframes contain information that `pandera` explicitly validates at runtime.
This is useful in production-critical or reproducible research settings. With
`pandera`, you can:

1. Define a schema once and use it to validate
   [different dataframe types](https://pandera.readthedocs.io/en/stable/supported_libraries.html)
   including [pandas](http://pandas.pydata.org), [polars](https://docs.pola.rs/),
   [dask](https://dask.org), [modin](https://modin.readthedocs.io/),
   and [pyspark](https://spark.apache.org/docs/3.2.0/api/python/user_guide/pandas_on_spark/index.html).
1. [Check](https://pandera.readthedocs.io/en/stable/checks.html) the types and
   properties of columns in a `DataFrame` or values in a `Series`.
1. Perform more complex statistical validation like
   [hypothesis testing](https://pandera.readthedocs.io/en/stable/hypothesis.html#hypothesis).
1. Seamlessly integrate with existing data analysis/processing pipelines
   via [function decorators](https://pandera.readthedocs.io/en/stable/decorators.html#decorators).
1. Define dataframe models with the
   [class-based API](https://pandera.readthedocs.io/en/stable/dataframe_models.html#dataframe-models)
   with pydantic-style syntax and validate dataframes using the typing syntax.
1. [Synthesize data](https://pandera.readthedocs.io/en/stable/data_synthesis_strategies.html#data-synthesis-strategies)
   from schema objects for property-based testing with pandas data structures.
1. [Lazily Validate](https://pandera.readthedocs.io/en/stable/lazy_validation.html)
   dataframes so that all validation checks are executed before raising an error.
1. [Integrate](https://pandera.readthedocs.io/en/stable/integrations.html) with
   a rich ecosystem of python tools like [pydantic](https://pydantic-docs.helpmanual.io),
   [fastapi](https://fastapi.tiangolo.com/), and [mypy](http://mypy-lang.org/).

## Documentation

The official documentation is hosted here: https://pandera.readthedocs.io


## Install

Using pip:

```
pip install pandera
```

Using conda:

```
conda install -c conda-forge pandera
```

### Extras

Installing additional functionality:

<details>

<summary><i>pip</i></summary>

```bash
pip install 'pandera[hypotheses]' # hypothesis checks
pip install 'pandera[io]'         # yaml/script schema io utilities
pip install 'pandera[strategies]' # data synthesis strategies
pip install 'pandera[mypy]'       # enable static type-linting of pandas
pip install 'pandera[fastapi]'    # fastapi integration
pip install 'pandera[dask]'       # validate dask dataframes
pip install 'pandera[pyspark]'    # validate pyspark dataframes
pip install 'pandera[modin]'      # validate modin dataframes
pip install 'pandera[modin-ray]'  # validate modin dataframes with ray
pip install 'pandera[modin-dask]' # validate modin dataframes with dask
pip install 'pandera[geopandas]'  # validate geopandas geodataframes
pip install 'pandera[polars]'     # validate polars dataframes
```

</details>

<details>

<summary><i>conda</i></summary>

```bash
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

</details>

## Quick Start

```python
import pandas as pd
import pandera as pa


# data to validate
df = pd.DataFrame({
    "column1": [1, 4, 0, 10, 9],
    "column2": [-1.3, -1.4, -2.9, -10.1, -20.4],
    "column3": ["value_1", "value_2", "value_3", "value_2", "value_1"]
})

# define schema
schema = pa.DataFrameSchema({
    "column1": pa.Column(int, checks=pa.Check.le(10)),
    "column2": pa.Column(float, checks=pa.Check.lt(-1.2)),
    "column3": pa.Column(str, checks=[
        pa.Check.str_startswith("value_"),
        # define custom checks as functions that take a series as input and
        # outputs a boolean or boolean Series
        pa.Check(lambda s: s.str.split("_", expand=True).shape[1] == 2)
    ]),
})

validated_df = schema(df)
print(validated_df)

#     column1  column2  column3
#  0        1     -1.3  value_1
#  1        4     -1.4  value_2
#  2        0     -2.9  value_3
#  3       10    -10.1  value_2
#  4        9    -20.4  value_1
```

## DataFrame Model

`pandera` also provides an alternative API for expressing schemas inspired
by [dataclasses](https://docs.python.org/3/library/dataclasses.html) and
[pydantic](https://pydantic-docs.helpmanual.io/). The equivalent `DataFrameModel`
for the above `DataFrameSchema` would be:


```python
from pandera.typing import Series

class Schema(pa.DataFrameModel):

    column1: int = pa.Field(le=10)
    column2: float = pa.Field(lt=-1.2)
    column3: str = pa.Field(str_startswith="value_")

    @pa.check("column3")
    def column_3_check(cls, series: Series[str]) -> Series[bool]:
        """Check that values have two elements after being split with '_'"""
        return series.str.split("_", expand=True).shape[1] == 2

Schema.validate(df)
```

## Development Installation

```
git clone https://github.com/pandera-dev/pandera.git
cd pandera
export PYTHON_VERSION=...  # specify desired python version
pip install -r dev/requirements-${PYTHON_VERSION}.txt
pip install -e .
```

## Tests

```
pip install pytest
pytest tests
```

## Contributing to pandera [![GitHub contributors](https://img.shields.io/github/contributors/pandera-dev/pandera.svg?style=for-the-badge)](https://github.com/pandera-dev/pandera/graphs/contributors)

All contributions, bug reports, bug fixes, documentation improvements,
enhancements and ideas are welcome.

A detailed overview on how to contribute can be found in the
[contributing guide](https://github.com/pandera-dev/pandera/blob/main/.github/CONTRIBUTING.md)
on GitHub.

## Issues

Go [here](https://github.com/pandera-dev/pandera/issues) to submit feature
requests or bugfixes.

## Need Help?

There are many ways of getting help with your questions. You can ask a question
on [Github Discussions](https://github.com/pandera-dev/pandera/discussions/categories/q-a)
page or reach out to the maintainers and pandera community on
[Discord](https://discord.gg/vyanhWuaKB)

## Why `pandera`?

- [dataframe-centric data types](https://pandera.readthedocs.io/en/stable/dtypes.html),
  [column nullability](https://pandera.readthedocs.io/en/stable/dataframe_schemas.html#null-values-in-columns),
  and [uniqueness](https://pandera.readthedocs.io/en/stable/dataframe_schemas.html#validating-the-joint-uniqueness-of-columns)
  are first-class concepts.
- Define [dataframe models](https://pandera.readthedocs.io/en/stable/schema_models.html) with the class-based API with
  [pydantic](https://pydantic-docs.helpmanual.io/)-style syntax and validate dataframes using the typing syntax.
- `check_input` and `check_output` [decorators](https://pandera.readthedocs.io/en/stable/decorators.html#decorators-for-pipeline-integration)
  enable seamless integration with existing code.
- [`Check`s](https://pandera.readthedocs.io/en/stable/checks.html) provide flexibility and performance by providing access to `pandas`
  API by design and offers built-in checks for common data tests.
- [`Hypothesis`](https://pandera.readthedocs.io/en/stable/hypothesis.html) class provides a tidy-first interface for statistical hypothesis
  testing.
- `Check`s and `Hypothesis` objects support both [tidy and wide data validation](https://pandera.readthedocs.io/en/stable/checks.html#wide-checks).
- Use schemas as generative contracts to [synthesize data](https://pandera.readthedocs.io/en/stable/data_synthesis_strategies.html) for unit testing.
- [Schema inference](https://pandera.readthedocs.io/en/stable/schema_inference.html) allows you to bootstrap schemas from data.

## How to Cite

If you use `pandera` in the context of academic or industry research, please
consider citing the **paper** and/or **software package**.

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

[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.3385265-blue?style=for-the-badge)](https://doi.org/10.5281/zenodo.3385265)


## License and Credits

`pandera` is licensed under the [MIT license](license.txt) and is written and
maintained by Niels Bantilan (niels@pandera.ci)
