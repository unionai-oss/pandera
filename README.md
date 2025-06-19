<br>
<div align="center"><a href="https://www.union.ai/pandera"><img src="docs/source/_static/pandera-banner.png" width="400"></a></div>

<h1 align="center">
  The Open-source Framework for Validating DataFrame-like Objects
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
[![Slack](https://img.shields.io/badge/Slack-4A154B?logo=slack&logoColor=fff&style=for-the-badge)](https://flyte-org.slack.com/archives/C08FDTY2X3L)

Pandera is a [Union.ai](https://union.ai/blog-post/pandera-joins-union-ai) open
source project that provides a flexible and expressive API for performing data
validation on dataframe-like objects. The goal of Pandera is to make data
processing pipelines more readable and robust with statistically typed
dataframes.

## Install

Pandera supports [multiple dataframe libraries](https://pandera.readthedocs.io/en/stable/supported_libraries.html), including [pandas](http://pandas.pydata.org), [polars](https://docs.pola.rs/), [pyspark](https://spark.apache.org/docs/latest/api/python/index.html), and more. To validate `pandas` DataFrames, install Pandera with the `pandas` extra:

**With `pip`:**

```
pip install 'pandera[pandas]'
```

**With `uv`:**

```
uv pip install 'pandera[pandas]'
```

**With `conda`:**

```
conda install -c conda-forge pandera-pandas
```

## Get started

First, create a dataframe:

```python
import pandas as pd
import pandera.pandas as pa

# data to validate
df = pd.DataFrame({
    "column1": [1, 2, 3],
    "column2": [1.1, 1.2, 1.3],
    "column3": ["a", "b", "c"],
})
```

Validate the data using the object-based API:

```python
# define a schema
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
})

print(schema.validate(df))
#    column1  column2 column3
# 0        1      1.1       a
# 1        2      1.2       b
# 2        3      1.3       c
```

Or validate the data using the class-based API:

```python
# define a schema
class Schema(pa.DataFrameModel):
    column1: int = pa.Field(ge=0)
    column2: float = pa.Field(lt=10)
    column3: str = pa.Field(isin=[*"abc"])

    @pa.check("column3")
    def custom_check(cls, series: pd.Series) -> pd.Series:
        return series.str.len() == 1

print(Schema.validate(df))
#    column1  column2 column3
# 0        1      1.1       a
# 1        2      1.2       b
# 2        3      1.3       c
```


> [!WARNING]
> Pandera `v0.24.0` introduces the `pandera.pandas` module, which is now the
> (highly) recommended way of defining `DataFrameSchema`s and `DataFrameModel`s
> for `pandas` data structures like `DataFrame`s. Defining a dataframe schema from
> the top-level `pandera` module will produce a `FutureWarning`:
>
> ```python
> import pandera as pa
>
> schema = pa.DataFrameSchema({"col": pa.Column(str)})
> ```
>
> Update your import to:
>
> ```python
> import pandera.pandas as pa
> ```
>
> And all of the rest of your pandera code should work. Using the top-level
> `pandera` module to access `DataFrameSchema` and the other pandera classes
> or functions will be deprecated in a future version


## Next steps

See the [official documentation](https://pandera.readthedocs.io) to learn more.
