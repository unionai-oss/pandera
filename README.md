<br>
<div align="center"><img src="https://raw.githubusercontent.com/pandera-dev/pandera/master/docs/source/_static/pandera-banner.png" width="400"></div>

<hr>

*A data validation library for scientists, engineers, and analysts seeking
correctness.*

<br>

[![Build Status](https://travis-ci.org/pandera-dev/pandera.svg?branch=master)](https://travis-ci.org/pandera-dev/pandera)
[![Documentation Status](https://readthedocs.org/projects/pandera/badge/?version=stable)](https://pandera.readthedocs.io/en/stable/?badge=stable)
[![PyPI version shields.io](https://img.shields.io/pypi/v/pandera.svg)](https://pypi.org/project/pandera/)
[![PyPI license](https://img.shields.io/pypi/l/pandera.svg)](https://pypi.python.org/pypi/)
[![pyOpenSci](https://tinyurl.com/y22nb8up)](https://github.com/pyOpenSci/software-review/issues/12)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Documentation Status](https://readthedocs.org/projects/pandera/badge/?version=latest)](https://pandera.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/pandera-dev/pandera/branch/master/graph/badge.svg)](https://codecov.io/gh/pandera-dev/pandera)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pandera.svg)](https://pypi.python.org/pypi/pandera/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3926689.svg)](https://doi.org/10.5281/zenodo.3926689)
[![asv](http://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=flat)](https://pandera-dev.github.io/pandera-asv-logs/)

`pandas` data structures contain information that `pandera` explicitly
validates at runtime. This is useful in production-critical or reproducible
research settings. With `pandera`, you can:

1. Check the types and properties of columns in a `DataFrame` or values in
   a `Series`.
1. Perform more complex statistical validation like hypothesis testing.
1. Seamlessly integrate with existing data analysis/processing pipelines
   via function decorators.

`pandera` provides a flexible and expressive API for performing data validation
on tidy (long-form) and wide data to make data processing pipelines more
readable and robust.


## Documentation

The official documentation is hosted on ReadTheDocs: https://pandera.readthedocs.io

## Install

Using pip:

```
pip install pandera
```

Using conda:

```
conda install -c conda-forge pandera
```

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
    "column1": pa.Column(pa.Int, checks=pa.Check.less_than_or_equal_to(10)),
    "column2": pa.Column(pa.Float, checks=pa.Check.less_than(-1.2)),
    "column3": pa.Column(pa.String, checks=[
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

## Development Installation

```
git clone https://github.com/pandera-dev/pandera.git
cd pandera
pip install -r requirements-dev.txt
pip install -e .
```

## Tests

```
pip install pytest
pytest tests
```

## Contributing to pandera [![GitHub contributors](https://img.shields.io/github/contributors/pandera-dev/pandera.svg)](https://github.com/pandera-dev/pandera/graphs/contributors)

All contributions, bug reports, bug fixes, documentation improvements,
enhancements and ideas are welcome.

A detailed overview on how to contribute can be found in the
[contributing guide](https://github.com/pandera-dev/pandera/blob/master/.github/CONTRIBUTING.md)
on GitHub.

## Issues

Go [here](https://github.com/pandera-dev/pandera/issues) to submit feature
requests or bugfixes.

## Other Data Validation Libraries

Here are a few other alternatives for validating Python data structures.

**Generic Python object data validation**

- [voloptuous](https://github.com/alecthomas/voluptuous)
- [schema](https://github.com/keleshev/schema)

**`pandas`-specific data validation**

- [opulent-pandas](https://github.com/danielvdende/opulent-pandas)
- [PandasSchema](https://github.com/TMiguelT/PandasSchema)
- [pandas-validator](https://github.com/c-data/pandas-validator)
- [table_enforcer](https://github.com/xguse/table_enforcer)

**Other tools that include data validation**

- [great_expectations](https://github.com/great-expectations/great_expectations)

## Why `pandera`?

- `pandas`-centric data types, column nullability, and uniqueness are
  first-class concepts.
- `check_input` and `check_output` decorators enable seamless integration with
  existing code.
- `Check`s provide flexibility and performance by providing access to `pandas`
  API by design.
- `Hypothesis` class provides a tidy-first interface for statistical hypothesis
  testing.
- `Check`s and `Hypothesis` objects support both tidy and wide data validation.
- Comprehensive documentation on key functionality.


### Citation Information

#### [Paper](https://conference.scipy.org/proceedings/scipy2020/niels_bantilan.html)

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

#### Software Package

```
@software{niels_bantilan_2020_3926689,
  author       = {Niels Bantilan and
                  Nigel Markey and
                  Riccardo Albertazzi and
                  Nemanja Radojković and
                  chr1st1ank and
                  Aditya Singh and
                  Anthony Truchet - C3.AI and
                  Steve Taylor and
                  Sunho Kim and
                  Zachary Lawrence},
  title        = {{pandera-dev/pandera: 0.4.4: bugfixes in yaml
                   serialization, error reporting, refactor internals}},
  month        = jul,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {0.4.4},
  doi          = {10.5281/zenodo.3926689},
  url          = {https://doi.org/10.5281/zenodo.3926689}
}
```
