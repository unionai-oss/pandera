<div align="left"><img src="https://raw.githubusercontent.com/pandera-dev/pandera/master/docs/source/_static/pandera-logo.png" width="140"></div>

# Pandera

A flexible and expressive [pandas](http://pandas.pydata.org) validation library.

<br>

[![Build Status](https://travis-ci.org/pandera-dev/pandera.svg?branch=master)](https://travis-ci.org/pandera-dev/pandera)
[![PyPI version shields.io](https://img.shields.io/pypi/v/pandera.svg)](https://pypi.org/project/pandera/)
[![PyPI license](https://img.shields.io/pypi/l/pandera.svg)](https://pypi.python.org/pypi/)
[![pyOpenSci](https://tinyurl.com/y22nb8up)](https://github.com/pyOpenSci/software-review/issues/12)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Documentation Status](https://readthedocs.org/projects/pandera/badge/?version=latest)](https://pandera.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/pandera-dev/pandera/branch/master/graph/badge.svg)](https://codecov.io/gh/pandera-dev/pandera)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pandera.svg)](https://pypi.python.org/pypi/pandera/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3385266.svg)](https://doi.org/10.5281/zenodo.3385266)
[![asv](http://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=flat)](https://pandera-dev.github.io/pandera-asv-logs/)

`pandas` data structures contain information that `pandera` explicitly
validates at runtime. This is useful in production-critical or reproducible research
settings. `pandera` enables users to:

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

## Example Usage

### `DataFrameSchema`

```python
import pandas as pd
import pandera as pa

from pandera import Column, DataFrameSchema, Check, check_output


# validate columns
schema = DataFrameSchema({
    # the check function expects a series argument and should output a boolean
    # or a boolean Series.
    "column1": Column(pa.Int, Check(lambda s: s <= 10)),
    "column2": Column(pa.Float, Check(lambda s: s < -1.2)),
    # you can provide a list of validators
    "column3": Column(pa.String, [
        Check(lambda s: s.str.startswith("value_")),
        Check(lambda s: s.str.split("_", expand=True).shape[1] == 2)
    ]),
})

df = pd.DataFrame({
    "column1": [1, 4, 0, 10, 9],
    "column2": [-1.3, -1.4, -2.9, -10.1, -20.4],
    "column3": ["value_1", "value_2", "value_3", "value_2", "value_1"]
})

validated_df = schema.validate(df)
print(validated_df)

#     column1  column2  column3
#  0        1     -1.3  value_1
#  1        4     -1.4  value_2
#  2        0     -2.9  value_3
#  3       10    -10.1  value_2
#  4        9    -20.4  value_1

# If you have an existing data pipeline that uses pandas data structures, you can use the check_input and check_output decorators to check function arguments or returned variables from existing functions.

@check_output(schema)
def custom_function(df):
    return df
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

All contributions, bug reports, bug fixes, documentation improvements, enhancements and ideas are welcome.

A detailed overview on how to contribute can be found in the [contributing guide](https://github.com/pandera-dev/pandera/blob/master/.github/CONTRIBUTING.md) on GitHub.

## Issues

Go [here](https://github.com/pandera-dev/pandera-dev/issues) to submit feature
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

```
@misc{niels_bantilan_2019_3385266,
  author       = {Niels Bantilan and
                  Nigel Markey and
                  Riccardo Albertazzi and
                  chr1st1ank},
  title        = {pandera-dev/pandera: 0.2.0 pre-release 1},
  month        = sep,
  year         = 2019,
  doi          = {10.5281/zenodo.3385266},
  url          = {https://doi.org/10.5281/zenodo.3385266}
}
```
