.. pandera documentation master file

A Statistical DataFrame Testing Toolkit
=======================================

*A dataframe validation library for scientists, engineers, and analysts seeking
correctness.*


.. image:: https://github.com/pandera-dev/pandera/workflows/CI%20Tests/badge.svg?branch=master
    :target: https://github.com/pandera-dev/pandera/actions?query=workflow%3A%22CI+Tests%22+branch%3Amaster
    :alt: CI Build

.. image:: https://readthedocs.org/projects/pandera/badge/?version=stable
    :target: https://pandera.readthedocs.io/en/stable/?badge=stable
    :alt: Documentation Stable Status

.. image:: https://img.shields.io/pypi/v/pandera.svg
    :target: https://pypi.org/project/pandera/
    :alt: pypi

.. image:: https://img.shields.io/pypi/l/pandera.svg
    :target: https://pypi.python.org/pypi/
    :alt: pypi versions

.. image:: https://tinyurl.com/y22nb8up
    :target: https://github.com/pyOpenSci/software-review/issues/12
    :alt: pyOpenSci Review

.. image:: https://www.repostatus.org/badges/latest/active.svg
    :target: https://www.repostatus.org/#active
    :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.

.. image:: https://readthedocs.org/projects/pandera/badge/?version=latest
    :target: https://pandera.readthedocs.io/en/stable/?badge=latest
    :alt: Documentation Latest Status

.. image:: https://codecov.io/gh/pandera-dev/pandera/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/pandera-dev/pandera
    :alt: Code Coverage

.. image:: https://img.shields.io/pypi/pyversions/pandera.svg
    :target: https://pypi.python.org/pypi/pandera/
    :alt: PyPI pyversions

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3385265.svg
    :target: https://doi.org/10.5281/zenodo.3385265
    :alt: DOI

.. image:: http://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=flat
    :target: https://pandera-dev.github.io/pandera-asv-logs/
    :alt: asv

.. image:: https://pepy.tech/badge/pandera/month
    :target: https://pepy.tech/project/pandera
    :alt: Monthly Downloads

.. image:: https://pepy.tech/badge/pandera
    :target: https://pepy.tech/badge/pandera
    :alt: Total Downloads

.. image:: https://img.shields.io/conda/dn/conda-forge/pandera?label=conda%20downloads
    :target: https://anaconda.org/conda-forge/pandera
    :alt: Conda Downloads

.. image:: https://img.shields.io/badge/discord-chat-purple?color=%235765F2&label=discord&logo=discord
    :target: https://discord.gg/vyanhWuaKB
    :alt: Discord Community

|

``pandera`` provides a flexible and expressive API for performing data
validation on dataframes to make data processing pipelines more readable and
robust.

Dataframes contain information that ``pandera`` explicitly validates at runtime.
This is useful in production-critical data pipelines or reproducible research
settings. With ``pandera``, you can:

#. Define a schema once and use it to validate :ref:`different dataframe types <supported-dataframe-libraries>`
   including `pandas <http://pandas.pydata.org>`_, `dask <https://dask.org/>`_,
   `modin <https://modin.readthedocs.io/>`_, and `koalas <https://koalas.readthedocs.io/>`_.
#. :ref:`Check<checks>` the types and properties of columns in a
   ``pd.DataFrame`` or values in a ``pd.Series``.
#. Perform more complex statistical validation like
   :ref:`hypothesis testing<hypothesis>`.
#. Seamlessly integrate with existing data analysis/processing pipelines
   via :ref:`function decorators<decorators>`.
#. Define schema models with the :ref:`class-based API<schema_models>` with
   pydantic-style syntax and validate dataframes using the typing syntax.
#. :ref:`Synthesize data<data synthesis strategies>` from schema objects for
   property-based testing with pandas data structures.
#. :ref:`Lazily Validate<lazy_validation>` dataframes so that all validation
   rules are executed before raising an error.
#. :ref:`Integrate <integrations>` with a rich ecosystem of python tools like
   `pydantic <https://pydantic-docs.helpmanual.io/>`_ and
   `mypy <http://mypy-lang.org/>`_.


.. _installation:


Install
-------

Install with `pip`:

.. code:: bash

    pip install pandera

Or conda:

.. code:: bash

    conda install -c conda-forge pandera-core  # core library functionality
    conda install -c conda-forge pandera       # pandera with all extensions


Extras
~~~~~~

Installing additional functionality:

.. code:: bash

    pip install pandera[hypotheses]  # hypothesis checks
    pip install pandera[io]          # yaml/script schema io utilities
    pip install pandera[strategies]  # data synthesis strategies
    pip install pandera[mypy]        # enable static type-linting of pandas
    pip install pandera[dask]        # validate dask dataframes
    pip install pandera[koalas]      # validate koalas dataframes
    pip install pandera[modin]       # validate modin dataframes
    pip install pandera[modin-ray]   # validate modin dataframes with ray
    pip install pandera[modin-dask]  # validate modin dataframes with dask
    pip install pandera[geopandas]   # validate geopandas geodataframes
    pip install pandera[all]         # all packages


Quick Start
-----------

.. testcode:: quick_start

    import pandas as pd
    import pandera as pa

    # data to validate
    df = pd.DataFrame({
        "column1": [1, 4, 0, 10, 9],
        "column2": [-1.3, -1.4, -2.9, -10.1, -20.4],
        "column3": ["value_1", "value_2", "value_3", "value_2", "value_1"],
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

.. testoutput:: quick_start

       column1  column2  column3
    0        1     -1.3  value_1
    1        4     -1.4  value_2
    2        0     -2.9  value_3
    3       10    -10.1  value_2
    4        9    -20.4  value_1

You can pass the built-in python types that are supported by
pandas, or strings representing the
`legal pandas datatypes <https://pandas.pydata.org/docs/user_guide/basics.html#dtypes>`_,
or pandera's ``DataType``:

.. testcode:: quick_start

    schema = pa.DataFrameSchema({
        # built-in python types
        "int_column": pa.Column(int),
        "float_column": pa.Column(float),
        "str_column": pa.Column(str),

        # pandas dtype string aliases
        "int_column2": pa.Column("int64"),
        "float_column2": pa.Column("float64"),
        # pandas > 1.0.0 support native "string" type
        "str_column2": pa.Column("str"),

        # pandera DataType
        "int_column3": pa.Column(pa.Int),
        "float_column3": pa.Column(pa.Float),
        "str_column3": pa.Column(pa.String),
    })

For more details on data types, see :class:`~pandera.dtypes.DataType`


Schema Model
------------

``pandera`` also provides an alternative API for expressing schemas inspired
by `dataclasses <https://docs.python.org/3/library/dataclasses.html>`_ and
`pydantic <https://pydantic-docs.helpmanual.io/>`_. The equivalent
:class:`~pandera.model.SchemaModel` for the above
:class:`~pandera.scheams.DataFrameSchema` would be:

.. testcode:: quick_start

   from pandera.typing import Series

   class Schema(pa.SchemaModel):

       column1: Series[int] = pa.Field(le=10)
       column2: Series[float] = pa.Field(lt=-1.2)
       column3: Series[str] = pa.Field(str_startswith="value_")

       @pa.check("column3")
       def column_3_check(cls, series: Series[str]) -> Series[bool]:
           """Check that column3 values have two elements after being split with '_'"""
           return series.str.split("_", expand=True).shape[1] == 2

   Schema.validate(df)


Informative Errors
------------------

If the dataframe does not pass validation checks, ``pandera`` provides
useful error messages. An ``error`` argument can also be supplied to
``Check`` for custom error messages.

In the case that a validation ``Check`` is violated:

.. testcode:: informative_errors

    import pandas as pd

    from pandera import Column, DataFrameSchema, Int, Check

    simple_schema = DataFrameSchema({
        "column1": Column(
            Int, Check(lambda x: 0 <= x <= 10, element_wise=True,
                       error="range checker [0, 10]"))
    })

    # validation rule violated
    fail_check_df = pd.DataFrame({
        "column1": [-20, 5, 10, 30],
    })

    simple_schema(fail_check_df)


.. testoutput:: informative_errors

    Traceback (most recent call last):
    ...
    SchemaError: <Schema Column: 'column1' type=<class 'int'>> failed element-wise validator 0:
    <Check <lambda>: range checker [0, 10]>
    failure cases:
       index  failure_case
    0      0           -20
    1      3            30


And in the case of a mis-specified column name:

.. testcode:: informative_errors

    # column name mis-specified
    wrong_column_df = pd.DataFrame({
       "foo": ["bar"] * 10,
       "baz": [1] * 10
    })

    simple_schema.validate(wrong_column_df)


.. testoutput:: informative_errors

    Traceback (most recent call last):
    ...
    pandera.SchemaError: column 'column1' not in dataframe
       foo  baz
    0  bar    1
    1  bar    1
    2  bar    1
    3  bar    1
    4  bar    1

Contributing
------------

All contributions, bug reports, bug fixes, documentation improvements,
enhancements and ideas are welcome.

A detailed overview on how to contribute can be found in the
`contributing
guide <https://github.com/pandera-dev/pandera/blob/master/.github/CONTRIBUTING.md>`__
on GitHub.

Issues
------

Submit issues, feature requests or bugfixes on
`github <https://github.com/pandera-dev/pandera/issues>`__.

Need Help?
----------

There are many ways of getting help with your questions. You can ask a question
on `Github Discussions <https://github.com/pandera-dev/pandera/discussions/categories/q-a>`__
page or reach out to the maintainers and pandera community on
`Discord <https://discord.gg/vyanhWuaKB>`__

.. toctree::
    :maxdepth: 6
    :caption: Introduction
    :hidden:

    self

.. toctree::
   :maxdepth: 6
   :caption: User Guide
   :hidden:

   dataframe_schemas
   schema_models
   series_schemas
   checks
   hypothesis
   dtypes
   decorators
   schema_inference
   lazy_validation
   data_synthesis_strategies
   extensions
   data_format_conversion
   supported_libraries
   integrations

.. toctree::
   :maxdepth: 6
   :caption: Reference
   :hidden:

   reference/index

.. toctree::
   :maxdepth: 6
   :caption: Community
   :hidden:

   CONTRIBUTING

How to Cite
-----------

If you use ``pandera`` in the context of academic or industry research, please
consider citing the paper and/or software package.

`Paper <https://conference.scipy.org/proceedings/scipy2020/niels_bantilan.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    @InProceedings{ niels_bantilan-proc-scipy-2020,
      author    = { {N}iels {B}antilan },
      title     = { pandera: {S}tatistical {D}ata {V}alidation of {P}andas {D}ataframes },
      booktitle = { {P}roceedings of the 19th {P}ython in {S}cience {C}onference },
      pages     = { 116 - 124 },
      year      = { 2020 },
      editor    = { {M}eghann {A}garwal and {C}hris {C}alloway and {D}illon {N}iederhut and {D}avid {S}hupe },
      doi       = { 10.25080/Majora-342d178e-010 }
    }

Software Package
~~~~~~~~~~~~~~~~

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3385265.svg
    :target: https://doi.org/10.5281/zenodo.3385265
    :alt: software package

|

License and Credits
-------------------

``pandera`` is licensed under the `MIT license <https://github.com/pandera-dev/pandera/blob/master/LICENSE.txt>`_.
and is written and maintained by Niels Bantilan (niels@pandera.ci)


Indices and tables
==================

* :ref:`genindex`
