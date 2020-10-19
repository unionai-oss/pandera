.. pandera documentation master file

Statistical Data Validation for Pandas
======================================

*A data validation library for scientists, engineers, and analysts seeking
correctness.*

``pandera`` provides a flexible and expressive API for performing data
validation on tidy (long-form) and wide data to make data processing pipelines
more readable and robust.

`pandas <http://pandas.pydata.org>`_ data structures contain information that
``pandera`` explicitly validates at runtime. This is useful in
production-critical data pipelines or reproducible research settings. With
``pandera``, you can:

#. :ref:`Check<checks>` the types and properties of columns in a
   ``pd.DataFrame`` or values in a ``pd.Series``.
#. Perform more complex statistical validation like
   :ref:`hypothesis testing<hypothesis>`.
#. Seamlessly integrate with existing data analysis/processing pipelines
   via :ref:`function decorators<decorators>`.
#. Define schema models with a :ref:`class-based API<schema_models>` with
   pydantic-style syntax and validate dataframes using the typing syntax.


Install
-------

Install with `pip`:

.. code:: bash

    pip install pandera


Or conda:

.. code:: bash

    conda install -c conda-forge pandera


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
        "column1": pa.Column(int, checks=pa.Check.less_than_or_equal_to(10)),
        "column2": pa.Column(float, checks=pa.Check.less_than(-1.2)),
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

Alternatively, you can pass the built-in python types that are supported by
pandas, or strings representing the
`legal pandas datatypes <https://pandas.pydata.org/docs/user_guide/basics.html#dtypes>`_:

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
        "str_column2": pa.Column("object"),
    })

For more details on data types, see :class:`~pandera.dtypes.PandasDtype`


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
    SchemaError: <Schema Column: 'column1' type=int> failed element-wise validator 0:
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

.. toctree::
   :maxdepth: 6
   :caption: Table of Contents
   :hidden:

   self
   dataframe_schemas
   series_schemas
   checks
   hypothesis
   decorators
   schema_inference
   schema_models
   lazy_validation
   API_reference

Indices and tables
==================

* :ref:`genindex`
