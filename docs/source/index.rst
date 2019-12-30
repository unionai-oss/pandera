.. pandera documentation master file

.. image:: _static/pandera-logo.svg
  :width: 140px

Pandera
=======

A flexible and expressive `pandas <http://pandas.pydata.org>`_ validation library.


Why?
----

``pandas`` data structures hide a lot of information, and explicitly
validating them at runtime in production-critical or reproducible research
settings is a good idea. ``pandera`` enables users to:

#. :ref:`Check<checks>` the types and properties of columns in a
   ``pa.DataFrame`` or values in a ``pa.Series``.
#. Perform more complex statistical validation like
   :ref:`hypothesis testing<hypothesis>`.
#. Seamlessly integrate with existing data analysis/processing pipelines
   via :ref:`function decorators<decorators>`.

``pandera`` provides a flexible and expressive API for performing data
validation on tidy (long-form) and wide data to make data processing pipelines
more readable and robust.


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

    from pandera import Column, DataFrameSchema, Check


    # validate columns
    schema = DataFrameSchema({
        # the check function expects a series argument and should output a boolean
        # or a boolean Series.
        "column1": Column(pa.Int, Check(lambda s: s <= 10)),
        "column2": Column(pa.Float, Check(lambda s: s < -1.2)),
        # you can provide a list of validators
        "column3": Column(pa.String, [
            Check(lambda s: s.str.startswith("value")),
            Check(lambda s: s.str.split("_", expand=True).shape[1] == 2)
        ]),
    })

    df = pd.DataFrame({
        "column1": [1, 4, 0, 10, 9],
        "column2": [-1.3, -1.4, -2.9, -10.1, -20.4],
        "column3": ["value_1", "value_2", "value_3", "value_2", "value_1"],
    })

    validated_df = schema.validate(df)
    print(validated_df)

.. testoutput:: quick_start

       column1  column2  column3
    0        1     -1.3  value_1
    1        4     -1.4  value_2
    2        0     -2.9  value_3
    3       10    -10.1  value_2
    4        9    -20.4  value_1


Alternatively, you can pass strings representing the
`legal pandas datatypes <http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes>`_:

.. testcode:: quick_start

   schema = DataFrameSchema({
       "column1": Column("int64", Check(lambda s: s <= 10)),
       "column2": Column("float64", Check(lambda s: s < -1.2)),
       "column3": Column("object", [
           Check(lambda s: s.str.startswith("value")),
           Check(lambda s: s.str.split("_", expand=True).shape[1] == 2)
       ]),
   })


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

    simple_schema.validate(fail_check_df)


.. testoutput:: informative_errors

    Traceback (most recent call last):
    ...
    pandera.SchemaError: <Schema Column: 'column1' type=int64> failed element-wise validator 0:
    <lambda>: range checker [0, 10]
    failure cases:
                 index  count
    failure_case
    -20            [0]      1
     30            [3]      1


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
   :maxdepth: 2
   :caption: Contents:

   self
   dataframe_schemas
   series_schemas
   checks
   hypothesis
   decorators
   API

Indices and tables
==================

* :ref:`genindex`
