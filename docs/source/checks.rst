.. pandera documentation for Checks

.. _checks:

Checks
======

Checking column properties
--------------------------

:class:`~pandera.checks.Check` objects accept a function as a required
argument, which is expected to have the following signature:

``pd.Series -> Union[bool, pd.Series]``

For the :class:`~pandera.checks.Check` to pass, all of the elements in the
boolean series must evaluate to ``True``.


.. testcode:: checks

    import pandera as pa

    from pandera import Column, Check, DataFrameSchema

    schema = DataFrameSchema({"column1": Column(pa.Int, Check(lambda s: s <= 10))})


Multiple checks can be applied to a column:

.. testcode:: checks

  schema = DataFrameSchema({
      "column2": Column(pa.String, [
          Check(lambda s: s.str.startswith("value")),
          Check(lambda s: s.str.split("_", expand=True).shape[1] == 2)
      ]),
  })

Built-in Checks
---------------

For common validation tasks built-in checks are available in ``pandera``.
They are provided as factory methods in the :class:`~pandera.checks.Check`
class.

This way comparison operations, string validations and whitelisting or
blacklisting of allowed values can be done more easily:

.. testcode:: builtin_checks

  import pandera as pa
  from pandera import Column, Check, DataFrameSchema

  schema = DataFrameSchema({
      "small_values": Column(pa.Float, [Check.less_than(100)]),
      "one_to_three": Column(pa.Int, [Check.isin([1, 2, 3])]),
      "phone_number": Column(pa.String, [Check.str_matches(r'^[a-z0-9-]+$')]),
  })


Vectorized vs. Element-wise Checks
------------------------------------

By default, :class:`~pandera.checks.Check` objects operate on ``pd.Series``
objects. If you want to make atomic checks for each element in the Column, then
you can provide the ``element_wise=True`` keyword argument:

.. testcode:: vectorized_element_wise_checks

    import pandas as pd
    import pandera as pa

    from pandera import Column, Check, DataFrameSchema

    schema = DataFrameSchema({
        "a": Column(
            pa.Int,
            [
                # a vectorized check that returns a bool
                Check(lambda s: s.mean() > 5, element_wise=False),
                # a vectorized check that returns a boolean series
                Check(lambda s: s > 0, element_wise=False),
                # an element-wise check that returns a bool
                Check(lambda x: x > 0, element_wise=True),
            ]
        ),
    })
    df = pd.DataFrame({"a": [4, 4, 5, 6, 6, 7, 8, 9]})
    schema.validate(df)


``element_wise == False`` by default so that you can take advantage of the
speed gains provided by the ``pd.Series`` API by writing vectorized
checks.

.. _grouping:

Column Check Groups
-------------------

:class:`~pandera.schema_components.Column` checks support grouping by a
different column so that you can make assertions about subsets of the
:class:`~pandera.schema_components.Column` of interest. This changes the
function signature of the :class:`~pandera.checks.Check` function so that its
input is a dict where keys are the group names and values are subsets of the
:class:`~pandera.schema_components.Column` series.

Specifying ``groupby`` as a column name, list of column names, or
callable changes the expected signature of the :class:`~pandera.checks.Check`
function argument to
``Dict[Any, pd.Series] -> Union[bool, pd.Series]``
where the dict keys are the discrete keys in the ``groupby`` columns.

.. testcode:: column_check_groups

    import pandas as pd
    import pandera as pa

    from pandera import Column, Check, DataFrameSchema

    schema = DataFrameSchema({
        "height_in_feet": Column(
            pa.Float, [
                # groupby as a single column
                Check(lambda g: g[False].mean() > 6,
                      groupby="age_less_than_20"),
                # define multiple groupby columns
                Check(lambda g: g[(True, "F")].sum() == 9.1,
                      groupby=["age_less_than_20", "sex"]),
                # groupby as a callable with signature:
                # (DataFrame) -> DataFrameGroupBy
                Check(lambda g: g[(False, "M")].median() == 6.75,
                      groupby=lambda df: (
                        df
                        .assign(age_less_than_15=lambda d: d["age"] < 15)
                        .groupby(["age_less_than_15", "sex"]))),
            ]),
        "age": Column(pa.Int, Check(lambda s: s > 0)),
        "age_less_than_20": Column(pa.Bool),
        "sex": Column(pa.String, Check(lambda s: s.isin(["M", "F"])))
    })

    df = (
        pd.DataFrame({
            "height_in_feet": [6.5, 7, 6.1, 5.1, 4],
            "age": [25, 30, 21, 18, 13],
            "sex": ["M", "M", "F", "F", "F"]
        })
        .assign(age_less_than_20=lambda x: x["age"] < 20)
    )

    schema.validate(df)

In the above example we define a :class:`~pandera.schemas.DataFrameSchema` with
column checks for ``height_in_feet`` using a single column, multiple columns,
and a more complex groupby function that creates a new column
``age_less_than_15`` on the fly.


Wide Checks
-----------

``pandera`` is primarily designed to operate on long-form data (commonly known
as `tidy data <https://vita.had.co.nz/papers/tidy-data.pdf>`_), where each row
is an observation and each column is an attribute associated with an
observation.

However, ``pandera`` also supports checks on wide-form data to operate across
columns in a ``DataFrame``.

For example, if you want to make assertions about ``height`` across two groups,
the tidy dataset and schema might look like this:

.. testcode:: wide_checks

    import pandas as pd
    import pandera as pa

    from pandera import DataFrameSchema, Column, Check

    df = pd.DataFrame({
        "height": [5.6, 6.4, 4.0, 7.1],
        "group": ["A", "B", "A", "B"],
    })

    schema = DataFrameSchema({
        "height": Column(
            pa.Float,
            Check(lambda g: g["A"].mean() < g["B"].mean(), groupby="group")
        ),
        "group": Column(pa.String)
    })

    schema.validate(df)


The equivalent wide-form schema would look like this:

.. testcode:: wide_checks

    df = pd.DataFrame({
        "height_A": [5.6, 4.0],
        "height_B": [6.4, 7.1],
    })

    schema = DataFrameSchema(
        columns={
            "height_A": Column(pa.Float),
            "height_B": Column(pa.Float),
        },
        # define checks at the DataFrameSchema-level
        checks=Check(lambda df: df["height_A"].mean() < df["height_B"].mean())
    )

    schema.validate(df)


Raise UserWarning on Check Failure
----------------------------------

In some cases, you might want to raise a ``UserWarning`` and continue execution
of your program. The ``Check`` and ``Hypothesis`` classes and their built-in
methods all support the keyword argument ``raise_warning``, which is ``False``
by default. If set to ``True``, the check will raise a warning instead of
throwing a ``SchemaError``.

.. warning::
    Use this feature carefully! If the check is for informational purposes and
    not critical for data integrity then use ``raise_warning=True``. However,
    if the assumptions expressed in a ``Check`` is a necessary condition to
    considering your data as valid, do not set this option to true.

One scenario where you'd want to do this would be in a data pipeline that
does some preprocessing, checks for normality in certain columns, and writes
the resulting dataset to a table. In this case, you want to see if your
normality assumptions are not fulfilled by certain columns, but you still
want the resulting table for further analysis.

.. testcode:: check_raise_warning

    import warnings

    import numpy as np
    import pandas as pd
    import pandera as pa

    from scipy.stats import normaltest
    from pandera import DataFrameSchema, Column, Hypothesis


    df = pd.DataFrame({
        "var1": np.random.normal(loc=5, scale=2, size=100),
        "var2": np.random.uniform(low=0, high=10, size=100),
        "var3": np.random.uniform(low=10, high=20, size=100),
        "var4": np.random.uniform(low=0, high=50, size=100),
    })

    normal_check = Hypothesis(
        test=normaltest,
        samples="normal_variable",
        # null hypotheses: sample comes from a normal distribution. The
        # relationship function checks if we cannot reject the null hypothesis,
        # i.e. the p-value is greater or equal to alpha.
        relationship=lambda stat, pvalue, alpha=0.05: (
            pvalue >= alpha
        ),
        error="normality test",
        raise_warning=True,
    )

    schema = DataFrameSchema(
        columns={
            "var1": Column(checks=normal_check),
            "var2": Column(checks=normal_check),
            "var3": Column(checks=normal_check),
            "var4": Column(checks=normal_check),
        }
    )

    # catch and print warnings
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        validated_df = schema(df)
        for warning in caught_warnings:
            print(warning.message)


.. testoutput:: check_raise_warning

    <Schema Column: 'var2' type=None> failed series validator 0:
    <Check _hypothesis_check: normality test>
    <Schema Column: 'var3' type=None> failed series validator 0:
    <Check _hypothesis_check: normality test>
    <Schema Column: 'var4' type=None> failed series validator 0:
    <Check _hypothesis_check: normality test>
