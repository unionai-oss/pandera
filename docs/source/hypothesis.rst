.. pandera documentation for Hypothesis Testing

.. _hypothesis:

Hypothesis Testing
==================

``Column`` ``Hypothesis`` tests support testing different column so that assertions
can be made about the relationships between ``Column``\s.

``Hypothesis`` contains built in methods, which can be called as in this example:

.. code:: python

    import pandas as pd

    from pandera import Column, DataFrameSchema, Float, Check, String, Hypothesis

    from scipy import stats

    df = (
        pd.DataFrame({
            "height_in_feet": [6.5, 7, 6.1, 5.1, 4],
            "sex": ["M", "M", "F", "F", "F"]
        })
    )

    schema = DataFrameSchema({
        "height_in_feet": Column(Float, [
            Hypothesis.two_sample_ttest(
                sample1="M",
                sample2="F",
                groupby="sex",
                relationship="greater_than",
                alpha=0.05,
                equal_var=True),
        ]),
        "sex": Column(String)
    })

    schema.validate(df)

    #] SchemaError: <Schema Column: 'height_in_feet' type=float64> failed series validator 0: _check_fn


``Hypothesis`` also supports passing custom ``test``'s and ``relationship``'s.
The ``test`` function takes as input an one or multiple array-like objects
and should return a ``stat``, which is the test statistic, and ``pvalue`` for
assessing statistical significance. It also takes key-word arguments supplied
by the ``test_kwargs`` dict when initializing a ``Hypothesis`` object.

The ``relationship`` function should take all of the outputs of ``test`` as
positional arguments, in addition to key-word arguments supplied by the
``relationship_kwargs`` dict.

This enables the user to use non-built in functions. Here is an implementation
of the two-sample t-test that uses the
`scipy implementation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html>`_:

.. code:: python

    schema = DataFrameSchema({
        "height_in_feet": Column(Float, [
            Hypothesis(
                test=stats.ttest_ind,
                samples=["M", "F"],
                groupby="sex",
                relationship=lambda stat, pvalue, alpha=0.01: (
                    stat > 0 and pvalue / 2 < alpha
                ),
                relationship_kwargs={"alpha": 0.5}
            )
        ]),
        "sex": Column(String)
    })


Wide Hypotheses
---------------

``pandera`` is primarily designed to operate on long-form data (commonly known
as `tidy data <https://vita.had.co.nz/papers/tidy-data.pdf>`_), where each row
is an observation and columns are attributes associated with the observation.

However, ``pandera`` also supports hypothesis testing on wide-form data to
operate across columns in a ``DataFrame``.

For example, if you want to make assertions about ``height`` across two groups,
the tidy dataset and schema might look like this:

.. code:: python

    import pandas as pd
    from pandera import DataFrameSchema, Column, Hypothesis, Float, String

    df = pd.DataFrame({
        "height": [5.6, 6.4, 4.0, 7.1],
        "group": ["A", "B", "A", "B"],
    })

    schema = DataFrameSchema({
        "height": Column(
            Float, Hypothesis.two_sample_ttest(
                "A", "B",
                groupby="group",
                relationship="less_than",
                alpha=0.5
            )
        ),
        "group": Column(String, Check(lambda s: s.isin(["A", "B"])))
    })

    schema.validate(df)


The equivalent wide-form schema would look like this:

.. code:: python

    import pandas as pd
    from pandera import DataFrameSchema, Column, Hypothesis, Float

    df = pd.DataFrame({
        "height_A": [5.6, 4.0],
        "height_B": [6.4, 7.1],
    })

    schema = DataFrameSchema(
        columns={
            "height_A": Column(Float),
            "height_B": Column(Float),
        },
        # define checks at the DataFrameSchema-level
        checks=Hypothesis.two_sample_ttest(
            "height_A", "height_B",
            relationship="less_than",
            alpha=0.5
        )
    )

    schema.validate(df)

