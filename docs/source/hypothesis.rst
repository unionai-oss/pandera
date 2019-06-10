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
                groupby="sex",
                group1="M",
                group2="F",
                relationship="greater_than",
                alpha=0.05,
                equal_var=True),
        ]),
        "sex": Column(String)
    })

    schema.validate(df)

    #] SchemaError: <Schema Column: 'height_in_feet' type=float64> failed series validator 0: _check_fn


``Hypothesis`` also supports passing custom ``test``'s and ``relationship``'s. This
enables the user to use non-built in functions as follows:

.. code:: python

    schema = DataFrameSchema({
        "height_in_feet": Column(Float, [
            Hypothesis(
                test=stats.ttest_ind,
                groupby="sex",
                groups=["M", "F"],
                relationship="greater_than",
                relationship_kwargs={"alpha":0.5, "equal_var": True}),
        ]),
        "sex": Column(String)
    })
