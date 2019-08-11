.. pandera documentation for Checks

.. _checks:

Checks
======

Checking values within a column
-------------------------------

By default, ``Column`` ``Check``\ s are functions which expect a ``pd.series``
argument and should output a boolean or a boolean Series.



.. code:: python

   schema = DataFrameSchema({"column1": Column(Int, Check(lambda s: s <= 10))})

Multiple checks can be applied to a column as i n

.. code:: python

   schema = DataFrameSchema({
       "column2": Column(String, [
           Check(lambda s: s.str.startswith("value")),
           Check(lambda s: s.str.split("_", expand=True).shape[1] == 2)
       ]),
   })

Vectorized vs. Element-wise Checks
----------------------------------

By default, the functions passed into ``Check``\ s are expected to have
the following signature: ``pd.Series -> bool|pd.Series[bool]``. For the
``Check`` to pass, all of the elements in the boolean series must
evaluate to ``True``.

If you want to make atomic checks for each element in the Column, then
you can provide the ``element_wise=True`` keyword argument:

.. code:: python

   schema = DataFrameSchema({
       "a": Column(Int, [
           # a vectorized check that returns a bool
           Check(lambda s: s.mean() > 5, element_wise=False),
           # a vectorized check that returns a boolean series
           Check(lambda s: s > 0, element_wise=False),
           # an element-wise check that returns a bool
           Check(lambda x: x > 0, element_wise=True),
       ]),
   })

   df = pd.DataFrame({"a": [4, 4, 5, 6, 6, 7, 8, 9]})
   schema.validate(df)

By default ``element_wise=False`` so that you can take advantage of the
speed gains provided by the ``pandas.Series`` API by writing vectorized
checks.

.. _grouping:

Column Check Groups
-------------------

``Column`` ``Check``s support grouping by a different column so that
you can make assertions about subsets of the ``Column`` of interest.
This changes the function signature of the ``Check`` function so that
its input is a dict where keys are the group names and keys are subsets
of the ``Column`` series.

Specifying ``groupby`` as a column name, list of column names, or
callable changes the expected signature of the ``Check`` function
argument to ``dict[Any|tuple[Any], Series] -> bool|Series[bool]`` where
the dict keys are the discrete keys in the ``groupby`` columns.

.. code:: python

   schema = DataFrameSchema({
       "height_in_feet": Column(Float, [
           # groupby as a single column
           Check(lambda g: g[False].mean() > 6, groupby="age_less_than_20"),
           # define multiple groupby columns
           Check(lambda g: g[(True, "F")].sum() == 9.1,
                 groupby=["age_less_than_20", "sex"]),
           # groupby as a callable with signature (DataFrame) -> DataFrameGroupBy
           Check(lambda g: g[(False, "M")].median() == 6.75,
                 groupby=lambda df: (
                   df
                   .assign(age_less_than_15=lambda d: d["age"] < 15)
                   .groupby(["age_less_than_15", "sex"]))),
       ]),
       "age": Column(Int, Check(lambda s: s > 0)),
       "age_less_than_20": Column(Bool),
       "sex": Column(String, Check(lambda s: s.isin(["M", "F"])))
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

In the above example we define a ``DataFrameSchema`` with column checks
for ``height_in_feet`` using a single column, multiple columns, and a
more complex groupby function that creates a new column
``age_less_than_15`` on the fly.


Wide Checks
-----------

``pandera`` is primarily designed to operate on long-form data (commonly known
as `tidy data <https://vita.had.co.nz/papers/tidy-data.pdf>`_), where each row
is an observation and columns are attributes associated with the observation.

However, ``pandera`` also supports checks on wide-form data to operate across
columns in a ``DataFrame``.

For example, if you want to make assertions about ``height`` across two groups,
the tidy dataset and schema might look like this:

.. code:: python

    import pandas as pd
    from pandera import DataFrameSchema, Column, Check, Float, String

    df = pd.DataFrame({
        "height": [5.6, 6.4, 4.0, 7.1],
        "group": ["A", "B", "A", "B"],
    })

    schema = DataFrameSchema({
        "height": Column(
            Float,
            Check(lambda g: g["A"].mean() < g["B"].mean(), groupby="group")
        ),
        "group": Column(String)
    })

    schema.validate(df)


The equivalent wide-form schema would look like this:

.. code:: python

    import pandas as pd
    from pandera import DataFrameSchema, Column, Check, Float

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
        checks=Check(lambda df: df["height_A"].mean() < df["height_B"].mean())
    )

    schema.validate(df)
