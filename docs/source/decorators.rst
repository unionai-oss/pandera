.. pandera documentation for check_input and check_output decorators

.. currentmodule:: pandera

.. _decorators:

Decorators for Pipeline Integration
===================================

If you have an existing data pipeline that uses pandas data structures,
you can use the :func:`~pandera.decorators.check_input` and :func:`~pandera.decorators.check_output` decorators
to easily check function arguments or returned variables from existing
functions.

Check Input
~~~~~~~~~~~

Validates input pandas DataFrame/Series before entering the wrapped
function.

.. testcode:: check_input_decorators

    import pandas as pd
    import pandera as pa

    from pandera import DataFrameSchema, Column, Check, check_input


    df = pd.DataFrame({
       "column1": [1, 4, 0, 10, 9],
       "column2": [-1.3, -1.4, -2.9, -10.1, -20.4],
    })

    in_schema = DataFrameSchema({
       "column1": Column(pa.Int,
                         Check(lambda x: 0 <= x <= 10, element_wise=True)),
       "column2": Column(pa.Float, Check(lambda x: x < -1.2)),
    })

    # by default, check_input assumes that the first argument is
    # dataframe/series.
    @check_input(in_schema)
    def preprocessor(dataframe):
        dataframe["column3"] = dataframe["column1"] + dataframe["column2"]
        return dataframe

    preprocessed_df = preprocessor(df)
    print(preprocessed_df)

.. testoutput:: check_input_decorators

       column1  column2  column3
    0        1     -1.3     -0.3
    1        4     -1.4      2.6
    2        0     -2.9     -2.9
    3       10    -10.1     -0.1
    4        9    -20.4    -11.4


You can also provide the argument name as a string

.. testcode:: check_input_decorators

    @check_input(in_schema, "dataframe")
    def preprocessor(dataframe):
        ...

Or an integer representing the index in the positional arguments.

.. testcode:: check_input_decorators

    @check_input(in_schema, 1)
    def preprocessor(foo, dataframe):
        ...


Check Output
~~~~~~~~~~~~

The same as ``check_input``, but this decorator checks the output
DataFrame/Series of the decorated function.

.. testcode:: check_output_decorators

    import pandas as pd
    import pandera as pa

    from pandera import DataFrameSchema, Column, Check, check_output


    preprocessed_df = pd.DataFrame({
       "column1": [1, 4, 0, 10, 9],
    })

    # assert that all elements in "column1" are zero
    out_schema = DataFrameSchema({
        "column1": Column(pa.Int, Check(lambda x: x == 0))
    })


    # by default assumes that the pandas DataFrame/Schema is the only output
    @check_output(out_schema)
    def zero_column_1(df):
        df["column1"] = 0
        return df


    # you can also specify in the index of the argument if the output is list-like
    @check_output(out_schema, 1)
    def zero_column_1_arg(df):
        df["column1"] = 0
        return "foobar", df


    # or the key containing the data structure to verify if the output is dict-like
    @check_output(out_schema, "out_df")
    def zero_column_1_dict(df):
        df["column1"] = 0
        return {"out_df": df, "out_str": "foobar"}


    # for more complex outputs, you can specify a function
    @check_output(out_schema, lambda x: x[1]["out_df"])
    def zero_column_1_custom(df):
        df["column1"] = 0
        return ("foobar", {"out_df": df})


    zero_column_1(preprocessed_df)
    zero_column_1_arg(preprocessed_df)
    zero_column_1_dict(preprocessed_df)
    zero_column_1_custom(preprocessed_df)
