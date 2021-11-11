.. currentmodule:: pandera

.. _integrations:

Integrations
============

Pydantic
--------

*new in 0.8.0*

:class:`~pandera.model.SchemaModel` is fully compatible with
`pydantic <https://pydantic-docs.helpmanual.io/>`_.

.. testcode:: dataframe_schema_model

    import pandas as pd
    import pandera as pa
    from pandera.typing import DataFrame, Series
    import pydantic


    class SimpleSchema(pa.SchemaModel):
        str_col: Series[str] = pa.Field(unique=True)


    class PydanticModel(pydantic.BaseModel):
        x: int
        df: DataFrame[SimpleSchema]


    valid_df = pd.DataFrame({"str_col": ["hello", "world"]})
    PydanticModel(x=1, df=valid_df)

    invalid_df = pd.DataFrame({"str_col": ["hello", "hello"]})
    PydanticModel(x=1, df=invalid_df)

.. testoutput:: dataframe_schema_model

    Traceback (most recent call last):
    ...
    ValidationError: 1 validation error for PydanticModel
    df
    series 'str_col' contains duplicate values:
    1    hello
    Name: str_col, dtype: object (type=value_error)

Other pandera components are also compatible with pydantic:

- :class:`~pandera.model.SchemaModel`
- :class:`~pandera.schemas.DataFrameSchema`
- :class:`~pandera.schemas.SeriesSchema`
- :class:`~pandera.schema_components.MultiIndex`
- :class:`~pandera.schema_components.Column`
- :class:`~pandera.schema_components.Index`


Mypy
----

*new in 0.8.0*

Pandera integrates with mypy out of the box to provide static type-linting of
dataframes, relying on `pandas-stubs <https://github.com/VirtusLab/pandas-stubs>`__
for typing information.

.. ::

   Mypy static type-linting is supported for only pandas dataframes.

In the example below, we define a few schemas to see how type-linting with
pandera works.

.. literalinclude:: ../../tests/core/static/pandas_dataframe.py
    :lines: 8-27

The mypy linter will complain if the output type of the function body doesn't
match the function's return signature.

.. literalinclude:: ../../tests/core/static/pandas_dataframe.py
    :lines: 30-43

It'll also complain if the input type doesn't match the expected input type.
Note that we're using the :py:class:`pandera.typing.pandas.DataFrame` generic
type to define dataframes that are validated against the
:py:class:`~pandera.model.SchemaModel` type variable on initialization.

.. literalinclude:: ../../tests/core/static/pandas_dataframe.py
    :lines: 47-60


To make mypy happy with respect to the return type, you can either initialize
a dataframe of the expected type:

.. literalinclude:: ../../tests/core/static/pandas_dataframe.py
    :lines: 63-64

.. note::
    If you use the approach above with the :py:func:`~pandera.check_types`
    decorator, pandera will do its best to not to validate the dataframe twice
    if it's already been initialized with the
    ``DataFrame[Schema](**data)`` syntax.

Or use :py:func:`typing.cast` to indicate to mypy that the return value of
the function is of the correct type.

.. literalinclude:: ../../tests/core/static/pandas_dataframe.py
    :lines: 67-68


Limitations
^^^^^^^^^^^

An important caveat to static type-linting with pandera dataframe types is that,
since pandas dataframes are mutable objects, there's no way for ``mypy`` to
know whether a mutated instance of a
:py:class:`~pandera.model.SchemaModel`-typed dataframe has the correct
contents. Fortunately, we can simply rely on the :py:func:`~pandera.check_types`
decorator to verify that the output dataframe is valid.

Consider the examples below:

.. literalinclude:: ../../tests/core/static/pandas_dataframe.py
    :lines: 63-80

Even though the outputs of these functions are incorrect, mypy doesn't catch
the error during static type-linting but pandera will raise a
:py:class:`~pandera.errors.SchemaError` or :py:class:`~pandera.errors.SchemaErrors`
exception at runtime, depending on whether you're doing
:ref:`lazy validation<lazy_validation>` or not.

.. literalinclude:: ../../tests/core/static/pandas_dataframe.py
    :lines: 83-87
