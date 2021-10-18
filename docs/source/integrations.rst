.. currentmodule:: pandera

.. _integrations:

Integrations
============


Pydantic
--------

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
